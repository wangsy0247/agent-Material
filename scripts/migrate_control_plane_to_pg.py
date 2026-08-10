#!/usr/bin/env python
"""Spec V2 第 13 章阶段 1：现有系统状态 → PG 控制面迁移脚本。

迁移内容：
1. ensure_schema 建控制面 5 表（幂等）。
2. 从 langchain_pg_embedding 按 source 分组扫描存量数据（small/large 两个
   collection），为每个 source 生成确定性 document_id 与 legacy version_id，
   并**回写 metadata**：给存量行补 document_id / version_id / chunk_id
   （已有这三个键的行跳过 —— 幂等可重入）。
3. rag_documents / rag_document_versions 建行：legacy 版本 status='active'，
   parent_count / child_count 来自 SQL 统计，documents.active_version_id
   指向 legacy 版本。
4. 读 SQLite rag_queue.db 的 file_states（只读），迁入 rag_file_states；
   对不上 document 的孤立行记入报告、不导入。
5. rag_kb_state 初始化：active_revision = active 版本总数，bm25_revision = 0
   （强制查询服务下次启动全量重建 BM25）。

确定性 ID 方案（重入安全的关键）：
- document_id = versioning.compute_document_id(source, kb_id)
  （source 经 normalize_source_path 规范化，与未来扫描/入库生成的值一致）。
- version_id = "legacy-" + sha256(该 source 全部子块文本按 uuid 排序拼接)[:16]。
- 子块 chunk_id = sha256(f"{document_id}:{version_id}:legacy:{行内序号}")[:32]，
  行内序号按 uuid 排序 —— 中途崩溃重跑时对同一批行生成相同 ID。
- 父块 chunk_id = 其既有 legacy parent_id（保持 parent/child 关联一致）。

用法：
    # 预演（只读，不写任何库）：
    python scripts/migrate_control_plane_to_pg.py --dry-run
    # 真跑（先按 Spec 13 阶段 0 冻结写入并备份）：
    python scripts/migrate_control_plane_to_pg.py
"""

import argparse
import hashlib
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "packages"))

# 规避已知 C 扩展加载顺序冲突：sqlalchemy 先于 psycopg 导入
# （与 backend/tests/test_pg_worker.py 相同的环境坑）。
import sqlalchemy  # noqa: F401,E402
import psycopg  # noqa: E402
import yaml  # noqa: E402

from agent_material.rag_engine import versioning  # noqa: E402
from agent_material.rag_engine.db.pg.schema import (  # noqa: E402
    ensure_schema,
    normalize_conn_str,
)

_LEGACY_ID_LEN = 32


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def compute_legacy_version_id(child_texts: list[str]) -> tuple[str, str]:
    """由全部子块文本（已按 uuid 排序）生成 legacy version_id 与 content_hash。"""
    content_hash = _sha256_text("\n".join(child_texts))
    return f"legacy-{content_hash[:16]}", content_hash


def compute_legacy_chunk_id(document_id: str, version_id: str, row_index: int) -> str:
    return _sha256_text(f"{document_id}:{version_id}:legacy:{row_index}")[:_LEGACY_ID_LEN]


# ── 存量数据扫描 ──────────────────────────────────────────────


def group_sources(conn, small_collection: str, large_collection: str) -> dict[str, dict]:
    """按 source 分组统计两个 collection 的存量行数（返回 {source: 统计}）。"""
    stats: dict[str, dict] = {}
    for collection, key in ((small_collection, "child_count"), (large_collection, "parent_count")):
        rows = conn.execute(
            """
            SELECT e.cmetadata->>'source' AS source, COUNT(*) AS cnt
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
            WHERE c.name = %s AND e.cmetadata->>'source' IS NOT NULL
            GROUP BY 1
            """,
            (collection,),
        ).fetchall()
        for source, cnt in rows:
            stats.setdefault(source, {"child_count": 0, "parent_count": 0})[key] = cnt
    return stats


def fetch_rows(conn, collection: str, source: str) -> list[tuple]:
    """取某 source 在某 collection 的全部行（uuid 排序，保证重入序号稳定）。"""
    return conn.execute(
        """
        SELECT e.uuid, e.document, e.cmetadata
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE c.name = %s AND e.cmetadata->>'source' = %s
        ORDER BY e.uuid
        """,
        (collection, source),
    ).fetchall()


def _needs_backfill(metadata: dict) -> bool:
    """已有 document_id/version_id/chunk_id 三个键的行跳过（幂等）。"""
    return not all(k in metadata for k in ("document_id", "version_id", "chunk_id"))


def backfill_metadata(
    conn,
    dry_run: bool,
    source_stats: dict[str, dict],
    small_collection: str,
    large_collection: str,
    kb_id: str,
) -> list[dict]:
    """回写存量行 metadata，返回每个 source 的迁移计划/结果列表。"""
    plans: list[dict] = []
    total = len(source_stats)
    for i, (source, counts) in enumerate(sorted(source_stats.items()), 1):
        normalized = versioning.normalize_source_path(source)
        document_id = versioning.compute_document_id(source, kb_id=kb_id)

        children = fetch_rows(conn, small_collection, source)
        version_id, content_hash = compute_legacy_version_id(
            [row[1] or "" for row in children]
        )

        # 子块：行内序号按 uuid 排序（含已回写行，保证崩溃重跑序号稳定）
        child_updates = []
        for idx, (row_uuid, _doc, metadata) in enumerate(children):
            if not _needs_backfill(metadata):
                continue
            child_updates.append(
                (
                    document_id,
                    version_id,
                    compute_legacy_chunk_id(document_id, version_id, idx),
                    str(row_uuid),
                )
            )

        # 父块：chunk_id 沿用既有 legacy parent_id（无 parent_id 时退化为序号 hash）
        parents = fetch_rows(conn, large_collection, source)
        parent_updates = []
        for idx, (row_uuid, _doc, metadata) in enumerate(parents):
            if not _needs_backfill(metadata):
                continue
            chunk_id = metadata.get("parent_id") or compute_legacy_chunk_id(
                document_id, version_id, idx
            )
            parent_updates.append((document_id, version_id, chunk_id, str(row_uuid)))

        if not dry_run:
            with conn.transaction():
                for updates, collection in (
                    (child_updates, small_collection),
                    (parent_updates, large_collection),
                ):
                    if not updates:
                        continue
                    with conn.cursor() as cur:
                        cur.executemany(
                            """
                            UPDATE langchain_pg_embedding e
                            SET cmetadata = e.cmetadata
                                || jsonb_build_object(
                                    'document_id', %s::text,
                                    'version_id',  %s::text,
                                    'chunk_id',    %s::text)
                            FROM langchain_pg_collection c
                            WHERE e.collection_id = c.uuid
                              AND c.name = %s
                              AND e.uuid = %s::uuid
                              AND NOT (e.cmetadata ? 'document_id'
                                       AND e.cmetadata ? 'version_id'
                                       AND e.cmetadata ? 'chunk_id')
                            """,
                            [
                                (doc_id, ver_id, chunk_id, collection, row_uuid)
                                for doc_id, ver_id, chunk_id, row_uuid in updates
                            ],
                        )

        plan = {
            "source": normalized,
            "source_filename": os.path.basename(normalized),
            "document_id": document_id,
            "version_id": version_id,
            "content_hash": content_hash,
            "child_count": counts["child_count"],
            "parent_count": counts["parent_count"],
            "children_updated": len(child_updates),
            "parents_updated": len(parent_updates),
        }
        plans.append(plan)
        print(
            f"[{i}/{total}] {plan['source_filename'][:60]} | "
            f"doc={document_id[:8]} ver={version_id[:20]} | "
            f"子块 {counts['child_count']}(回写 {len(child_updates)}) "
            f"父块 {counts['parent_count']}(回写 {len(parent_updates)})"
        )
    return plans


# ── 控制面建行 ────────────────────────────────────────────────


def upsert_documents_and_versions(conn, dry_run: bool, plans: list[dict], kb_id: str) -> None:
    """rag_documents / rag_document_versions 建行（幂等：ON CONFLICT DO NOTHING）。"""
    if dry_run:
        print(f"[计划] rag_documents 建行 {len(plans)} 条，rag_document_versions 建行 {len(plans)} 条")
        return
    with conn.transaction():
        for p in plans:
            conn.execute(
                """
                INSERT INTO rag_documents
                    (document_id, kb_id, source, source_filename, status, active_version_id)
                VALUES (%s, %s, %s, %s, 'active', %s)
                ON CONFLICT DO NOTHING
                """,
                (p["document_id"], kb_id, p["source"], p["source_filename"], p["version_id"]),
            )
            conn.execute(
                """
                INSERT INTO rag_document_versions
                    (document_id, version_id, content_hash, status,
                     parent_count, child_count, validated_at, activated_at)
                VALUES (%s, %s, %s, 'active', %s, %s, now(), now())
                ON CONFLICT (document_id, version_id) DO NOTHING
                """,
                (
                    p["document_id"],
                    p["version_id"],
                    p["content_hash"],
                    p["parent_count"],
                    p["child_count"],
                ),
            )


def migrate_file_states(
    conn, dry_run: bool, plans: list[dict], sqlite_path: str, kb_id: str
) -> tuple[int, list[str]]:
    """SQLite file_states（只读）→ rag_file_states。返回 (导入数, 孤立 source 列表)。"""
    if not os.path.exists(sqlite_path):
        print(f"[警告] 未找到 SQLite 文件 {sqlite_path}，跳过 file_states 迁移")
        return 0, []

    doc_by_source = {p["source"]: p for p in plans}
    imported = 0
    orphans: list[str] = []

    lite = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        rows = lite.execute(
            "SELECT file_path, md5_hash, file_mtime, file_size, last_indexed_at, status "
            "FROM file_states"
        ).fetchall()
    finally:
        lite.close()

    for file_path, md5_hash, file_mtime, file_size, last_indexed_at, status in rows:
        normalized = versioning.normalize_source_path(file_path)
        plan = doc_by_source.get(normalized)
        if plan is None:
            # 对不上 document 的孤立行：记入报告，不导入
            orphans.append(normalized)
            continue
        pg_status = "deleted" if status == "deleted" else "active"
        if dry_run:
            imported += 1
            continue
        conn.execute(
            """
            INSERT INTO rag_file_states
                (source, document_id, content_hash, file_mtime, file_size,
                 status, active_version_id, last_indexed_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::timestamptz, now())
            ON CONFLICT (source) DO NOTHING
            """,
            (
                normalized,
                plan["document_id"],
                md5_hash,
                file_mtime,
                file_size,
                pg_status,
                plan["version_id"],
                last_indexed_at,
            ),
        )
        imported += 1
    return imported, orphans


def init_kb_state(conn, dry_run: bool, plans: list[dict], kb_id: str) -> int:
    """active_revision = active 版本总数；bm25_revision = 0（强制全量重建 BM25）。"""
    active_total = len(plans)
    if dry_run:
        print(
            f"[计划] rag_kb_state: kb_id={kb_id} "
            f"active_revision={active_total} bm25_revision=0"
        )
        return active_total
    conn.execute(
        """
        INSERT INTO rag_kb_state (kb_id, active_revision, bm25_revision)
        VALUES (%s, %s, 0)
        ON CONFLICT (kb_id) DO UPDATE SET
            active_revision = excluded.active_revision,
            bm25_revision = 0,
            updated_at = now()
        """,
        (kb_id, active_total),
    )
    return active_total


# ── 迁移后一致性校验 ──────────────────────────────────────────


def verify(conn, small_collection: str, large_collection: str, kb_id: str) -> dict:
    """真跑后的只读校验：返回各项计数供报告。"""
    result = {}
    for collection, key in ((small_collection, "small"), (large_collection, "large")):
        result[f"{key}_missing_version_meta"] = conn.execute(
            """
            SELECT COUNT(*)
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
            WHERE c.name = %s
              AND NOT (e.cmetadata ? 'document_id'
                       AND e.cmetadata ? 'version_id'
                       AND e.cmetadata ? 'chunk_id')
            """,
            (collection,),
        ).fetchone()[0]
    result["documents"] = conn.execute(
        "SELECT COUNT(*) FROM rag_documents WHERE kb_id = %s", (kb_id,)
    ).fetchone()[0]
    result["active_versions"] = conn.execute(
        """
        SELECT COUNT(*) FROM rag_document_versions v
        JOIN rag_documents d ON v.document_id = d.document_id
        WHERE d.kb_id = %s AND v.status = 'active'
        """,
        (kb_id,),
    ).fetchone()[0]
    result["file_states"] = conn.execute(
        "SELECT COUNT(*) FROM rag_file_states"
    ).fetchone()[0]
    result["kb_state"] = conn.execute(
        "SELECT active_revision, bm25_revision FROM rag_kb_state WHERE kb_id = %s",
        (kb_id,),
    ).fetchone()
    return result


# ── 入口 ──────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="PG 控制面迁移（Spec V2 13 阶段 1）")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不写库")
    parser.add_argument("--conn-str", default=None, help="PG 连接串（默认读配置）")
    parser.add_argument("--kb-id", default=None, help="知识库标识（默认 character_name）")
    parser.add_argument("--sqlite", default=str(ROOT / "rag_queue.db"), help="SQLite 控制面路径")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "config" / "vector_store_config.yaml").read_text())
    conn_str = normalize_conn_str(args.conn_str or config["connection_string"])
    kb_id = args.kb_id or config.get("character_name", "agent")
    mg = config.get("multi_granularity", {})
    small_collection = f"{kb_id}{mg.get('small_collection_suffix', '_small')}"
    large_collection = f"{kb_id}{mg.get('large_collection_suffix', '_large')}"

    mode = "DRY-RUN（只读）" if args.dry_run else "真跑（写库）"
    print(f"=== PG 控制面迁移 | 模式: {mode} | kb={kb_id} | "
          f"collections={small_collection}/{large_collection} ===")

    if not args.dry_run:
        ensure_schema(conn_str)
        print("[1/5] 控制面表已就绪（ensure_schema）")
    else:
        print("[1/5] DRY-RUN 跳过 ensure_schema（真跑时幂等建表）")

    with psycopg.connect(conn_str) as conn:
        source_stats = group_sources(conn, small_collection, large_collection)
        print(f"[2/5] 发现 {len(source_stats)} 个 source "
              f"（子块 {sum(s['child_count'] for s in source_stats.values())} / "
              f"父块 {sum(s['parent_count'] for s in source_stats.values())}）")

        print("[3/5] 回写存量 metadata（按 source 分批）...")
        plans = backfill_metadata(
            conn, args.dry_run, source_stats, small_collection, large_collection, kb_id
        )

        print("[4/5] 控制面建 documents / versions / file_states ...")
        upsert_documents_and_versions(conn, args.dry_run, plans, kb_id)
        imported, orphans = migrate_file_states(
            conn, args.dry_run, plans, args.sqlite, kb_id
        )
        active_total = init_kb_state(conn, args.dry_run, plans, kb_id)
        if args.dry_run:
            conn.rollback()  # dry-run 双保险：任何意外写入都不落库
        else:
            conn.commit()

        # ── 报告 ──
        print("\n===== 迁移报告 =====")
        print(f"source 总数:        {len(plans)}")
        print(f"子块回写:           {sum(p['children_updated'] for p in plans)}")
        print(f"父块回写:           {sum(p['parents_updated'] for p in plans)}")
        print(f"file_states 导入:   {imported}")
        print(f"file_states 孤立:   {len(orphans)}")
        for o in orphans:
            print(f"  [孤立] {o}")
        zero_child = [p for p in plans if p["child_count"] == 0]
        zero_parent = [p for p in plans if p["parent_count"] == 0]
        if zero_child or zero_parent:
            print(f"[异常] 无子块 source: {len(zero_child)}，无父块 source: {len(zero_parent)}")
            for p in zero_child + zero_parent:
                print(f"  {p['source']}")
        print(f"kb_state:           active_revision={active_total} bm25_revision=0")

        if not args.dry_run:
            check = verify(conn, small_collection, large_collection, kb_id)
            print("\n===== 迁移后校验 =====")
            for k, v in check.items():
                print(f"{k}: {v}")
            ok = (
                check["small_missing_version_meta"] == 0
                and check["large_missing_version_meta"] == 0
                and check["active_versions"] == len(plans)
            )
            print("校验结果:", "通过" if ok else "存在差异，请检查上面的计数")
            return 0 if ok else 1

        print("\nDRY-RUN 完成，未写入任何数据。")
        print("真跑命令（先按 Spec 13 阶段 0 冻结写入并备份 PG 与 rag_queue.db）：")
        print("    python scripts/migrate_control_plane_to_pg.py")
        return 0


if __name__ == "__main__":
    sys.exit(main())
