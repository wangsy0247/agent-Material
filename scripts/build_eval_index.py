#!/usr/bin/env python3
"""从已验证的活跃文档克隆一个隔离、可复现的评测索引。

脚本复用源 collection 中已经计算好的 embedding，不调用 MinerU 或 Embedding
API。目标 collection、document/version/chunk 标识全部独立，线上 ``agent``
知识库不会被修改。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

import psycopg
from psycopg.types.json import Json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "packages"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from agent_material.rag_engine.db.pg.schema import (  # noqa: E402
    ensure_schema,
    normalize_conn_str,
)
from agent_material.rag_engine.versioning import (  # noqa: E402
    compute_content_hash,
    compute_document_id,
    compute_version_id,
)
from agent_material.utils.config_handler import vector_store_config  # noqa: E402


def _connection_string() -> str:
    configured = os.environ.get("EVAL_PGVECTOR_URL") or vector_store_config.get(
        "connection_string", ""
    )
    if not configured:
        raise RuntimeError("缺少 EVAL_PGVECTOR_URL/PGVECTOR_URL")
    return normalize_conn_str(configured)


def _delete_existing_target(conn, target_prefix: str, target_kb: str) -> None:
    """只删除显式命名的评测 collection 和对应控制面记录。"""
    collection_names = (f"{target_prefix}_small", f"{target_prefix}_large")
    conn.execute(
        "DELETE FROM langchain_pg_collection WHERE name IN (%s, %s)",
        collection_names,
    )
    doc_ids = [
        row[0]
        for row in conn.execute(
            "SELECT document_id FROM rag_documents WHERE kb_id = %s", (target_kb,)
        ).fetchall()
    ]
    if doc_ids:
        conn.execute(
            "DELETE FROM rag_document_versions WHERE document_id = ANY(%s)",
            (doc_ids,),
        )
        conn.execute(
            "DELETE FROM rag_file_states WHERE document_id = ANY(%s)", (doc_ids,)
        )
        conn.execute("DELETE FROM rag_documents WHERE document_id = ANY(%s)", (doc_ids,))
    conn.execute("DELETE FROM rag_kb_state WHERE kb_id = %s", (target_kb,))


def build(
    *,
    conn_str: str,
    source_prefix: str,
    source_suffix: str,
    target_prefix: str,
    target_kb: str,
    target_source: str,
    source_file: Path,
    replace: bool,
) -> dict:
    ensure_schema(conn_str)
    content_hash = (
        compute_content_hash(source_file.read_bytes())
        if source_file.exists()
        else hashlib.sha256(source_suffix.encode()).hexdigest()
    )
    document_id = compute_document_id(target_source, target_kb)
    version_id = compute_version_id(content_hash)

    source_small = f"{source_prefix}_small"
    source_large = f"{source_prefix}_large"
    target_small = f"{target_prefix}_small"
    target_large = f"{target_prefix}_large"

    with psycopg.connect(conn_str) as conn:
        source_rows = conn.execute(
            """
            SELECT DISTINCT e.cmetadata->>'document_id',
                            e.cmetadata->>'version_id',
                            e.cmetadata->>'source'
            FROM langchain_pg_collection c
            JOIN langchain_pg_embedding e ON e.collection_id = c.uuid
            WHERE c.name = %s
              AND e.cmetadata->>'is_active' = 'true'
              AND e.cmetadata->>'source' LIKE %s
            ORDER BY 3
            """,
            (source_small, f"%{source_suffix}"),
        ).fetchall()
        if len(source_rows) != 1:
            raise RuntimeError(
                f"源文档必须唯一，实际命中 {len(source_rows)} 个: {source_rows[:5]}"
            )
        source_document_id, source_version_id, original_source = source_rows[0]

        existing = conn.execute(
            "SELECT name, COUNT(*) FROM langchain_pg_collection "
            "WHERE name IN (%s, %s) GROUP BY name",
            (target_small, target_large),
        ).fetchall()
        if existing and not replace:
            raise RuntimeError(f"目标评测索引已存在，使用 --replace 重建: {existing}")
        if replace:
            _delete_existing_target(conn, target_prefix, target_kb)

        small_collection_id = uuid.uuid4()
        large_collection_id = uuid.uuid4()
        collection_meta = {
            "kb_id": target_kb,
            "purpose": "evaluation",
            "source_document_id": source_document_id,
            "source_version_id": source_version_id,
        }
        conn.execute(
            "INSERT INTO langchain_pg_collection (name, cmetadata, uuid) "
            "VALUES (%s, %s, %s), (%s, %s, %s)",
            (
                target_small,
                Json(collection_meta),
                small_collection_id,
                target_large,
                Json(collection_meta),
                large_collection_id,
            ),
        )

        common_metadata = json.dumps(
            {
                "document_id": document_id,
                "version_id": version_id,
                "source": target_source,
                "source_filename": source_file.name,
                "original_source": original_source,
                "is_active": True,
                "kb_id": target_kb,
                "evaluation_index": True,
            },
            ensure_ascii=False,
        )

        small_count = conn.execute(
            """
            INSERT INTO langchain_pg_embedding
                (collection_id, embedding, document, cmetadata, custom_id, uuid, id)
            SELECT
                %s,
                e.embedding,
                e.document,
                COALESCE(e.cmetadata, '{}'::jsonb)
                    || %s::jsonb
                    || jsonb_build_object(
                        'parent_id', md5(%s || ':parent:' || (e.cmetadata->>'parent_id')),
                        'chunk_id', md5(%s || ':chunk:' || (e.cmetadata->>'chunk_id'))
                    ),
                md5(%s || ':custom:small:' || COALESCE((e.cmetadata->>'chunk_id'), e.id)),
                gen_random_uuid(),
                md5(%s || ':id:small:' || COALESCE((e.cmetadata->>'chunk_id'), e.id))
            FROM langchain_pg_collection c
            JOIN langchain_pg_embedding e ON e.collection_id = c.uuid
            WHERE c.name = %s
              AND e.cmetadata->>'is_active' = 'true'
              AND e.cmetadata->>'document_id' = %s
              AND e.cmetadata->>'version_id' = %s
            RETURNING 1
            """,
            (
                small_collection_id,
                common_metadata,
                document_id,
                document_id,
                target_prefix,
                target_prefix,
                source_small,
                source_document_id,
                source_version_id,
            ),
        ).rowcount

        large_count = conn.execute(
            """
            INSERT INTO langchain_pg_embedding
                (collection_id, embedding, document, cmetadata, custom_id, uuid, id)
            SELECT
                %s,
                e.embedding,
                e.document,
                COALESCE(e.cmetadata, '{}'::jsonb)
                    || %s::jsonb
                    || jsonb_build_object(
                        'parent_id', md5(%s || ':parent:' || (e.cmetadata->>'parent_id'))
                    ),
                md5(%s || ':custom:large:' || COALESCE((e.cmetadata->>'parent_id'), e.id)),
                gen_random_uuid(),
                md5(%s || ':id:large:' || COALESCE((e.cmetadata->>'parent_id'), e.id))
            FROM langchain_pg_collection c
            JOIN langchain_pg_embedding e ON e.collection_id = c.uuid
            WHERE c.name = %s
              AND e.cmetadata->>'is_active' = 'true'
              AND e.cmetadata->>'document_id' = %s
              AND e.cmetadata->>'version_id' = %s
            RETURNING 1
            """,
            (
                large_collection_id,
                common_metadata,
                document_id,
                target_prefix,
                target_prefix,
                source_large,
                source_document_id,
                source_version_id,
            ),
        ).rowcount

        conn.execute(
            """
            INSERT INTO rag_documents
                (document_id, kb_id, source, source_filename, status, active_version_id)
            VALUES (%s, %s, %s, %s, 'active', %s)
            """,
            (document_id, target_kb, target_source, source_file.name, version_id),
        )
        conn.execute(
            """
            INSERT INTO rag_document_versions
                (document_id, version_id, content_hash, status, parent_count,
                 child_count, validated_at, activated_at)
            VALUES (%s, %s, %s, 'active', %s, %s, now(), now())
            """,
            (document_id, version_id, content_hash, large_count, small_count),
        )
        conn.execute(
            """
            INSERT INTO rag_file_states
                (source, document_id, content_hash, file_size, status,
                 active_version_id, last_indexed_at)
            VALUES (%s, %s, %s, %s, 'active', %s, now())
            """,
            (
                target_source,
                document_id,
                content_hash,
                source_file.stat().st_size if source_file.exists() else None,
                version_id,
            ),
        )
        conn.execute(
            """
            INSERT INTO rag_kb_state (kb_id, active_revision, bm25_revision)
            VALUES (%s, 1, 0)
            """,
            (target_kb,),
        )
        conn.commit()

    return {
        "target_kb": target_kb,
        "target_collections": [target_small, target_large],
        "target_source": target_source,
        "document_id": document_id,
        "version_id": version_id,
        "content_hash": content_hash,
        "child_count": small_count,
        "parent_count": large_count,
        "source_collection_prefix": source_prefix,
        "source_document_id": source_document_id,
        "source_version_id": source_version_id,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-prefix", default="agent")
    parser.add_argument("--source-suffix", default="abacus.pdf")
    parser.add_argument("--target-prefix", default="agent_eval")
    parser.add_argument("--target-kb", default="agent_eval")
    parser.add_argument("--target-source", default="eval://abacus.pdf")
    parser.add_argument("--source-file", type=Path, default=ROOT / "data" / "abacus.pdf")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = build(
        conn_str=_connection_string(),
        source_prefix=args.source_prefix,
        source_suffix=args.source_suffix,
        target_prefix=args.target_prefix,
        target_kb=args.target_kb,
        target_source=args.target_source,
        source_file=args.source_file.resolve(),
        replace=args.replace,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
