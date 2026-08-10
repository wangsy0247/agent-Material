"""PostgreSQL 控制面集成测试（Spec V2 阶段 1）。

使用真实 PG（127.0.0.1:5433，连接串读自 config/vector_store_config.yaml），
table_prefix="test_rag_" 隔离测试表；测试前后 DROP 全部 test_rag_* 表，
测试 collection（test_ctrl_small/test_ctrl_large）用后清理。
PG 不可达时整个模块 skip。
"""

import threading
import uuid
from pathlib import Path

import psycopg
import pytest
import yaml
from psycopg.types.json import Jsonb

from agent_material.rag_engine.db.pg import (
    FileStateRepository,
    KbStateRepository,
    TaskRepository,
    VersionRepository,
    drop_schema,
    ensure_schema,
    t,
)
from agent_material.rag_engine.db.pg.schema import BASE_TABLES, normalize_conn_str

ROOT = Path(__file__).resolve().parents[2]
CONN_STR = normalize_conn_str(
    yaml.safe_load((ROOT / "config" / "vector_store_config.yaml").read_text())[
        "connection_string"
    ]
)
PREFIX = "test_rag_"
SMALL = "test_ctrl_small"
LARGE = "test_ctrl_large"

TASKS = t("rag_ingest_tasks", PREFIX)
VERSIONS = t("rag_document_versions", PREFIX)


def _pg_available() -> bool:
    try:
        with psycopg.connect(CONN_STR, connect_timeout=3):
            return True
    except Exception:
        return False


def _cleanup_collections(conn_str: str) -> None:
    with psycopg.connect(conn_str) as conn:
        conn.execute(
            r"""
            DELETE FROM langchain_pg_embedding
            WHERE collection_id IN (
                SELECT uuid FROM langchain_pg_collection
                WHERE name LIKE 'test\_ctrl\_%'
            )
            """
        )
        conn.execute(
            r"DELETE FROM langchain_pg_collection WHERE name LIKE 'test\_ctrl\_%'"
        )
        conn.commit()


@pytest.fixture(scope="module")
def pg():
    if not _pg_available():
        pytest.skip("PostgreSQL 127.0.0.1:5433 不可达")
    drop_schema(CONN_STR, PREFIX)  # 防上次残留
    _cleanup_collections(CONN_STR)
    ensure_schema(CONN_STR, PREFIX)
    yield CONN_STR
    drop_schema(CONN_STR, PREFIX)
    _cleanup_collections(CONN_STR)


@pytest.fixture()
def collections(pg):
    """创建测试用 small/large collection，用后连同 embedding 一起清理。"""
    _cleanup_collections(pg)
    with psycopg.connect(pg) as conn:
        for name in (SMALL, LARGE):
            conn.execute(
                "INSERT INTO langchain_pg_collection (uuid, name, cmetadata) "
                "VALUES (%s, %s, '{}')",
                (str(uuid.uuid4()), name),
            )
        conn.commit()
    yield SMALL, LARGE
    _cleanup_collections(pg)


def _insert_chunks(conn_str, collection, document_id, version_id, count, is_active):
    with psycopg.connect(conn_str) as conn:
        for i in range(count):
            conn.execute(
                """
                INSERT INTO langchain_pg_embedding
                    (uuid, collection_id, document, cmetadata)
                VALUES (
                    %s,
                    (SELECT uuid FROM langchain_pg_collection WHERE name = %s),
                    %s, %s
                )
                """,
                (
                    str(uuid.uuid4()),
                    collection,
                    f"chunk {i} of {version_id}",
                    Jsonb(
                        {
                            "document_id": document_id,
                            "version_id": version_id,
                            "source": f"/data/{document_id}.pdf",
                            "is_active": is_active,
                            "chunk_id": f"{version_id}-c{i}",
                            "parent_id": f"{version_id}-p0",
                        }
                    ),
                ),
            )
        conn.commit()


def _active_flags(conn_str, collection, document_id, version_id) -> list[bool]:
    with psycopg.connect(conn_str) as conn:
        rows = conn.execute(
            """
            SELECT (e.cmetadata->>'is_active')::boolean
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
            WHERE c.name = %s
              AND e.cmetadata->>'document_id' = %s
              AND e.cmetadata->>'version_id' = %s
            ORDER BY e.cmetadata->>'chunk_id'
            """,
            (collection, document_id, version_id),
        ).fetchall()
    return [r[0] for r in rows]


def _expire_lease(conn_str, task_id: int) -> None:
    """用 SQL 把租约改到过去，模拟 Worker 崩溃后的租约过期。"""
    with psycopg.connect(conn_str) as conn:
        conn.execute(
            f"UPDATE {TASKS} SET lease_expires_at = now() - interval '1 second' "
            "WHERE task_id = %s",
            (task_id,),
        )
        conn.commit()


# ── Schema ──────────────────────────────────────────────────


class TestSchema:
    def test_ensure_schema_idempotent(self, pg):
        ensure_schema(pg, PREFIX)
        ensure_schema(pg, PREFIX)  # 第二次不报错
        with psycopg.connect(pg) as conn:
            rows = conn.execute(
                "SELECT tablename FROM pg_tables WHERE tablename LIKE %s",
                (f"{PREFIX}%",),
            ).fetchall()
        created = {r[0] for r in rows}
        assert created == {t(name, PREFIX) for name in BASE_TABLES}

    def test_one_active_version_partial_unique_index(self, pg):
        repo = VersionRepository(pg, PREFIX)
        repo.ensure_document("doc-idx", "kb-idx", "/data/idx.pdf")
        with psycopg.connect(pg) as conn:
            conn.execute(
                f"INSERT INTO {VERSIONS} (document_id, version_id, status) "
                "VALUES ('doc-idx', 'v1', 'active')"
            )
            conn.commit()
        # 同一 document_id 第二个 active 版本必须被拒绝
        with (
            psycopg.connect(pg) as conn,
            pytest.raises(psycopg.errors.UniqueViolation),
        ):
            conn.execute(
                f"INSERT INTO {VERSIONS} (document_id, version_id, status) "
                "VALUES ('doc-idx', 'v2', 'active')"
            )


# ── TaskRepository ──────────────────────────────────────────


class TestTaskRepository:
    def test_enqueue_dedup(self, pg):
        repo = TaskRepository(pg, PREFIX)
        tid = repo.enqueue("ingest_file", {"source": "/data/a.pdf"}, "dedup-enqueue")
        assert isinstance(tid, int)
        # pending 状态下同 key 拒绝重复入队
        assert repo.enqueue("ingest_file", {"source": "/data/a.pdf"}, "dedup-enqueue") is None
        # processing 状态下同样拒绝
        task = repo.claim("w1")
        assert task["task_id"] == tid
        assert repo.enqueue("ingest_file", {"source": "/data/a.pdf"}, "dedup-enqueue") is None
        # completed 后允许再入队（partial index 不再覆盖）
        assert repo.complete(tid, "w1") is True
        tid2 = repo.enqueue("ingest_file", {"source": "/data/a.pdf"}, "dedup-enqueue")
        assert isinstance(tid2, int) and tid2 != tid
        # 收尾：清掉 pending，避免影响后续并发 claim 测试
        repo.complete(repo.claim("w1")["task_id"], "w1")

    def test_claim_atomic_concurrent(self, pg):
        repo = TaskRepository(pg, PREFIX)
        tid = repo.enqueue("ingest_file", {"source": "/data/b.pdf"}, "dedup-claim")
        barrier = threading.Barrier(2)
        results: list[dict | None] = []

        def worker(wid: str):
            barrier.wait()
            results.append(repo.claim(wid))

        threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(2)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        got = [r for r in results if r is not None]
        assert len(got) == 1
        assert got[0]["task_id"] == tid
        assert got[0]["status"] == "processing"
        assert got[0]["locked_by"] in ("w0", "w1")
        assert got[0]["lease_expires_at"] is not None
        # 已领取任务不可被再 claim
        assert repo.claim("w2") is None

    def test_lease_expiry_reclaim_and_heartbeat(self, pg):
        repo = TaskRepository(pg, PREFIX)
        tid = repo.enqueue("ingest_file", {"source": "/data/c.pdf"}, "dedup-lease")

        task = repo.claim("w1", lease_seconds=300)
        assert task["task_id"] == tid
        # 心跳刷新成功，租约内其他 Worker 抢不到
        assert repo.heartbeat(tid, "w1", lease_seconds=300) is True
        assert repo.claim("w2") is None

        # 模拟租约过期 → 另一 Worker 可恢复，且 retries 不增加
        _expire_lease(pg, tid)
        stale = repo.list_stale()
        assert [s["task_id"] for s in stale] == [tid]
        reclaimed = repo.claim("w2")
        assert reclaimed["task_id"] == tid
        assert reclaimed["locked_by"] == "w2"
        assert reclaimed["retries"] == 0
        # 旧 Worker 心跳失效（锁已易主）
        assert repo.heartbeat(tid, "w1") is False
        assert repo.heartbeat(tid, "w2") is True

    def test_fail_retry_semantics(self, pg):
        repo = TaskRepository(pg, PREFIX)
        tid = repo.enqueue(
            "ingest_file", {"source": "/data/d.pdf"}, "dedup-fail", max_retries=2
        )
        repo.claim("w1")
        # 第一次失败：回 pending、retries+1、释放锁
        assert repo.fail(tid, "w1", "parse error", stage="parse") == "pending"
        row = repo.get(tid)
        assert row["status"] == "pending"
        assert row["retries"] == 1
        assert row["locked_by"] is None
        assert row["current_stage"] == "parse"
        assert row["error_message"] == "parse error"

        # 第二次失败：达到 max_retries → failed，保留失败现场
        repo.claim("w1")
        assert repo.fail(tid, "w1", "embed timeout", stage="embed") == "failed"
        row = repo.get(tid)
        assert row["status"] == "failed"
        assert row["retries"] == 2
        assert row["current_stage"] == "embed"
        assert row["error_message"] == "embed timeout"
        # failed 任务不可再被 claim
        assert repo.claim("w1") is None
        # 非持锁 worker 无法操作
        assert repo.fail(tid, "wX", "x") is None
        assert repo.complete(tid, "wX") is False

    def test_summary(self, pg):
        repo = TaskRepository(pg, PREFIX)
        summary = repo.summary()
        with psycopg.connect(pg) as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM {TASKS}").fetchone()[0]
        assert sum(summary.values()) == total
        assert summary.get("failed", 0) >= 1  # test_fail_retry_semantics 遗留


# ── FileStateRepository ─────────────────────────────────────


class TestFileStateRepository:
    def test_file_state_lifecycle(self, pg):
        repo = FileStateRepository(pg, PREFIX)
        src = "/data/fs/a.pdf"
        repo.upsert_indexed(src, "doc-fs", "hash-1", 1000.0, 100, "v1")
        row = repo.get(src)
        assert row["status"] == "active"
        assert row["content_hash"] == "hash-1"
        assert row["active_version_id"] == "v1"
        assert row["last_indexed_at"] is not None

        # 再索引（新 hash / 新版本）
        repo.upsert_indexed(src, "doc-fs", "hash-2", 2000.0, 200, "v2")
        row = repo.get(src)
        assert row["content_hash"] == "hash-2"
        assert row["active_version_id"] == "v2"

        # 扫描比对：只读 list_tracked
        tracked = repo.list_tracked("/data/fs/")
        assert [r["source"] for r in tracked] == [src]
        assert repo.list_tracked("/data/nonexistent/") == []

        # 删除事务成功后标记 deleted → 不再出现在 active 追踪列表
        assert repo.mark_deleted(src) is True
        assert repo.get(src)["status"] == "deleted"
        assert repo.list_tracked("/data/fs/") == []

        # 物理移除
        assert repo.remove(src) is True
        assert repo.get(src) is None


# ── VersionRepository ───────────────────────────────────────


class TestVersionRepository:
    def test_version_state_machine(self, pg):
        repo = VersionRepository(pg, PREFIX)
        repo.ensure_document("doc-sm", "kb-sm", "/data/sm.pdf", "sm.pdf")
        repo.ensure_document("doc-sm", "kb-sm", "/data/sm.pdf")  # 幂等

        repo.register_staging("doc-sm", "v1", "hash-1")
        repo.register_staging("doc-sm", "v1", "hash-1")  # 幂等，不降级
        assert repo.get("doc-sm", "v1")["status"] == "staging"

        assert repo.mark_validated("doc-sm", "v1", 2, 5) is True
        v = repo.get("doc-sm", "v1")
        assert v["status"] == "validated"
        assert (v["parent_count"], v["child_count"]) == (2, 5)
        assert v["validated_at"] is not None

        assert repo.mark_failed("doc-sm", "v1", "embed", "boom") is True
        v = repo.get("doc-sm", "v1")
        assert v["status"] == "failed"
        assert v["failure_stage"] == "embed"
        assert v["failure_reason"] == "boom"

        # failed 重试必须重新进 staging，不得直接激活
        assert repo.requeue_failed("doc-sm", "v1") is True
        v = repo.get("doc-sm", "v1")
        assert v["status"] == "staging"
        assert v["validated_at"] is None

        assert repo.mark_validated("doc-sm", "v1", 2, 5) is True
        assert len(repo.list_versions("doc-sm")) == 1
        assert repo.get_active("doc-sm") is None

    def test_activate_version_full_chain(self, pg, collections):
        small, large = collections
        repo = VersionRepository(pg, PREFIX)
        kb_repo = KbStateRepository(pg, PREFIX)
        fs_repo = FileStateRepository(pg, PREFIX)
        doc, kb, source = "doc-act", "kb-act", "/data/act.pdf"
        repo.ensure_document(doc, kb, source, "act.pdf")

        # v1：staging 写入（is_active=false）→ validated → 激活
        _insert_chunks(pg, small, doc, "v1", 2, False)
        _insert_chunks(pg, large, doc, "v1", 1, False)
        repo.register_staging(doc, "v1", "hash-v1")
        repo.mark_validated(doc, "v1", 1, 2)
        assert repo.activate_version(doc, "v1", source, "hash-v1", 1000.0, 100,
                                     small, large) is None
        assert all(_active_flags(pg, small, doc, "v1"))
        assert all(_active_flags(pg, large, doc, "v1"))

        # v2：修改产生新版本，激活后 v1 退役
        _insert_chunks(pg, small, doc, "v2", 3, False)
        _insert_chunks(pg, large, doc, "v2", 1, False)
        repo.register_staging(doc, "v2", "hash-v2")
        repo.mark_validated(doc, "v2", 1, 3)
        old = repo.activate_version(doc, "v2", source, "hash-v2", 2000.0, 200,
                                    small, large)
        assert old == "v1"

        # embedding is_active 翻转
        assert _active_flags(pg, small, doc, "v1") == [False, False]
        assert _active_flags(pg, large, doc, "v1") == [False]
        assert all(_active_flags(pg, small, doc, "v2"))
        assert all(_active_flags(pg, large, doc, "v2"))

        # 版本状态
        v1 = repo.get(doc, "v1")
        assert v1["status"] == "retired" and v1["retired_at"] is not None
        v2 = repo.get(doc, "v2")
        assert v2["status"] == "active" and v2["activated_at"] is not None
        assert repo.get_active(doc)["version_id"] == "v2"

        # 文档 / 文件状态 / revision
        assert repo.get_document(doc)["active_version_id"] == "v2"
        fs = fs_repo.get(source)
        assert fs["content_hash"] == "hash-v2"
        assert fs["active_version_id"] == "v2"
        assert fs["status"] == "active"
        assert kb_repo.get(kb)["active_revision"] == 2

    def test_activate_rollback_when_target_not_validated(self, pg, collections):
        small, large = collections
        repo = VersionRepository(pg, PREFIX)
        kb_repo = KbStateRepository(pg, PREFIX)
        doc, kb, source = "doc-rb", "kb-rb", "/data/rb.pdf"
        repo.ensure_document(doc, kb, source)

        # 基线：v1 已激活
        _insert_chunks(pg, small, doc, "v1", 1, False)
        _insert_chunks(pg, large, doc, "v1", 1, False)
        repo.register_staging(doc, "v1", "hash-v1")
        repo.mark_validated(doc, "v1", 1, 1)
        repo.activate_version(doc, "v1", source, "hash-v1", 1000.0, 100,
                              small, large)

        # v2 仍是 staging（未校验）→ 激活必须整体回滚
        _insert_chunks(pg, small, doc, "v2", 2, False)
        _insert_chunks(pg, large, doc, "v2", 1, False)
        repo.register_staging(doc, "v2", "hash-v2")
        with pytest.raises(ValueError, match="不可激活"):
            repo.activate_version(doc, "v2", source, "hash-v2", 2000.0, 200,
                                  small, large)

        # 旧版本继续服务：embedding 不变、版本状态不变、revision 不变
        assert all(_active_flags(pg, small, doc, "v1"))
        assert all(_active_flags(pg, large, doc, "v1"))
        assert _active_flags(pg, small, doc, "v2") == [False, False]
        assert _active_flags(pg, large, doc, "v2") == [False]
        assert repo.get(doc, "v1")["status"] == "active"
        assert repo.get(doc, "v2")["status"] == "staging"
        assert repo.get_document(doc)["active_version_id"] == "v1"
        assert kb_repo.get(kb)["active_revision"] == 1

    def test_delete_document_idempotent(self, pg, collections):
        small, large = collections
        repo = VersionRepository(pg, PREFIX)
        kb_repo = KbStateRepository(pg, PREFIX)
        fs_repo = FileStateRepository(pg, PREFIX)
        doc, kb, source = "doc-del", "kb-del", "/data/del.pdf"
        repo.ensure_document(doc, kb, source)
        _insert_chunks(pg, small, doc, "v1", 2, False)
        _insert_chunks(pg, large, doc, "v1", 1, False)
        repo.register_staging(doc, "v1", "hash-v1")
        repo.mark_validated(doc, "v1", 1, 2)
        repo.activate_version(doc, "v1", source, "hash-v1", 1000.0, 100,
                              small, large)
        rev_before = kb_repo.get(kb)["active_revision"]

        # 逻辑删除
        assert repo.delete_document(doc, small, large) is True
        assert _active_flags(pg, small, doc, "v1") == [False, False]
        assert _active_flags(pg, large, doc, "v1") == [False]
        d = repo.get_document(doc)
        assert d["status"] == "deleted"
        assert d["active_version_id"] is None
        assert repo.get(doc, "v1")["status"] == "retired"
        assert fs_repo.get(source)["status"] == "deleted"
        assert kb_repo.get(kb)["active_revision"] == rev_before + 1

        # 重复删除幂等：直接返回，revision 不再递增
        assert repo.delete_document(doc, small, large) is True
        assert kb_repo.get(kb)["active_revision"] == rev_before + 1

        # 不存在的文档
        assert repo.delete_document("doc-not-exist", small, large) is False


# ── KbStateRepository ───────────────────────────────────────


class TestKbStateRepository:
    def test_revision_lifecycle(self, pg):
        repo = KbStateRepository(pg, PREFIX)
        kb = "kb-state-test"
        state = repo.get(kb)  # 不存在则初始化
        assert state["active_revision"] == 0
        assert state["bm25_revision"] == 0

        # increment 单调递增并返回新值
        assert repo.increment_active_revision(kb) == 1
        assert repo.increment_active_revision(kb) == 2
        assert repo.increment_active_revision(kb) == 3

        # BM25 刷新判断
        assert repo.needs_bm25_refresh(kb, 3) is False
        assert repo.needs_bm25_refresh(kb, 2) is True

        repo.update_bm25_revision(kb, 3)
        assert repo.get(kb)["bm25_revision"] == 3
        assert repo.needs_bm25_refresh(kb, 3) is False
