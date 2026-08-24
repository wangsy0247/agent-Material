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
    FencingRejectedError,
    FileStateRepository,
    KbStateRepository,
    RowcountMismatchError,
    TaskRepository,
    VersionRepository,
    drop_schema,
    ensure_schema,
    t,
)
from agent_material.rag_engine.db.pg.schema import BASE_TABLES, normalize_conn_str

ROOT = Path(__file__).resolve().parents[2]
from agent_material.utils.config_handler import get_vector_store_config

CONN_STR = normalize_conn_str(get_vector_store_config()["connection_string"])
PREFIX = "test_rag_"
SMALL = "test_ctrl_small"
LARGE = "test_ctrl_large"
VISUAL = "test_ctrl_visual"

TASKS = t("rag_ingest_tasks", PREFIX)
VERSIONS = t("rag_document_versions", PREFIX)


def _claim_task(pg, key: str, worker_id: str = "w-test") -> dict:
    """入队并领取一个任务，返回含 lease_generation 的 task dict（fencing token）。

    激活/删除事务的调用前提是持有一个有效任务（ACT-001 / DEL-002）。
    """
    repo = TaskRepository(pg, PREFIX)
    tid = repo.enqueue(
        "ingest_file", {"source": key}, f"{key}-{uuid.uuid4().hex}", max_retries=1
    )
    assert tid is not None
    task = repo.claim(worker_id, lease_seconds=300)
    assert task is not None and task["task_id"] == tid
    return task


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

    def test_lease_generation_column_and_alter_compat(self, pg):
        """WRK-002：lease_generation 列存在；存量表（缺列）经 ensure_schema 幂等补列。"""
        tasks_table = t("rag_ingest_tasks", PREFIX)
        with psycopg.connect(pg) as conn:
            cols = {
                r[0]
                for r in conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = %s",
                    (tasks_table,),
                ).fetchall()
            }
        assert "lease_generation" in cols

        # 模拟存量库：删列后 ensure_schema 必须能通过 ALTER 幂等补回
        with psycopg.connect(pg) as conn:
            conn.execute(f"ALTER TABLE {tasks_table} DROP COLUMN lease_generation")
            conn.commit()
        ensure_schema(pg, PREFIX)
        with psycopg.connect(pg) as conn:
            cols = {
                r[0]
                for r in conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = %s",
                    (tasks_table,),
                ).fetchall()
            }
        assert "lease_generation" in cols

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
        assert repo.complete(tid, "w1", lease_generation=task["lease_generation"]) is True
        tid2 = repo.enqueue("ingest_file", {"source": "/data/a.pdf"}, "dedup-enqueue")
        assert isinstance(tid2, int) and tid2 != tid
        # 收尾：清掉 pending，避免影响后续并发 claim 测试
        t = repo.claim("w1")
        repo.complete(t["task_id"], "w1", lease_generation=t["lease_generation"])

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
        """T-WRK-02 / T-WRK-04：租约过期回收 generation 递增并接管；
        旧 Worker 心跳/完成/失败均因 fencing 不匹配失效。"""
        repo = TaskRepository(pg, PREFIX)
        tid = repo.enqueue("ingest_file", {"source": "/data/c.pdf"}, "dedup-lease")

        task = repo.claim("w1", lease_seconds=300)
        assert task["task_id"] == tid
        gen1 = task["lease_generation"]
        # 心跳刷新成功，租约内其他 Worker 抢不到
        assert repo.heartbeat(tid, "w1", lease_seconds=300, lease_generation=gen1) is True
        assert repo.claim("w2") is None

        # 模拟租约过期 → 另一 Worker 可恢复：generation 递增、retries 不增加
        _expire_lease(pg, tid)
        stale = repo.list_stale()
        assert [s["task_id"] for s in stale] == [tid]
        reclaimed = repo.claim("w2")
        assert reclaimed["task_id"] == tid
        assert reclaimed["locked_by"] == "w2"
        assert reclaimed["retries"] == 0
        gen2 = reclaimed["lease_generation"]
        assert gen2 > gen1  # WRK-002：回收即递增 fencing token
        # 旧 Worker 心跳/完成/失败全部失效（WRK-003/004，更新 0 行不误报）
        assert repo.heartbeat(tid, "w1", lease_generation=gen1) is False
        assert repo.complete(tid, "w1", lease_generation=gen1) is False
        assert repo.fail(tid, "w1", "x", lease_generation=gen1) is None
        assert repo.get(tid)["status"] == "processing"  # 未被旧 Worker 篡改
        assert repo.get(tid)["retries"] == 0
        # 旧 Worker 即使盗用新 generation 也因 locked_by 不匹配被拒
        assert repo.heartbeat(tid, "w1", lease_generation=gen2) is False
        # 新 Worker 心跳正常
        assert repo.heartbeat(tid, "w2", lease_generation=gen2) is True
        # 收尾
        assert repo.complete(tid, "w2", lease_generation=gen2) is True

    def test_fail_retry_semantics(self, pg):
        repo = TaskRepository(pg, PREFIX)
        tid = repo.enqueue(
            "ingest_file", {"source": "/data/d.pdf"}, "dedup-fail", max_retries=2
        )
        t = repo.claim("w1")
        # 第一次失败：回 pending、retries+1、释放锁
        assert repo.fail(tid, "w1", "parse error", stage="parse",
                         lease_generation=t["lease_generation"]) == "pending"
        row = repo.get(tid)
        assert row["status"] == "pending"
        assert row["retries"] == 1
        assert row["locked_by"] is None
        assert row["current_stage"] == "parse"
        assert row["error_message"] == "parse error"

        # 第二次失败：达到 max_retries → failed，保留失败现场
        t = repo.claim("w1")
        assert repo.fail(tid, "w1", "embed timeout", stage="embed",
                         lease_generation=t["lease_generation"]) == "failed"
        row = repo.get(tid)
        assert row["status"] == "failed"
        assert row["retries"] == 2
        assert row["current_stage"] == "embed"
        assert row["error_message"] == "embed timeout"
        # failed 任务不可再被 claim
        assert repo.claim("w1") is None
        # 非持锁 worker / 错误 generation 无法操作
        assert repo.fail(tid, "wX", "x", lease_generation=t["lease_generation"]) is None
        assert repo.complete(tid, "wX", lease_generation=t["lease_generation"]) is False

        # WRK-006：failed 后同 dedup_key 拒绝新入队（不得无限重置重试预算），
        # 只能显式 retry_failed 重置
        assert (
            repo.enqueue("ingest_file", {"source": "/data/d.pdf"}, "dedup-fail") is None
        )
        assert repo.retry_failed(tid) is True
        row = repo.get(tid)
        assert row["status"] == "pending" and row["retries"] == 0
        # 重置后可重新领取；收尾：再次打满重试回 failed，避免影响 test_summary
        t = repo.claim("w1")
        assert t is not None and t["task_id"] == tid
        assert repo.fail(tid, "w1", "again", stage="embed",
                         lease_generation=t["lease_generation"]) == "pending"
        t = repo.claim("w1")
        assert repo.fail(tid, "w1", "again", stage="embed",
                         lease_generation=t["lease_generation"]) == "failed"

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
        t1 = _claim_task(pg, "act-v1")
        assert repo.activate_version(doc, "v1", source, "hash-v1", 1000.0, 100,
                                     small, large,
                                     task_id=t1["task_id"], worker_id="w-test",
                                     lease_generation=t1["lease_generation"]) is None
        # ACT-004：任务已在激活事务内 completed
        assert TaskRepository(pg, PREFIX).get(t1["task_id"])["status"] == "completed"
        assert all(_active_flags(pg, small, doc, "v1"))
        assert all(_active_flags(pg, large, doc, "v1"))

        # v2：修改产生新版本，激活后 v1 退役
        _insert_chunks(pg, small, doc, "v2", 3, False)
        _insert_chunks(pg, large, doc, "v2", 1, False)
        repo.register_staging(doc, "v2", "hash-v2")
        repo.mark_validated(doc, "v2", 1, 3)
        t2 = _claim_task(pg, "act-v2")
        old = repo.activate_version(doc, "v2", source, "hash-v2", 2000.0, 200,
                                    small, large,
                                    task_id=t2["task_id"], worker_id="w-test",
                                    lease_generation=t2["lease_generation"])
        assert old == "v1"
        assert TaskRepository(pg, PREFIX).get(t2["task_id"])["status"] == "completed"

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

    def test_artifact_rows_follow_version_activation(self, pg, collections):
        small, large = collections
        repo = VersionRepository(pg, PREFIX)
        doc, kb, source = "doc-artifact", "kb-artifact", "/data/artifact.pdf"
        repo.ensure_document(doc, kb, source, "artifact.pdf")

        def artifact(version: str, ordinal: int = 0) -> dict:
            return {
                "artifact_id": f"artifact-{version}-{ordinal}",
                "artifact_type": "figure",
                "ordinal": ordinal,
                "page_start": 1,
                "page_end": 1,
                "bbox": [10, 20, 300, 220],
                "caption": f"图 {version}",
                "nearby_context": "实验结果",
                "searchable_text": f"[图表] 图 {version} 实验结果",
                "structured_content": "",
                "asset_uri": f"rag-asset://{doc}/{version}/figure.png",
                "content_hash": f"hash-{version}",
                "parser": "mineru",
                "visual_description": {
                    "schema_version": "visual-chart-v1",
                    "chart_type": "line_chart",
                },
                "visual_summary": "三条曲线随温度升高而下降",
                "visual_model": "fake-vl",
                "visual_status": "completed",
                "visual_confidence": 0.9,
                "visual_prompt_version": "visual-chart-v1",
                "visual_content_hash": f"visual-{version}",
            }

        _insert_chunks(pg, small, doc, "v1", 1, False)
        _insert_chunks(pg, large, doc, "v1", 1, False)
        repo.register_staging(doc, "v1", "hash-v1")
        assert repo.replace_staging_artifacts(doc, "v1", [artifact("v1")]) == 1
        assert repo.validate_artifacts(doc, "v1", 1)[0] is True
        repo.mark_validated(doc, "v1", 1, 1, artifact_count=1)
        t1 = _claim_task(pg, "artifact-v1")
        repo.activate_version(
            doc, "v1", source, "hash-v1", 1000.0, 100, small, large,
            task_id=t1["task_id"], worker_id="w-test",
            lease_generation=t1["lease_generation"],
        )
        active_artifact = repo.list_artifacts(doc, "v1")[0]
        assert active_artifact["status"] == "active"
        assert active_artifact["visual_status"] == "completed"
        assert active_artifact["visual_description"]["chart_type"] == "line_chart"

        _insert_chunks(pg, small, doc, "v2", 1, False)
        _insert_chunks(pg, large, doc, "v2", 1, False)
        repo.register_staging(doc, "v2", "hash-v2")
        repo.replace_staging_artifacts(doc, "v2", [artifact("v2")])
        repo.mark_validated(doc, "v2", 1, 1, artifact_count=1)
        t2 = _claim_task(pg, "artifact-v2")
        assert repo.activate_version(
            doc, "v2", source, "hash-v2", 2000.0, 200, small, large,
            task_id=t2["task_id"], worker_id="w-test",
            lease_generation=t2["lease_generation"],
        ) == "v1"

        assert repo.list_artifacts(doc, "v1", active_only=False)[0]["status"] == "retired"
        assert repo.list_artifacts(doc, "v2")[0]["status"] == "active"

    def test_visual_vectors_activate_and_retire_in_same_version_transaction(
        self, pg, collections
    ):
        small, large = collections
        with psycopg.connect(pg) as conn:
            conn.execute(
                "INSERT INTO langchain_pg_collection (uuid, name, cmetadata) "
                "VALUES (%s, %s, '{}')",
                (str(uuid.uuid4()), VISUAL),
            )
            conn.commit()

        repo = VersionRepository(pg, PREFIX)
        doc, kb, source = "doc-visual", "kb-visual", "/data/visual.pdf"
        repo.ensure_document(doc, kb, source, "visual.pdf")
        _insert_chunks(pg, small, doc, "v1", 1, False)
        _insert_chunks(pg, large, doc, "v1", 1, False)
        _insert_chunks(pg, VISUAL, doc, "v1", 1, False)
        repo.register_staging(doc, "v1", "hash-v1")
        repo.mark_validated(doc, "v1", 1, 1, visual_count=1)

        activate_task = _claim_task(pg, "visual-v1")
        repo.activate_version(
            doc,
            "v1",
            source,
            "hash-v1",
            1000.0,
            100,
            small,
            large,
            VISUAL,
            task_id=activate_task["task_id"],
            worker_id="w-test",
            lease_generation=activate_task["lease_generation"],
        )
        assert _active_flags(pg, VISUAL, doc, "v1") == [True]

        delete_task = _claim_task(pg, "visual-delete")
        assert repo.delete_document(
            doc,
            small,
            large,
            VISUAL,
            task_id=delete_task["task_id"],
            worker_id="w-test",
            lease_generation=delete_task["lease_generation"],
        ) == "deleted"
        assert _active_flags(pg, VISUAL, doc, "v1") == [False]

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
        t1 = _claim_task(pg, "rb-v1")
        repo.activate_version(doc, "v1", source, "hash-v1", 1000.0, 100,
                              small, large,
                              task_id=t1["task_id"], worker_id="w-test",
                              lease_generation=t1["lease_generation"])

        # v2 仍是 staging（未校验）→ 激活必须整体回滚
        _insert_chunks(pg, small, doc, "v2", 2, False)
        _insert_chunks(pg, large, doc, "v2", 1, False)
        repo.register_staging(doc, "v2", "hash-v2")
        t2 = _claim_task(pg, "rb-v2")
        with pytest.raises(ValueError, match="不可激活"):
            repo.activate_version(doc, "v2", source, "hash-v2", 2000.0, 200,
                                  small, large,
                                  task_id=t2["task_id"], worker_id="w-test",
                                  lease_generation=t2["lease_generation"])
        # 任务不得标记 completed（ACT-003）
        assert TaskRepository(pg, PREFIX).get(t2["task_id"])["status"] == "processing"

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
        t0 = _claim_task(pg, "del-v1")
        repo.activate_version(doc, "v1", source, "hash-v1", 1000.0, 100,
                              small, large,
                              task_id=t0["task_id"], worker_id="w-test",
                              lease_generation=t0["lease_generation"])
        rev_before = kb_repo.get(kb)["active_revision"]

        # 逻辑删除（T-DEL 主路径：任务在删除事务内 completed）
        t1 = _claim_task(pg, "del-1")
        assert repo.delete_document(doc, small, large,
                                    task_id=t1["task_id"], worker_id="w-test",
                                    lease_generation=t1["lease_generation"]) == "deleted"
        assert TaskRepository(pg, PREFIX).get(t1["task_id"])["status"] == "completed"
        assert _active_flags(pg, small, doc, "v1") == [False, False]
        assert _active_flags(pg, large, doc, "v1") == [False]
        d = repo.get_document(doc)
        assert d["status"] == "deleted"
        assert d["active_version_id"] is None
        assert repo.get(doc, "v1")["status"] == "retired"
        assert fs_repo.get(source)["status"] == "deleted"
        assert kb_repo.get(kb)["active_revision"] == rev_before + 1

        # T-DEL-03：重复删除幂等 completed，revision 不再递增
        t2 = _claim_task(pg, "del-2")
        assert repo.delete_document(doc, small, large,
                                    task_id=t2["task_id"], worker_id="w-test",
                                    lease_generation=t2["lease_generation"]) == "already_deleted"
        row2 = TaskRepository(pg, PREFIX).get(t2["task_id"])
        assert row2["status"] == "completed"
        assert row2["current_stage"] == "delete_already_deleted"
        assert kb_repo.get(kb)["active_revision"] == rev_before + 1

        # T-DEL-04：删除不存在的文档 → no-op completed，终态语义明确
        t3 = _claim_task(pg, "del-3")
        assert repo.delete_document("doc-not-exist", small, large,
                                    task_id=t3["task_id"], worker_id="w-test",
                                    lease_generation=t3["lease_generation"]) == "not_found"
        row3 = TaskRepository(pg, PREFIX).get(t3["task_id"])
        assert row3["status"] == "completed"
        assert row3["current_stage"] == "delete_noop"
        assert kb_repo.get(kb)["active_revision"] == rev_before + 1


# ── 激活/删除事务加固（ACT-001~004 / INV-005，DEL-002/003）───


class TestActivationGuards:
    """行数校验与 fencing 校验失败必须整体回滚（旧版本继续服务）。"""

    def _setup_active(self, pg, collections, tag: str,
                      v1_parents: int = 1, v1_children: int = 1):
        """构造 v1 已激活的文档基线，返回 (repo, doc, kb, source)。"""
        small, large = collections
        repo = VersionRepository(pg, PREFIX)
        doc, kb, source = f"doc-{tag}", f"kb-{tag}", f"/data/{tag}.pdf"
        repo.ensure_document(doc, kb, source)
        _insert_chunks(pg, small, doc, "v1", v1_children, False)
        _insert_chunks(pg, large, doc, "v1", v1_parents, False)
        repo.register_staging(doc, "v1", "hash-v1")
        repo.mark_validated(doc, "v1", v1_parents, v1_children)
        t = _claim_task(pg, f"{tag}-v1")
        repo.activate_version(doc, "v1", source, "hash-v1", 1000.0, 100,
                              small, large,
                              task_id=t["task_id"], worker_id="w-test",
                              lease_generation=t["lease_generation"])
        return repo, doc, kb, source

    def test_t_act_01_parent_rowcount_mismatch_rolls_back(self, pg, collections):
        """T-ACT-01：父块 rowcount 少于 expected → 整体回滚，旧版本仍 active。"""
        small, large = collections
        repo, doc, kb, source = self._setup_active(pg, collections, "act01")
        rev_before = KbStateRepository(pg, PREFIX).get(kb)["active_revision"]

        # v2：validated 记录 parent_count=2/child_count=2，但 large 只写入 1 行父块
        _insert_chunks(pg, small, doc, "v2", 2, False)
        _insert_chunks(pg, large, doc, "v2", 1, False)
        repo.register_staging(doc, "v2", "hash-v2")
        repo.mark_validated(doc, "v2", 2, 2)
        t = _claim_task(pg, "act01-v2")
        with pytest.raises(RowcountMismatchError, match="rowcount_mismatch"):
            repo.activate_version(doc, "v2", source, "hash-v2", 2000.0, 200,
                                  small, large,
                                  task_id=t["task_id"], worker_id="w-test",
                                  lease_generation=t["lease_generation"])

        # 整体回滚：v1 仍 active 且 is_active 未变；v2 不上线、停留 validated
        assert repo.get(doc, "v1")["status"] == "active"
        assert repo.get(doc, "v2")["status"] == "validated"
        assert all(_active_flags(pg, small, doc, "v1"))
        assert all(_active_flags(pg, large, doc, "v1"))
        assert _active_flags(pg, small, doc, "v2") == [False, False]
        assert _active_flags(pg, large, doc, "v2") == [False]
        assert repo.get_document(doc)["active_version_id"] == "v1"
        assert KbStateRepository(pg, PREFIX).get(kb)["active_revision"] == rev_before
        # 任务不得标记 completed（ACT-003）
        assert TaskRepository(pg, PREFIX).get(t["task_id"])["status"] == "processing"

    def test_t_act_02_child_rowcount_zero_rolls_back(self, pg, collections):
        """T-ACT-02：子块 rowcount 为 0 → 整体回滚，revision 不变。"""
        small, large = collections
        repo, doc, kb, source = self._setup_active(pg, collections, "act02")
        rev_before = KbStateRepository(pg, PREFIX).get(kb)["active_revision"]

        # v2：validated 记录 parent=1/child=2，small 实际 0 行
        _insert_chunks(pg, large, doc, "v2", 1, False)
        repo.register_staging(doc, "v2", "hash-v2")
        repo.mark_validated(doc, "v2", 1, 2)
        t = _claim_task(pg, "act02-v2")
        with pytest.raises(RowcountMismatchError):
            repo.activate_version(doc, "v2", source, "hash-v2", 2000.0, 200,
                                  small, large,
                                  task_id=t["task_id"], worker_id="w-test",
                                  lease_generation=t["lease_generation"])

        assert repo.get(doc, "v1")["status"] == "active"
        assert repo.get(doc, "v2")["status"] == "validated"
        assert all(_active_flags(pg, small, doc, "v1"))
        assert all(_active_flags(pg, large, doc, "v1"))
        assert _active_flags(pg, large, doc, "v2") == [False]
        assert KbStateRepository(pg, PREFIX).get(kb)["active_revision"] == rev_before
        assert TaskRepository(pg, PREFIX).get(t["task_id"])["status"] == "processing"

    def test_t_act_04_fencing_mismatch_rejected_and_rolls_back(self, pg, collections):
        """T-ACT-04：fencing 不匹配（generation 错误）→ 激活事务失败回滚，
        任务不被误标 completed。"""
        small, large = collections
        repo, doc, kb, source = self._setup_active(pg, collections, "act04")
        rev_before = KbStateRepository(pg, PREFIX).get(kb)["active_revision"]

        _insert_chunks(pg, small, doc, "v2", 1, False)
        _insert_chunks(pg, large, doc, "v2", 1, False)
        repo.register_staging(doc, "v2", "hash-v2")
        repo.mark_validated(doc, "v2", 1, 1)
        t = _claim_task(pg, "act04-v2")
        with pytest.raises(FencingRejectedError):
            repo.activate_version(doc, "v2", source, "hash-v2", 2000.0, 200,
                                  small, large,
                                  task_id=t["task_id"], worker_id="w-test",
                                  lease_generation=t["lease_generation"] + 999)

        # 未发生任何状态变更
        assert repo.get(doc, "v1")["status"] == "active"
        assert repo.get(doc, "v2")["status"] == "validated"
        assert all(_active_flags(pg, small, doc, "v1"))
        assert _active_flags(pg, small, doc, "v2") == [False]
        assert KbStateRepository(pg, PREFIX).get(kb)["active_revision"] == rev_before
        assert TaskRepository(pg, PREFIX).get(t["task_id"])["status"] == "processing"

    def test_t_del_01_parent_offline_mismatch_rolls_back(self, pg, collections):
        """T-DEL-01：父块实际下线行数与记录不一致 → 删除事务回滚，文件仍 active。"""
        small, large = collections
        repo, doc, kb, source = self._setup_active(
            pg, collections, "del01", v1_parents=2, v1_children=2
        )
        rev_before = KbStateRepository(pg, PREFIX).get(kb)["active_revision"]

        # 模拟数据缺失：外部删掉一行 active 父块（下线时将少更新 1 行）
        with psycopg.connect(pg) as conn:
            conn.execute(
                """
                DELETE FROM langchain_pg_embedding
                WHERE uuid = (
                    SELECT e.uuid FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    WHERE c.name = %s
                      AND e.cmetadata->>'document_id' = %s
                      AND e.cmetadata->>'version_id' = 'v1'
                    LIMIT 1
                )
                """,
                (large, doc),
            )
            conn.commit()

        t = _claim_task(pg, "del01-del")
        with pytest.raises(RowcountMismatchError):
            repo.delete_document(doc, small, large,
                                 task_id=t["task_id"], worker_id="w-test",
                                 lease_generation=t["lease_generation"])

        # 整体回滚：文件仍 active、版本仍 active、revision 不变、任务未 completed
        assert repo.get(doc, "v1")["status"] == "active"
        assert repo.get_document(doc)["status"] == "active"
        assert all(_active_flags(pg, small, doc, "v1"))
        assert _active_flags(pg, large, doc, "v1") == [True]  # 仅剩的 1 行也未被下线
        assert FileStateRepository(pg, PREFIX).get(source)["status"] == "active"
        assert KbStateRepository(pg, PREFIX).get(kb)["active_revision"] == rev_before
        assert TaskRepository(pg, PREFIX).get(t["task_id"])["status"] == "processing"


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
