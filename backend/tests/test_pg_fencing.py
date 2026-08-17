"""Worker fencing / 取消 / 文档锁并发测试（Spec V3 阶段 B：WRK-001~005）。

真实 PG（127.0.0.1:5433，连接串读自 config/vector_store_config.yaml）
+ fake pipeline / fake index，不打 MinerU / Embedding 真实 API。

隔离约定：表前缀 test_fw_rag_、collection test_fw_small / test_fw_large
（与 test_pg_worker / test_fault_injection 的前缀错开，可安全清理）。
PG 不可达时整个模块 skip。

覆盖：
- T-WRK-01 新建 Worker 实例无新写请求自动恢复 pending 和租约过期任务
- T-WRK-03 旧 Worker 心跳发现易主后不再 store/validate/activate（取消生效）
- T-WRK-05 两 Worker 同一文档 staging 变更窗口严格串行
- T-WRK-06 不同文档可并行处理（无全局串行退化）
"""

import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

# 规避已知 C 扩展加载顺序冲突：sqlalchemy 必须先于 psycopg 完成导入
# （与 test_pg_worker.py 相同的环境坑，勿调整顺序）。
import sqlalchemy  # noqa: F401
import psycopg
import pytest
import yaml
from langchain_core.documents import Document
from psycopg.types.json import Jsonb

from agent_material.rag_engine import versioning
from agent_material.rag_engine.db.pg import (
    FileStateRepository,
    KbStateRepository,
    TaskRepository,
    VersionRepository,
    drop_schema,
    ensure_schema,
    t,
)
from agent_material.rag_engine.db.pg.schema import normalize_conn_str
from agent_material.rag_engine.db.worker import BackgroundWorker

ROOT = Path(__file__).resolve().parents[2]
from agent_material.utils.config_handler import get_vector_store_config

CONN_STR = normalize_conn_str(get_vector_store_config()["connection_string"])
PREFIX = "test_fw_rag_"
KB = "test_fw"
SMALL = "test_fw_small"
LARGE = "test_fw_large"

TASKS = t("rag_ingest_tasks", PREFIX)


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
                WHERE name LIKE 'test\_fw\_%'
            )
            """
        )
        conn.execute(r"DELETE FROM langchain_pg_collection WHERE name LIKE 'test\_fw\_%'")
        conn.commit()


@pytest.fixture(scope="module")
def pg():
    if not _pg_available():
        pytest.skip("PostgreSQL 127.0.0.1:5433 不可达")
    drop_schema(CONN_STR, PREFIX)
    _cleanup_collections(CONN_STR)
    ensure_schema(CONN_STR, PREFIX)
    yield CONN_STR
    drop_schema(CONN_STR, PREFIX)
    _cleanup_collections(CONN_STR)


# ── fake pipeline / fake index ──────────────────────────────


class FakePipeline:
    """从文件内容确定性地生成 2 父 3 子块，并应用真实版本上下文逻辑。"""

    def __init__(self, sleep_seconds: float = 0.0):
        self.calls = 0
        self.sleep_seconds = sleep_seconds
        self.empty = False

    def process(self, file_path, version_context=None, **kwargs):
        self.calls += 1
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        if self.empty:
            return []
        from agent_material.rag_engine.document_pipeline import ProcessingPipeline

        data = Path(file_path).read_bytes()
        raw = []
        for i in range(3):
            pid = "p0" if i < 2 else "p1"
            raw.append(
                Document(
                    page_content=f"{Path(file_path).name} 子块{i} {data[:16]!r}",
                    metadata={
                        "parent_id": pid,
                        "parent_doc": f"父块 {pid} 内容 {data[:16]!r}",
                        "parent_metadata": {"heading_path": [f"章节 {pid}"]},
                    },
                )
            )
        return ProcessingPipeline._apply_version_context(raw, version_context)


def _fake_split_text(text: str, source: str, version_context: dict):
    """ingest_text 路径的确定性切分 fake（2 父 3 子，不打真实 splitter）。"""
    from agent_material.rag_engine.document_pipeline import ProcessingPipeline

    raw = []
    for i in range(3):
        pid = "p0" if i < 2 else "p1"
        raw.append(
            Document(
                page_content=f"{text[:24]} 子块{i}",
                metadata={
                    "parent_id": pid,
                    "parent_doc": f"父块 {pid} 内容",
                    "parent_metadata": {"heading_path": [f"章节 {pid}"]},
                },
            )
        )
    return ProcessingPipeline._apply_version_context(raw, version_context)


class FakePgIndex:
    """fake 多粒度索引：真实读写 langchain_pg_embedding（不 embedding）。

    insert_sleep / insert_intervals 用于观测 staging 写入窗口的并发行文。
    """

    enabled = True
    base_name = KB
    small_name = SMALL
    large_name = LARGE

    def __init__(self, conn_str: str):
        self._conn_str = conn_str
        self.insert_calls = 0
        self.delete_calls: list[tuple] = []
        self.insert_sleep = 0.0
        self.insert_intervals: list[tuple[float, float]] = []

    def _delete_rows(self, conn, collection, document_id, version_id):
        conn.execute(
            """
            DELETE FROM langchain_pg_embedding
            WHERE collection_id = (
                SELECT uuid FROM langchain_pg_collection WHERE name = %s
            )
              AND cmetadata->>'document_id' = %s
              AND cmetadata->>'version_id' = %s
            """,
            (collection, document_id, version_id),
        )

    def _insert_row(self, conn, collection, text, meta):
        conn.execute(
            """
            INSERT INTO langchain_pg_embedding (uuid, collection_id, document, cmetadata)
            VALUES (
                %s,
                (SELECT uuid FROM langchain_pg_collection WHERE name = %s),
                %s, %s
            )
            """,
            (str(uuid.uuid4()), collection, text, Jsonb(meta)),
        )

    def delete_by_version(self, document_id, version_id):
        self.delete_calls.append((document_id, version_id))
        with psycopg.connect(self._conn_str) as conn:
            for col in (SMALL, LARGE):
                self._delete_rows(conn, col, document_id, version_id)
            conn.commit()

    def insert(self, chunks, source="", versioned=False):
        assert versioned is True
        self.insert_calls += 1
        start = time.monotonic()
        try:
            if self.insert_sleep:
                time.sleep(self.insert_sleep)
            parents: dict[str, tuple[str, dict]] = {}
            for c in chunks:
                pid = c.metadata["parent_id"]
                if pid not in parents:
                    parents[pid] = (
                        c.metadata.get("parent_doc", ""),
                        dict(c.metadata.get("parent_metadata") or {}),
                    )
            with psycopg.connect(self._conn_str) as conn:
                for pid, (ptext, pmeta) in parents.items():
                    meta = {
                        **pmeta,
                        "chunk_type": "parent",
                        "source": source,
                        "parent_id": pid,
                        "is_active": False,
                    }
                    self._insert_row(conn, LARGE, ptext, meta)
                for c in chunks:
                    meta = {
                        k: v
                        for k, v in c.metadata.items()
                        if k not in ("parent_doc", "parent_metadata")
                    }
                    meta.update(chunk_type="child", source=source, is_active=False)
                    self._insert_row(conn, SMALL, c.page_content, meta)
                conn.commit()
            return list(parents)
        finally:
            self.insert_intervals.append((start, time.monotonic()))

    def validate_version(self, document_id, version_id):
        stats = {
            "parent_count": self._count(LARGE, document_id, version_id),
            "child_count": self._count(SMALL, document_id, version_id),
        }
        ok = stats["parent_count"] > 0 and stats["child_count"] > 0
        return ok, stats

    def _count(self, collection, document_id, version_id) -> int:
        with psycopg.connect(self._conn_str) as conn:
            return conn.execute(
                """
                SELECT COUNT(*)
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = %s
                  AND e.cmetadata->>'document_id' = %s
                  AND e.cmetadata->>'version_id' = %s
                """,
                (collection, document_id, version_id),
            ).fetchone()[0]


# ── fixtures / helpers ──────────────────────────────────────


def _make_worker(env, worker_id: str, **kwargs) -> BackgroundWorker:
    """创建一个挂了 fake pipeline/index 的 Worker 实例（不启动线程）。"""
    w = BackgroundWorker(
        task_repo=env.tasks,
        version_repo=env.versions,
        file_state_repo=env.files,
        kb_repo=env.kb,
        **kwargs,
    )
    w._pipeline = env.pipeline
    w._multi_index = env.index
    w._worker_id = worker_id
    w._split_text_pg = _fake_split_text
    return w


@pytest.fixture()
def env(pg, tmp_path):
    _cleanup_collections(pg)
    with psycopg.connect(pg) as conn:
        for name in (SMALL, LARGE):
            conn.execute(
                "INSERT INTO langchain_pg_collection (uuid, name, cmetadata) "
                "VALUES (%s, %s, '{}')",
                (str(uuid.uuid4()), name),
            )
        # 控制面表跨用例共享：每个用例清空，保证隔离
        for name in (
            "rag_ingest_tasks",
            "rag_file_states",
            "rag_document_versions",
            "rag_documents",
            "rag_kb_state",
        ):
            conn.execute(f"DELETE FROM {t(name, PREFIX)}")
        conn.commit()

    tasks = TaskRepository(pg, PREFIX)
    versions = VersionRepository(pg, PREFIX)
    files = FileStateRepository(pg, PREFIX)
    kb = KbStateRepository(pg, PREFIX)

    doc_file = tmp_path / "a.md"
    doc_file.write_bytes(b"hello v1")

    yield SimpleNamespace(
        pg=pg,
        tmp_path=tmp_path,
        file=doc_file,
        tasks=tasks,
        versions=versions,
        files=files,
        kb=kb,
        pipeline=FakePipeline(),
        index=FakePgIndex(pg),
    )

    _cleanup_collections(pg)


def _doc_ver(env) -> tuple[str, str]:
    data = env.file.read_bytes()
    content_hash = versioning.compute_content_hash(data)
    document_id = versioning.compute_document_id(str(env.file), kb_id=KB)
    return document_id, versioning.compute_version_id(content_hash)


def _expire_lease(conn_str: str, task_id: int) -> None:
    """用 SQL 把租约改到过去，模拟 Worker 崩溃后的租约过期。"""
    with psycopg.connect(conn_str) as conn:
        conn.execute(
            f"UPDATE {TASKS} SET lease_expires_at = now() - interval '1 second' "
            "WHERE task_id = %s",
            (task_id,),
        )
        conn.commit()


def _wait_status(env, task_id: int, status: str, timeout: float = 15.0) -> dict:
    """轮询任务直到目标状态（超时失败）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        row = env.tasks.get(task_id)
        if row["status"] == status:
            return row
        time.sleep(0.1)
    raise AssertionError(f"任务 {task_id} 未在 {timeout}s 内进入 {status}: {row['status']}")


# ── T-WRK-01：启动即恢复 pending / 租约过期任务 ──────────────


class TestWorkerStartupRecovery:
    def test_start_recovers_pending_and_stale_tasks(self, env):
        """新建 Worker 实例（模拟服务重启）后无新写请求也会恢复执行。"""
        document_id, version_id = _doc_ver(env)
        # 1) 一个租约过期的 processing 任务（旧 Worker 崩溃遗留，另一文档）
        file_b = env.tmp_path / "b.md"
        file_b.write_bytes(b"hello b")
        hash_b = versioning.compute_content_hash(b"hello b")
        doc_b = versioning.compute_document_id(str(file_b), kb_id=KB)
        tid_stale = env.tasks.enqueue(
            "ingest_file",
            {"file_path": str(file_b)},
            f"ingest:{doc_b}:{versioning.compute_version_id(hash_b)}",
            max_retries=1,
        )
        ghost = env.tasks.claim("ghost-worker", lease_seconds=60)
        assert ghost["task_id"] == tid_stale
        gen_ghost = ghost["lease_generation"]
        _expire_lease(env.pg, tid_stale)
        # 2) 一个 pending 任务（从未被领取）
        tid_pending = env.tasks.enqueue(
            "ingest_file",
            {"file_path": str(env.file)},
            f"ingest:{document_id}:{version_id}",
            max_retries=1,
        )

        # 模拟重启：全新 Worker 实例，start() 后应立即领取两类任务
        worker = _make_worker(env, "fw-restarted", poll_interval=0.1)
        worker.start()
        try:
            row_pending = _wait_status(env, tid_pending, "completed")
            row_stale = _wait_status(env, tid_stale, "completed")
        finally:
            worker.stop()
            worker.join(timeout=5)

        assert row_pending["lease_generation"] == 1  # 首次领取
        # T-WRK-02：过期回收 generation 递增
        assert row_stale["lease_generation"] > gen_ghost
        # 两个文档都已激活
        assert env.versions.get_active(document_id) is not None
        assert env.versions.get_active(doc_b) is not None


# ── T-WRK-03：租约易主后旧 Worker 取消执行 ────────────────────


class TestLeaseLostCancellation:
    def test_old_worker_aborts_before_store(self, env):
        """旧 Worker 心跳发现易主 → 取消生效：不再 store/validate/activate，
        不调 fail/complete 覆盖新 Worker 状态。"""
        document_id, version_id = _doc_ver(env)
        tid = env.tasks.enqueue(
            "ingest_file",
            {"file_path": str(env.file)},
            f"ingest:{document_id}:{version_id}",
            max_retries=3,
        )
        # 旧 Worker 领取（短租约，心跳间隔 1/3s）
        old_worker = _make_worker(env, "old-worker", lease_seconds=1)
        old_worker._pipeline.sleep_seconds = 0.8  # pipeline 期间租约被抢
        claimed = env.tasks.claim("old-worker", lease_seconds=1)
        gen_old = claimed["lease_generation"]

        th = threading.Thread(
            target=old_worker._process_pg, args=(claimed,), daemon=True
        )
        th.start()
        time.sleep(0.3)  # 让旧 Worker 进入慢 pipeline 阶段

        # 模拟租约过期 + 新 Worker 接管（generation 递增）。
        # 旧 Worker 心跳可能恰好刷新租约，故 expire+claim 做小步重试直至接管成功
        reclaimed = None
        for _ in range(50):
            _expire_lease(env.pg, tid)
            reclaimed = env.tasks.claim("new-worker", lease_seconds=60)
            if reclaimed is not None:
                break
            time.sleep(0.05)
        assert reclaimed is not None and reclaimed["task_id"] == tid
        assert reclaimed["lease_generation"] == gen_old + 1

        th.join(timeout=15)
        assert not th.is_alive()

        # 取消在 staging 清理前生效：没有任何 staging 写入/校验/激活
        assert env.index.delete_calls == []
        assert env.index.insert_calls == 0
        v = env.versions.get(document_id, version_id)
        assert v["status"] == "staging"  # 只登记，未 validate/activate
        assert env.versions.get_active(document_id) is None

        # 旧 Worker 未篡改任务状态：仍归新 Worker 所有，retries 不增加
        row = env.tasks.get(tid)
        assert row["status"] == "processing"
        assert row["locked_by"] == "new-worker"
        assert row["lease_generation"] == gen_old + 1
        assert row["retries"] == 0
        assert row["error_message"] is None

        # 收尾：新 Worker 走正常流程完成任务（幂等续跑成功）
        new_worker = _make_worker(env, "new-worker")
        new_worker._process_pg(env.tasks.get(tid))
        assert env.tasks.get(tid)["status"] == "completed"
        assert env.versions.get_active(document_id)["version_id"] == version_id


class TestExpiredLeaseFencing:
    def test_expired_owner_cannot_heartbeat_update_or_finish(self, env):
        tid = env.tasks.enqueue(
            "ingest_file", {"file_path": str(env.file)}, "expired-owner-test"
        )
        claimed = env.tasks.claim("expired-worker", lease_seconds=60)
        generation = claimed["lease_generation"]
        _expire_lease(env.pg, tid)

        assert env.tasks.heartbeat(
            tid, "expired-worker", lease_generation=generation
        ) is False
        assert env.tasks.update_stage(
            tid, "expired-worker", "store", lease_generation=generation
        ) is False
        assert env.tasks.complete(
            tid, "expired-worker", lease_generation=generation
        ) is False
        assert env.tasks.fail(
            tid, "expired-worker", "too late", lease_generation=generation
        ) is None


# ── T-WRK-05/06：文档级 advisory lock 串行化 ─────────────────


class TestDocumentLockSerialization:
    def _enqueue_text(self, env, source: str, text: str) -> int:
        content_hash = versioning.compute_content_hash(text.encode("utf-8"))
        document_id = versioning.compute_document_id(source, kb_id=KB)
        version_id = versioning.compute_version_id(content_hash)
        tid = env.tasks.enqueue(
            "ingest_text",
            {"text": text, "source": source},
            f"ingest_text:{document_id}:{version_id}",
            max_retries=1,
        )
        assert tid is not None
        return tid

    def test_t_wrk_05_same_document_staging_window_serialized(self, env):
        """T-WRK-05：两 Worker 处理同一文档的不同版本，staging 变更窗口
        （文档锁覆盖 store→activate）严格串行。"""
        env.index.insert_sleep = 0.4
        source = "manual://fw-shared-doc"
        document_id = versioning.compute_document_id(source, kb_id=KB)
        t1 = self._enqueue_text(env, source, "共享文档内容 v1")
        t2 = self._enqueue_text(env, source, "共享文档内容 v2")

        w1 = _make_worker(env, "fw-worker-1")
        w2 = _make_worker(env, "fw-worker-2")
        c1 = env.tasks.claim("fw-worker-1", lease_seconds=60)
        c2 = env.tasks.claim("fw-worker-2", lease_seconds=60)
        assert {c1["task_id"], c2["task_id"]} == {t1, t2}

        barrier = threading.Barrier(2)

        def run(w, c):
            barrier.wait()
            w._process_pg(c)

        threads = [
            threading.Thread(target=run, args=(w1, c1)),
            threading.Thread(target=run, args=(w2, c2)),
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=30)
            assert not th.is_alive()

        assert env.tasks.get(t1)["status"] == "completed"
        assert env.tasks.get(t2)["status"] == "completed"

        # staging 写入窗口零交错（严格串行）
        intervals = sorted(env.index.insert_intervals)
        assert len(intervals) == 2
        assert intervals[1][0] >= intervals[0][1], (
            f"同一文档 staging 窗口发生交错: {intervals}"
        )

        # 两个版本顺序激活：最终仅一个 active version，revision 恰好 +2
        versions = env.versions.list_versions(document_id)
        actives = [v for v in versions if v["status"] == "active"]
        assert len(actives) == 1
        retired = [v for v in versions if v["status"] == "retired"]
        assert len(retired) == 1
        assert env.kb.get(KB)["active_revision"] == 2

    def test_t_wrk_06_different_documents_parallel(self, env):
        """T-WRK-06：不同 document_id 的 staging 窗口可并行（无全局串行退化）。"""
        env.index.insert_sleep = 0.4
        t1 = self._enqueue_text(env, "manual://fw-doc-a", "文档 A 内容")
        t2 = self._enqueue_text(env, "manual://fw-doc-b", "文档 B 内容")

        w1 = _make_worker(env, "fw-worker-a")
        w2 = _make_worker(env, "fw-worker-b")
        c1 = env.tasks.claim("fw-worker-a", lease_seconds=60)
        c2 = env.tasks.claim("fw-worker-b", lease_seconds=60)

        barrier = threading.Barrier(2)

        def run(w, c):
            barrier.wait()
            w._process_pg(c)

        threads = [
            threading.Thread(target=run, args=(w1, c1)),
            threading.Thread(target=run, args=(w2, c2)),
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=30)
            assert not th.is_alive()

        assert env.tasks.get(t1)["status"] == "completed"
        assert env.tasks.get(t2)["status"] == "completed"

        # 写入窗口存在交错 → 并行生效
        intervals = sorted(env.index.insert_intervals)
        assert len(intervals) == 2
        assert intervals[1][0] < intervals[0][1], (
            f"不同文档被意外串行: {intervals}"
        )
        assert env.kb.get(KB)["active_revision"] == 2
