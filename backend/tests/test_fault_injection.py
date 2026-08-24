"""故障注入测试（Spec V2 14.1 / 14.2 剩余场景）。

真实 PG（127.0.0.1:5433，连接串读自 config/vector_store_config.yaml）
+ fake pipeline / fake index，不打 MinerU / Embedding 真实 API。

与 test_pg_worker.py 的隔离差异：使用独立表前缀 test_fi_rag_ 与独立测试
collection test_fi_small / test_fi_large —— 避免与并行运行的全量套件
（test_rag_ 前缀 / agent_test2_* collection）互相清库。用后全部清理。
PG 不可达时整个模块 skip。

覆盖场景：
- 14.1 父块写入成功、子块写入失败 → staging 不可见、任务 failed、重试成功
- 14.1 子块写入成功、validate 前崩溃 → 同上语义
- 14.1 激活事务中 file_states 写入失败 → 整体回滚、旧版本继续服务、任务重试
- 14.1 激活成功、Worker 返回前崩溃 → 租约过期被另一 Worker 回收、幂等跳过
- 14.1 file_states 更新失败 → 激活回滚、file_states 行不存在
- 14.1 删除事务中途失败 → file_state 不标 deleted、版本仍 active、重试成功
- 14.1 重复删除幂等
- 14.2 两个 Worker 并发 claim 同一批任务 → 每任务只被一个 Worker 执行
- 14.2 同一文件重复入队 → dedup_key 第二次返回 None
- 14.2 超最大重试 → failed 且不再被 claim
"""

import threading
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
from psycopg.rows import dict_row
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
PREFIX = "test_fi_rag_"
KB = "test_fi"
SMALL = "test_fi_small"
LARGE = "test_fi_large"

FILES = t("rag_file_states", PREFIX)


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
                WHERE name LIKE 'test\_fi\_%'
            )
            """
        )
        conn.execute(r"DELETE FROM langchain_pg_collection WHERE name LIKE 'test\_fi\_%'")
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

    def __init__(self):
        self.calls = 0
        self.empty = False

    def process(self, file_path, version_context=None, **kwargs):
        self.calls += 1
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


class FakePgIndex:
    """fake 多粒度索引：真实读写 langchain_pg_embedding（不 embedding），
    记录调用并可注入故障。

    fail_children=True 时模拟"父块写入成功、子块写入失败"：父块先提交，
    子块写入前抛错。
    """

    enabled = True
    base_name = KB
    small_name = SMALL
    large_name = LARGE

    def __init__(self, conn_str: str):
        self._conn_str = conn_str
        self.insert_calls = 0
        self.fail_children = False

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
        with psycopg.connect(self._conn_str) as conn:
            for col in (SMALL, LARGE):
                conn.execute(
                    """
                    DELETE FROM langchain_pg_embedding
                    WHERE collection_id = (
                        SELECT uuid FROM langchain_pg_collection WHERE name = %s
                    )
                      AND cmetadata->>'document_id' = %s
                      AND cmetadata->>'version_id' = %s
                    """,
                    (col, document_id, version_id),
                )
            conn.commit()

    def insert(self, chunks, source="", versioned=False):
        assert versioned is True
        self.insert_calls += 1
        parents: dict[str, tuple[str, dict]] = {}
        for c in chunks:
            pid = c.metadata["parent_id"]
            if pid not in parents:
                parents[pid] = (
                    c.metadata.get("parent_doc", ""),
                    dict(c.metadata.get("parent_metadata") or {}),
                )
        # 父块先写入并提交 —— fail_children 时父块已落库、子块未写
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
            conn.commit()
        if self.fail_children:
            raise RuntimeError("child insert boom")
        with psycopg.connect(self._conn_str) as conn:
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

    def validate_version(self, document_id, version_id):
        stats = {
            "parent_count": self._count(LARGE, document_id, version_id),
            "child_count": self._count(SMALL, document_id, version_id),
        }
        ok = stats["parent_count"] > 0 and stats["child_count"] > 0
        return ok, stats


class SabotagedConn:
    """连接代理：执行到含 marker 的 SQL 时抛错（模拟事务中途断连）。

    用于在 activate_version / delete_document 事务内部的指定步骤
    （如 file_states 写入）注入失败，验证整体回滚语义。
    """

    def __init__(self, real, marker: str):
        self._real = real
        self._marker = marker

    def execute(self, sql, params=None):
        if self._marker in sql:
            raise psycopg.OperationalError(
                f"injected connection loss at: {self._marker}"
            )
        return self._real.execute(sql, params)

    def transaction(self):
        return self._real.transaction()

    def commit(self):
        return self._real.commit()

    def rollback(self):
        return self._real.rollback()

    def close(self):
        return self._real.close()

    def __enter__(self):
        self._real.__enter__()
        return self

    def __exit__(self, *args):
        return self._real.__exit__(*args)


def _sabotage_repo_connect(monkeypatch, env, marker: str):
    """把 version_repo 的 _connect 换成带故障注入的代理连接。"""

    def sabotaged_connect():
        return SabotagedConn(psycopg.connect(env.pg, row_factory=dict_row), marker)

    monkeypatch.setattr(env.versions, "_connect", sabotaged_connect)


# ── fixtures / helpers ──────────────────────────────────────


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

    index = FakePgIndex(pg)
    pipeline = FakePipeline()
    activated: list[int] = []

    worker = BackgroundWorker(
        task_repo=tasks,
        version_repo=versions,
        file_state_repo=files,
        kb_repo=kb,
        on_activated=lambda: activated.append(1),
    )
    worker._pipeline = pipeline
    worker._multi_index = index

    yield SimpleNamespace(
        pg=pg,
        tmp_path=tmp_path,
        file=doc_file,
        tasks=tasks,
        versions=versions,
        files=files,
        kb=kb,
        worker=worker,
        pipeline=pipeline,
        index=index,
        activated=activated,
    )

    _cleanup_collections(pg)


def _doc_ver(env) -> tuple[str, str]:
    data = env.file.read_bytes()
    content_hash = versioning.compute_content_hash(data)
    document_id = versioning.compute_document_id(str(env.file), kb_id=KB)
    return document_id, versioning.compute_version_id(content_hash)


def _enqueue_and_process(env, task_type, payload, dedup_key, max_retries=1) -> int:
    """入队 → claim → _process_pg（同步执行），返回 task_id。"""
    tid = env.tasks.enqueue(task_type, payload, dedup_key, max_retries=max_retries)
    assert tid is not None
    claimed = env.tasks.claim(env.worker._worker_id, lease_seconds=60)
    assert claimed is not None and claimed["task_id"] == tid
    env.worker._process_pg(claimed)
    return tid


def _ingest_via_queue(env, max_retries=1) -> int:
    document_id, version_id = _doc_ver(env)
    return _enqueue_and_process(
        env,
        "ingest_file",
        {"file_path": str(env.file)},
        f"ingest:{document_id}:{version_id}",
        max_retries=max_retries,
    )


def _retry_and_process(env, tid: int) -> None:
    """显式重置 failed 任务并重新处理（WRK-006：retry_failed 是唯一预算重置入口）。"""
    assert env.tasks.retry_failed(tid) is True
    claimed = env.tasks.claim(env.worker._worker_id, lease_seconds=60)
    assert claimed is not None and claimed["task_id"] == tid
    env.worker._process_pg(claimed)


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


def _single_active_version(env, document_id) -> dict:
    """断言该文档有且仅有一个 active version，返回它。"""
    versions = env.versions.list_versions(document_id)
    active = [v for v in versions if v["status"] == "active"]
    assert len(active) == 1
    assert env.versions.get_active(document_id)["version_id"] == active[0]["version_id"]
    return active[0]


# ── 14.1 父块写入成功、子块写入失败 ──────────────────────────


class TestChildInsertFailure:
    def test_parent_written_child_fails_retry_succeeds(self, env):
        env.index.fail_children = True
        document_id, version_id = _doc_ver(env)
        source = str(env.file)

        tid = _ingest_via_queue(env)
        row = env.tasks.get(tid)
        assert row["status"] == "failed"
        assert row["current_stage"] == "store"
        assert "child insert boom" in row["error_message"]

        # staging 不可见：父块已落库但 is_active 全 false，子块未写入
        assert _active_flags(env.pg, LARGE, document_id, version_id) == [False, False]
        assert _active_flags(env.pg, SMALL, document_id, version_id) == []
        assert env.versions.get(document_id, version_id)["status"] == "staging"
        assert env.versions.get_active(document_id) is None
        assert env.files.get(source) is None

        # 修复后显式重试（WRK-006）：staging 残留被清理后重建，最终只有一个 active version
        env.index.fail_children = False
        _retry_and_process(env, tid)
        assert env.tasks.get(tid)["status"] == "completed"

        v = _single_active_version(env, document_id)
        assert v["version_id"] == version_id
        assert v["parent_count"] == 2 and v["child_count"] == 3
        assert _active_flags(env.pg, LARGE, document_id, version_id) == [True, True]
        assert _active_flags(env.pg, SMALL, document_id, version_id) == [True] * 3
        assert env.files.get(source)["active_version_id"] == version_id


# ── 14.1 子块写入成功、validate 前崩溃 ───────────────────────


class TestCrashBeforeValidate:
    def test_validate_crash_retry_succeeds(self, env, monkeypatch):
        document_id, version_id = _doc_ver(env)

        def boom(*args, **kwargs):
            raise RuntimeError("validate crash boom")

        monkeypatch.setattr(env.index, "validate_version", boom)
        tid = _ingest_via_queue(env)
        row = env.tasks.get(tid)
        assert row["status"] == "failed"
        assert row["current_stage"] == "validate"

        # 数据已落库但未激活：is_active 全 false，版本停留 staging
        assert _active_flags(env.pg, SMALL, document_id, version_id) == [False] * 3
        assert _active_flags(env.pg, LARGE, document_id, version_id) == [False, False]
        assert env.versions.get(document_id, version_id)["status"] == "staging"
        assert env.versions.get_active(document_id) is None
        assert env.files.get(str(env.file)) is None

        # 显式重试后成功，最终只有一个 active version
        monkeypatch.undo()
        _retry_and_process(env, tid)
        assert env.tasks.get(tid)["status"] == "completed"
        v = _single_active_version(env, document_id)
        assert v["version_id"] == version_id
        assert _active_flags(env.pg, SMALL, document_id, version_id) == [True] * 3


# ── 14.1 激活事务中任一步失败（file_states 写入点断连）────────


class TestActivateTransactionFailure:
    def test_activate_file_states_write_fails_rolls_back(self, env, monkeypatch):
        # v1 先激活
        _ingest_via_queue(env)
        document_id, v1 = _doc_ver(env)
        rev_before = env.kb.get(KB)["active_revision"]

        # 修改文件 → v2
        env.file.write_bytes(b"hello v2 modified")
        _, v2 = _doc_ver(env)

        # 注入故障：激活事务内 file_states 写入时断连
        _sabotage_repo_connect(monkeypatch, env, f"INSERT INTO {FILES}")
        tid = _ingest_via_queue(env)
        row = env.tasks.get(tid)
        assert row["status"] == "failed"
        assert row["current_stage"] == "activate"

        # 整体回滚：旧版本 is_active 不变、registry 状态不变、revision 不变
        assert env.versions.get_active(document_id)["version_id"] == v1
        assert env.versions.get(document_id, v1)["status"] == "active"
        assert env.versions.get(document_id, v2)["status"] == "validated"
        assert env.versions.get_document(document_id)["active_version_id"] == v1
        assert _active_flags(env.pg, SMALL, document_id, v1) == [True] * 3
        assert _active_flags(env.pg, LARGE, document_id, v1) == [True, True]
        assert _active_flags(env.pg, SMALL, document_id, v2) == [False] * 3
        assert _active_flags(env.pg, LARGE, document_id, v2) == [False, False]
        assert env.kb.get(KB)["active_revision"] == rev_before
        assert env.files.get(str(env.file))["active_version_id"] == v1

        # 修复后显式重试成功：v2 激活、v1 退役，只有一个 active version
        monkeypatch.undo()
        _retry_and_process(env, tid)
        assert env.tasks.get(tid)["status"] == "completed"
        v = _single_active_version(env, document_id)
        assert v["version_id"] == v2
        assert env.versions.get(document_id, v1)["status"] == "retired"
        assert env.kb.get(KB)["active_revision"] == rev_before + 1


# ── 14.1 激活提交即任务完成（ACT-004 / T-ACT-05）──────────────


class TestCrashAfterActivateBeforeReturn:
    def test_activate_commit_completes_task_no_double_execution(self, env):
        """T-ACT-05：任务 completed 与激活在同一事务提交——崩溃后不存在
        "已激活但任务未完成"的中间态；陈旧重试幂等退出、不二次增 revision。"""
        document_id, version_id = _doc_ver(env)
        tid = env.tasks.enqueue(
            "ingest_file",
            {"file_path": str(env.file)},
            f"ingest:{document_id}:{version_id}",
            max_retries=1,
        )
        claimed = env.tasks.claim(env.worker._worker_id, lease_seconds=60)

        # 直接跑处理器（绕过 _process_pg 的外层 complete）：
        # 激活事务内已完成任务（ACT-004）
        env.worker._lease_generation = claimed["lease_generation"]
        env.worker._handle_ingest_pg(claimed)
        assert env.versions.get_active(document_id)["version_id"] == version_id
        assert env.tasks.get(tid)["status"] == "completed"
        rev_after = env.kb.get(KB)["active_revision"]

        # 模拟崩溃后的陈旧重试（重复跑同一处理器）：幂等跳过，
        # pipeline/store 不重跑，revision 不二次增加
        env.worker._handle_ingest_pg(claimed)
        env.worker._lease_generation = None
        assert env.pipeline.calls == 1  # pipeline 未重跑
        assert env.index.insert_calls == 1  # store 未重跑
        assert env.kb.get(KB)["active_revision"] == rev_after

        # 无重复数据，仍只有一个 active version
        assert env.index._count(SMALL, document_id, version_id) == 3
        assert env.index._count(LARGE, document_id, version_id) == 2
        _single_active_version(env, document_id)


# ── 14.1 file_states 更新失败 → 激活回滚（新文档场景）─────────


class TestFileStateWriteFailure:
    def test_file_state_failure_rolls_back_activation(self, env, monkeypatch):
        document_id, version_id = _doc_ver(env)
        source = str(env.file)

        _sabotage_repo_connect(monkeypatch, env, f"INSERT INTO {FILES}")
        tid = _ingest_via_queue(env)
        row = env.tasks.get(tid)
        assert row["status"] == "failed"
        assert row["current_stage"] == "activate"

        # file_states 行不存在（file_states 只能由激活事务写入）
        assert env.files.get(source) is None
        # 版本未激活：停留 validated，is_active 全 false，无 active version
        assert env.versions.get(document_id, version_id)["status"] == "validated"
        assert env.versions.get_active(document_id) is None
        assert _active_flags(env.pg, SMALL, document_id, version_id) == [False] * 3
        assert _active_flags(env.pg, LARGE, document_id, version_id) == [False, False]

        # 修复后显式重试成功，file_state 落库
        monkeypatch.undo()
        _retry_and_process(env, tid)
        assert env.tasks.get(tid)["status"] == "completed"
        fs = env.files.get(source)
        assert fs is not None and fs["status"] == "active"
        assert fs["active_version_id"] == version_id
        _single_active_version(env, document_id)


# ── 14.2 两个 Worker 并发 claim 同一批任务 ───────────────────


class TestConcurrentClaim:
    def test_two_workers_claim_disjoint(self, env):
        task_ids = []
        for i in range(5):
            tid = env.tasks.enqueue(
                "ingest_file",
                {"file_path": f"/tmp/f{i}.md"},
                f"ingest:doc{i}:hash{i}",
                max_retries=1,
            )
            assert tid is not None
            task_ids.append(tid)

        barrier = threading.Barrier(2)
        claimed_by: dict[str, list[tuple[int, int]]] = {"worker-A": [], "worker-B": []}

        def claim_loop(worker_id: str):
            barrier.wait()
            while True:
                task = env.tasks.claim(worker_id, lease_seconds=60)
                if task is None:
                    break
                claimed_by[worker_id].append((task["task_id"], task["lease_generation"]))

        threads = [
            threading.Thread(target=claim_loop, args=(wid,))
            for wid in ("worker-A", "worker-B")
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=30)

        # 每个任务只被一个 Worker 领取
        all_claimed = [tid for ids in claimed_by.values() for tid, _ in ids]
        assert sorted(all_claimed) == sorted(task_ids)
        assert {t for t, _ in claimed_by["worker-A"]}.isdisjoint(
            t for t, _ in claimed_by["worker-B"]
        )

        # 收尾：清掉 processing 任务
        for wid, ids in claimed_by.items():
            for tid, gen in ids:
                env.tasks.complete(tid, wid, lease_generation=gen)


# ── 14.2 同一文件重复入队 ────────────────────────────────────


class TestDedupEnqueue:
    def test_duplicate_enqueue_returns_none(self, env):
        document_id, version_id = _doc_ver(env)
        dedup = f"ingest:{document_id}:{version_id}"

        tid = env.tasks.enqueue(
            "ingest_file", {"file_path": str(env.file)}, dedup, max_retries=1
        )
        assert tid is not None
        # 相同 dedup_key 的 pending 任务已存在 → 第二次返回 None
        assert (
            env.tasks.enqueue("ingest_file", {"file_path": str(env.file)}, dedup)
            is None
        )
        # 收尾：清掉 pending 任务
        claimed = env.tasks.claim(env.worker._worker_id, lease_seconds=60)
        env.tasks.complete(
            claimed["task_id"],
            env.worker._worker_id,
            lease_generation=claimed["lease_generation"],
        )


# ── 14.2 超最大重试 ──────────────────────────────────────────


class TestMaxRetries:
    def test_exceeds_max_retries_failed_not_claimable(self, env):
        """T-WRK-07：达 max_retries 进 failed 后不再自动领取，
        同内容新任务不得重置预算，只能显式 retry_failed 恢复。"""
        env.pipeline.empty = True  # pipeline 产出空 chunks，稳定失败
        document_id, version_id = _doc_ver(env)

        tid = env.tasks.enqueue(
            "ingest_file",
            {"file_path": str(env.file)},
            f"ingest:{document_id}:{version_id}",
            max_retries=2,
        )
        # 第 1 次失败：retries=1 < 2 → 回 pending
        claimed = env.tasks.claim(env.worker._worker_id, lease_seconds=60)
        env.worker._process_pg(claimed)
        row = env.tasks.get(tid)
        assert row["status"] == "pending" and row["retries"] == 1

        # 第 2 次失败：retries=2 >= max_retries → failed
        claimed = env.tasks.claim(env.worker._worker_id, lease_seconds=60)
        env.worker._process_pg(claimed)
        row = env.tasks.get(tid)
        assert row["status"] == "failed" and row["retries"] == 2
        assert row["current_stage"] == "pipeline"

        # failed 任务不再被 claim
        assert env.tasks.claim(env.worker._worker_id, lease_seconds=60) is None

        # WRK-006：同内容（同 dedup_key）新入队被拒绝，不得无限重置失败预算
        assert env.tasks.enqueue(
            "ingest_file",
            {"file_path": str(env.file)},
            f"ingest:{document_id}:{version_id}",
            max_retries=2,
        ) is None
        assert env.tasks.claim(env.worker._worker_id, lease_seconds=60) is None

        # 显式重试入口：重置后才可再领取；失败现场保留
        assert env.tasks.retry_failed(tid) is True
        row = env.tasks.get(tid)
        assert row["status"] == "pending" and row["retries"] == 0
        assert row["current_stage"] == "pipeline"
        claimed = env.tasks.claim(env.worker._worker_id, lease_seconds=60)
        assert claimed is not None and claimed["task_id"] == tid
        # 收尾：清掉 processing 任务
        env.tasks.complete(
            tid, env.worker._worker_id,
            lease_generation=claimed["lease_generation"],
        )


# ── 14.1 删除事务中途失败 ────────────────────────────────────


class TestDeleteTransactionFailure:
    def test_delete_mid_tx_failure_rolls_back(self, env, monkeypatch):
        _ingest_via_queue(env)
        document_id, version_id = _doc_ver(env)
        source = str(env.file)
        rev_before = env.kb.get(KB)["active_revision"]

        # 注入故障：删除事务内 file_states 更新时断连
        _sabotage_repo_connect(monkeypatch, env, f"UPDATE {FILES}")
        tid = _enqueue_and_process(
            env, "delete_document", {"source": source}, f"delete:{document_id}"
        )
        row = env.tasks.get(tid)
        assert row["status"] == "failed"
        assert row["current_stage"] == "delete"

        # 整体回滚：file_state 未被标记 deleted、版本仍 active、文档仍 active
        assert env.files.get(source)["status"] == "active"
        assert env.versions.get(document_id, version_id)["status"] == "active"
        assert env.versions.get_document(document_id)["status"] == "active"
        assert _active_flags(env.pg, SMALL, document_id, version_id) == [True] * 3
        assert env.kb.get(KB)["active_revision"] == rev_before

        # 修复后显式重试删除成功
        monkeypatch.undo()
        _retry_and_process(env, tid)
        assert env.tasks.get(tid)["status"] == "completed"
        assert env.files.get(source)["status"] == "deleted"
        assert env.versions.get(document_id, version_id)["status"] == "retired"


# ── 14.1 重复删除幂等 ────────────────────────────────────────


class TestRepeatDelete:
    def test_delete_idempotent(self, env):
        _ingest_via_queue(env)
        document_id, _ = _doc_ver(env)
        source = str(env.file)
        rev_before = env.kb.get(KB)["active_revision"]

        tid1 = _enqueue_and_process(
            env, "delete_document", {"source": source}, f"delete:{document_id}"
        )
        assert env.tasks.get(tid1)["status"] == "completed"
        assert env.kb.get(KB)["active_revision"] == rev_before + 1

        # 重复删除：仍 completed，revision 不再递增，状态不再变化
        tid2 = _enqueue_and_process(
            env, "delete_document", {"source": source}, f"delete:{document_id}"
        )
        assert env.tasks.get(tid2)["status"] == "completed"
        assert env.kb.get(KB)["active_revision"] == rev_before + 1
        assert env.files.get(source)["status"] == "deleted"
        assert env.versions.get_document(document_id)["status"] == "deleted"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
