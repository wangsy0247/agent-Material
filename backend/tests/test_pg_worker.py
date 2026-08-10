"""PG 控制面 Worker 集成测试（Spec V2 阶段 2：离线索引重构）。

真实 PG（127.0.0.1:5433，连接串读自 config/vector_store_config.yaml）
+ fake pipeline / fake index，不打 MinerU / Embedding 真实 API。

- 控制面表：table_prefix="test_rag_"，测试前后 DROP；
- 测试 collection：agent_test2_small / agent_test2_large（用后清理）；
- PG 不可达时整个模块 skip。
"""

import os
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

# 规避已知 C 扩展加载顺序冲突：sqlalchemy 必须先于 psycopg 完成导入，
# 否则 psycopg → gateway app → langchain_text_splitters(tiktoken) 的导入顺序
# 会在导入期 Segmentation Fault（WSL/Anaconda 环境，LD_PRELOAD 无法覆盖）。
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
from agent_material.rag_engine.db.worker import BackgroundWorker, pg_scan_and_enqueue

ROOT = Path(__file__).resolve().parents[2]
CONN_STR = normalize_conn_str(
    yaml.safe_load((ROOT / "config" / "vector_store_config.yaml").read_text())[
        "connection_string"
    ]
)
PREFIX = "test_rag_"
KB = "agent_test2"
SMALL = "agent_test2_small"
LARGE = "agent_test2_large"

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
                WHERE name LIKE 'agent\_test2\_%'
            )
            """
        )
        conn.execute(
            r"DELETE FROM langchain_pg_collection WHERE name LIKE 'agent\_test2\_%'"
        )
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
        self.last_version_context = None

    def process(self, file_path, version_context=None, **kwargs):
        self.calls += 1
        self.last_version_context = version_context
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


class FakePgIndex:
    """fake 多粒度索引：真实读写 langchain_pg_embedding（不 embedding），
    记录调用并可注入故障。"""

    enabled = True
    base_name = KB
    small_name = SMALL
    large_name = LARGE

    def __init__(self, conn_str: str):
        self._conn_str = conn_str
        self.insert_calls = 0
        self.delete_calls: list[tuple] = []
        self.fail_delete = False      # staging 清理失败注入
        self.drop_last_child = False  # 少写一条子块（validate 数量不符注入）

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
        if self.fail_delete:
            raise RuntimeError("staging cleanup boom")
        with psycopg.connect(self._conn_str) as conn:
            for col in (SMALL, LARGE):
                self._delete_rows(conn, col, document_id, version_id)
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
        children = chunks[:-1] if self.drop_last_child else chunks
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
            for c in children:
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
        # 控制面表跨用例共享：每个用例清空，保证隔离（任务/文件状态/版本/revision）
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


# ── ingest 快乐路径 / 幂等 ──────────────────────────────────


class TestPgIngestHappyPath:
    def test_enqueue_claim_complete(self, env):
        document_id, version_id = _doc_ver(env)
        source = str(env.file)

        # 激活前 file_states 不得有记录（Spec 5.4：只能由激活事务写入）
        assert env.files.get(source) is None

        tid = _ingest_via_queue(env)
        row = env.tasks.get(tid)
        assert row["status"] == "completed"
        assert row["completed_at"] is not None

        # 版本已激活，文档指向它
        assert env.versions.get_active(document_id)["version_id"] == version_id
        assert env.versions.get_document(document_id)["active_version_id"] == version_id

        # file_states 在激活后由激活事务更新
        fs = env.files.get(source)
        assert fs is not None
        assert fs["status"] == "active"
        assert fs["active_version_id"] == version_id
        assert fs["content_hash"] == versioning.compute_content_hash(
            env.file.read_bytes()
        )

        # embedding is_active 翻牌
        assert all(_active_flags(env.pg, SMALL, document_id, version_id))
        assert all(_active_flags(env.pg, LARGE, document_id, version_id))

        # store 阶段先清 staging 残留再写入；bm25 回调触发
        assert env.index.delete_calls == [(document_id, version_id)]
        assert env.activated == [1]
        # pipeline 收到版本上下文
        assert env.pipeline.last_version_context == {
            "document_id": document_id,
            "version_id": version_id,
        }

    def test_idempotent_skip_and_enqueue_dedup(self, env):
        document_id, version_id = _doc_ver(env)
        dedup = f"ingest:{document_id}:{version_id}"

        _ingest_via_queue(env)
        # 重复 enqueue 同 dedup_key：completed 后允许再入队，但处理时幂等跳过
        tid2 = env.tasks.enqueue(
            "ingest_file", {"file_path": str(env.file)}, dedup, max_retries=1
        )
        claimed = env.tasks.claim(env.worker._worker_id, lease_seconds=60)
        assert claimed["task_id"] == tid2
        env.worker._process_pg(claimed)
        assert env.tasks.get(tid2)["status"] == "completed"

        # 第二次 ingest 幂等跳过：pipeline / insert 不再执行
        assert env.pipeline.calls == 1
        assert env.index.insert_calls == 1

        # 同 dedup_key 的 pending 任务重复入队 → None
        tid3 = env.tasks.enqueue(
            "ingest_file", {"file_path": str(env.file)}, f"{dedup}-x", max_retries=1
        )
        assert tid3 is not None
        assert (
            env.tasks.enqueue(
                "ingest_file", {"file_path": str(env.file)}, f"{dedup}-x"
            )
            is None
        )
        # 收尾：清掉 pending 任务，避免影响其他用例的 claim 断言
        claimed = env.tasks.claim(env.worker._worker_id, lease_seconds=60)
        env.tasks.complete(claimed["task_id"], env.worker._worker_id)


# ── 失败语义 ────────────────────────────────────────────────


class TestPgIngestFailures:
    def test_empty_chunks_fails_at_pipeline(self, env):
        env.pipeline.empty = True
        document_id, version_id = _doc_ver(env)

        tid = _ingest_via_queue(env)
        row = env.tasks.get(tid)
        # 空 chunks 必须明确失败，不得 silently completed（修正 V1 行为）
        assert row["status"] == "failed"
        assert row["current_stage"] == "pipeline"
        assert "空 chunks" in row["error_message"]

        v = env.versions.get(document_id, version_id)
        assert v["status"] == "failed"
        assert v["failure_stage"] == "pipeline"
        assert env.index.insert_calls == 0
        assert env.files.get(str(env.file)) is None

    def test_staging_cleanup_failure_aborts_task(self, env):
        env.index.fail_delete = True
        document_id, version_id = _doc_ver(env)

        tid = _ingest_via_queue(env)
        row = env.tasks.get(tid)
        assert row["status"] == "failed"
        assert row["current_stage"] == "store"
        assert "staging cleanup boom" in row["error_message"]

        # staging 未激活、未写任何 embedding、file_state 未更新
        assert env.versions.get(document_id, version_id)["status"] == "staging"
        assert env.versions.get_active(document_id) is None
        assert env.index.insert_calls == 0
        assert env.files.get(str(env.file)) is None

    def test_validate_count_mismatch_fails(self, env):
        env.index.drop_last_child = True  # 少写一条子块 → 实际 2 != 预期 3
        document_id, version_id = _doc_ver(env)

        tid = _ingest_via_queue(env)
        row = env.tasks.get(tid)
        assert row["status"] == "failed"
        assert row["current_stage"] == "validate"
        assert "子块数量不符" in row["error_message"]

        v = env.versions.get(document_id, version_id)
        assert v["status"] == "failed"
        assert v["failure_stage"] == "validate"
        # 版本不激活：embedding 全部 is_active=false
        assert _active_flags(env.pg, SMALL, document_id, version_id) == [False, False]
        assert _active_flags(env.pg, LARGE, document_id, version_id) == [False, False]

    def test_failed_version_retry_recovers(self, env):
        """validate 失败 → 修复后重试：failed 版本回 staging 并完整恢复。"""
        env.index.drop_last_child = True
        _ingest_via_queue(env)
        document_id, version_id = _doc_ver(env)
        assert env.versions.get(document_id, version_id)["status"] == "failed"

        env.index.drop_last_child = False
        _ingest_via_queue(env)
        v = env.versions.get(document_id, version_id)
        assert v["status"] == "active"
        assert all(_active_flags(env.pg, SMALL, document_id, version_id))

    def test_activate_failure_rolls_back_old_keeps_serving(self, env, monkeypatch):
        """激活时目标版本非 validated → 整体回滚，旧版本继续 active。"""
        # v1 先激活
        _ingest_via_queue(env)
        document_id, v1 = _doc_ver(env)

        # 修改文件 → v2
        env.file.write_bytes(b"hello v2 modified")
        _, v2 = _doc_ver(env)

        # 注入故障：activate 前把 v2 状态翻回 staging（模拟校验后状态被篡改）
        real_activate = env.versions.activate_version

        def flip_then_activate(doc_id, new_version_id, *args, **kwargs):
            with psycopg.connect(env.pg) as conn:
                conn.execute(
                    f"UPDATE {VERSIONS} SET status = 'staging' "
                    "WHERE document_id = %s AND version_id = %s",
                    (doc_id, new_version_id),
                )
                conn.commit()
            return real_activate(doc_id, new_version_id, *args, **kwargs)

        monkeypatch.setattr(env.versions, "activate_version", flip_then_activate)
        tid = _ingest_via_queue(env)
        row = env.tasks.get(tid)
        assert row["status"] == "failed"
        assert row["current_stage"] == "activate"
        assert "不可激活" in row["error_message"]

        # 旧版本继续服务：v1 active、is_active 未变、revision 未递增
        assert env.versions.get_active(document_id)["version_id"] == v1
        assert all(_active_flags(env.pg, SMALL, document_id, v1))
        assert all(_active_flags(env.pg, LARGE, document_id, v1))
        # v2 staging 数据不上线
        assert _active_flags(env.pg, SMALL, document_id, v2) == [False] * 3
        assert _active_flags(env.pg, LARGE, document_id, v2) == [False] * 2
        assert env.kb.get(KB)["active_revision"] == 1
        # file_state 仍指向 v1
        assert env.files.get(str(env.file))["active_version_id"] == v1


# ── 修改文件：新版本激活、旧版本退役 ────────────────────────


class TestPgReingestModified:
    def test_new_version_activates_old_retires(self, env):
        _ingest_via_queue(env)
        document_id, v1 = _doc_ver(env)
        assert env.kb.get(KB)["active_revision"] == 1

        env.file.write_bytes(b"hello v2 modified content")
        _, v2 = _doc_ver(env)
        assert v2 != v1
        _ingest_via_queue(env)

        # 版本状态：v1 retired、v2 active
        assert env.versions.get(document_id, v1)["status"] == "retired"
        assert env.versions.get(document_id, v2)["status"] == "active"
        assert env.versions.get_document(document_id)["active_version_id"] == v2

        # embedding is_active 翻转
        assert _active_flags(env.pg, SMALL, document_id, v1) == [False] * 3
        assert _active_flags(env.pg, LARGE, document_id, v1) == [False] * 2
        assert all(_active_flags(env.pg, SMALL, document_id, v2))
        assert all(_active_flags(env.pg, LARGE, document_id, v2))

        # file_state / revision 更新
        fs = env.files.get(str(env.file))
        assert fs["active_version_id"] == v2
        assert env.kb.get(KB)["active_revision"] == 2


# ── 删除（逻辑删除事务）─────────────────────────────────────


class TestPgDelete:
    def test_logical_delete(self, env):
        _ingest_via_queue(env)
        document_id, version_id = _doc_ver(env)
        source = str(env.file)
        rev_before = env.kb.get(KB)["active_revision"]

        tid = _enqueue_and_process(
            env, "delete_document", {"source": source}, f"delete:{document_id}"
        )
        assert env.tasks.get(tid)["status"] == "completed"

        # 检索层面 is_active 全 false；版本 retired；文档 deleted
        assert _active_flags(env.pg, SMALL, document_id, version_id) == [False] * 3
        assert _active_flags(env.pg, LARGE, document_id, version_id) == [False] * 2
        assert env.versions.get(document_id, version_id)["status"] == "retired"
        doc = env.versions.get_document(document_id)
        assert doc["status"] == "deleted"
        assert doc["active_version_id"] is None
        assert env.files.get(source)["status"] == "deleted"
        assert env.kb.get(KB)["active_revision"] == rev_before + 1

        # 重复删除幂等：仍 completed，revision 不再递增
        tid2 = _enqueue_and_process(
            env, "delete_document", {"source": source}, f"delete:{document_id}"
        )
        assert env.tasks.get(tid2)["status"] == "completed"
        assert env.kb.get(KB)["active_revision"] == rev_before + 1

    def test_delete_failure_keeps_file_state(self, env, monkeypatch):
        """delete_document 抛错 → 任务 failed，file_state 不得被标记 deleted。"""
        _ingest_via_queue(env)
        document_id, _ = _doc_ver(env)
        source = str(env.file)
        assert env.files.get(source)["status"] == "active"

        def boom(*args, **kwargs):
            raise RuntimeError("delete tx boom")

        monkeypatch.setattr(env.versions, "delete_document", boom)
        tid = _enqueue_and_process(
            env, "delete_document", {"source": source}, f"delete:{document_id}"
        )
        row = env.tasks.get(tid)
        assert row["status"] == "failed"
        assert "delete tx boom" in row["error_message"]
        # 文件状态未被标记 deleted，文档仍 active
        assert env.files.get(source)["status"] == "active"
        assert env.versions.get_document(document_id)["status"] == "active"


# ── 心跳（长任务不被抢占）───────────────────────────────────


class TestPgHeartbeat:
    def test_heartbeat_during_long_task(self, env):
        env.worker._lease_seconds = 1  # 短租约，心跳间隔 1/3s
        env.pipeline.sleep_seconds = 0.8
        document_id, version_id = _doc_ver(env)

        tid = env.tasks.enqueue(
            "ingest_file",
            {"file_path": str(env.file)},
            f"ingest:{document_id}:{version_id}",
            max_retries=1,
        )
        claimed = env.tasks.claim(env.worker._worker_id, lease_seconds=1)
        with mock.patch.object(env.tasks, "heartbeat", wraps=env.tasks.heartbeat) as hb:
            env.worker._process_pg(claimed)
        # 慢 pipeline 期间心跳被调用（不被其他 Worker 抢占）
        assert hb.call_count >= 1
        assert env.tasks.get(tid)["status"] == "completed"


# ── 扫描器（Spec 6.1：只创建任务）───────────────────────────


class TestPgScan:
    def test_scan_only_enqueues(self, env):
        doc_id, version_id = _doc_ver(env)
        source = str(env.file)

        # 新文件 → ingest 任务；扫描不得提前写 file_state
        ids = pg_scan_and_enqueue(
            str(env.tmp_path), [".md"], env.tasks, env.files, KB
        )
        assert len(ids) == 1
        task = env.tasks.get(ids[0])
        assert task["task_type"] == "ingest_file"
        assert task["payload"]["file_path"] == source
        assert env.files.get(source) is None

        # 同文件同内容重复扫描 → dedup 命中，不重复入队
        assert (
            pg_scan_and_enqueue(str(env.tmp_path), [".md"], env.tasks, env.files, KB)
            == []
        )

        # 处理完成（激活）后再次扫描 → 快速路径 unchanged，无新任务
        claimed = env.tasks.claim(env.worker._worker_id, lease_seconds=60)
        assert claimed["task_id"] == ids[0]
        env.worker._process_pg(claimed)
        assert env.files.get(source)["active_version_id"] == version_id
        assert (
            pg_scan_and_enqueue(str(env.tmp_path), [".md"], env.tasks, env.files, KB)
            == []
        )

    def test_scan_enqueues_delete_for_vanished_file(self, env):
        _ingest_via_queue(env)
        document_id, _ = _doc_ver(env)

        env.file.unlink()
        ids = pg_scan_and_enqueue(
            str(env.tmp_path), [".md"], env.tasks, env.files, KB
        )
        assert len(ids) == 1
        task = env.tasks.get(ids[0])
        assert task["task_type"] == "delete_document"
        assert task["dedup_key"] == f"delete:{document_id}"
        # 扫描不提前标记 deleted（删除事务成功后才改）
        assert env.files.get(str(env.file))["status"] == "active"
        # 收尾：清掉 pending 任务，避免影响其他用例的 claim 断言
        claimed = env.tasks.claim(env.worker._worker_id, lease_seconds=60)
        env.tasks.complete(claimed["task_id"], env.worker._worker_id)

    def test_scan_detects_content_change_not_mtime_only(self, env):
        _ingest_via_queue(env)
        document_id, _ = _doc_ver(env)

        # 只 touch（mtime 变、内容不变）→ 内容 hash 二次确认后跳过
        os.utime(env.file, (time.time() + 100, time.time() + 100))
        assert (
            pg_scan_and_enqueue(str(env.tmp_path), [".md"], env.tasks, env.files, KB)
            == []
        )

        # 内容真变 → 新 ingest 任务（新 dedup_key，含完整 content_hash）
        env.file.write_bytes(b"changed content")
        ids = pg_scan_and_enqueue(
            str(env.tmp_path), [".md"], env.tasks, env.files, KB
        )
        assert len(ids) == 1
        new_hash = versioning.compute_content_hash(b"changed content")
        assert env.tasks.get(ids[0])["dedup_key"] == f"ingest:{document_id}:{new_hash}"


# ── 写入接口统一（Spec 11，TestClient + fake）───────────────

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.gateway.app import app  # noqa: E402
from app.gateway.auth.dependencies import get_current_user  # noqa: E402
from app.gateway.auth.schemas import UserInfo  # noqa: E402
from app.gateway.routers import rag as rag_router  # noqa: E402
from agent_material.utils import config_handler  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


def _fake_user() -> UserInfo:
    return UserInfo(user_id="test-u", username="tester", created_at=time.time())


class FakeStore:
    """实现写入接口统一后端点用到的 VectorStore 接口。"""

    def __init__(self):
        self.text_tasks: list[tuple[str, str]] = []
        self.load_document_calls = 0
        self._next_id = 100

    def load_document(self):
        self.load_document_calls += 1
        return [11, 12, 13]

    def enqueue_ingest_text(self, text, source):
        self.text_tasks.append((text, source))
        self._next_id += 1
        return self._next_id

    def get_task_status(self, task_id):
        if task_id == 999:
            return None
        return {"task_id": task_id, "status": "completed", "current_stage": None}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def authed_client(client):
    app.dependency_overrides[get_current_user] = _fake_user
    try:
        yield client
    finally:
        app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture()
def fake_store(monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(rag_router, "_get_vector_store", lambda: store)
    return store


def _set_flags(monkeypatch, **flags):
    monkeypatch.setattr(
        config_handler, "vector_store_config", {"feature_flags": flags}
    )


class TestWriteApiUnification:
    def test_load_documents_returns_task_ids(self, authed_client, fake_store):
        resp = authed_client.post("/api/v1/rag/load-documents")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 200
        assert body["data"]["task_ids"] == [11, 12, 13]
        assert body["data"]["accepted"] == 3
        # 不再暗示"加载完成"
        assert "任务" in body["message"] and "加载完成" not in body["message"]
        assert fake_store.load_document_calls == 1

    def test_add_chunks_default_410(self, authed_client, fake_store, monkeypatch):
        _set_flags(monkeypatch, enable_add_chunks=False, pg_control_plane=True)
        resp = authed_client.post("/api/v1/rag/add-chunks", json={"chunks": ["x"]})
        assert resp.status_code == 410
        assert "/rag/add-documents" in resp.json()["detail"]

    def test_add_documents_enqueues_ingest_text(
        self, authed_client, fake_store, monkeypatch
    ):
        _set_flags(monkeypatch, pg_control_plane=True)
        resp = authed_client.post(
            "/api/v1/rag/add-documents",
            json={"documents": ["文本一", "文本二"], "sources": ["s1", "s2"]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 200
        assert body["data"]["accepted"] == 2
        assert len(body["data"]["task_ids"]) == 2
        assert fake_store.text_tasks == [("文本一", "s1"), ("文本二", "s2")]

    def test_task_status_endpoint(self, authed_client, fake_store):
        resp = authed_client.get("/api/v1/rag/tasks/101")
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["task_id"] == 101
        assert body["data"]["status"] == "completed"

        resp404 = authed_client.get("/api/v1/rag/tasks/999")
        assert resp404.status_code == 404
