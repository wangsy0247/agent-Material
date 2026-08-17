"""入口统一测试（Spec V3 阶段 A：ENT-001~005 / T-ENT-01~06）。

- PG 集成部分：真实 PG（127.0.0.1:5433，连接串读自 config/vector_store_config.yaml）
  + table_prefix="test_ent_" 隔离表，测试后 DROP；DB 不可达时相关用例 skip；
- REST / Agent Tool 部分：TestClient + fake store，验证两个入口
  都收敛到同一个 RagIndexingService / 同一张 PG 任务表；
- 直写封禁部分：VectorStore.__new__ 裸实例 + 开关替换，不触碰真实 DB。
"""

import sys
import time
from pathlib import Path

# 规避已知 C 扩展加载顺序冲突：sqlalchemy 必须先于 psycopg 完成导入
# （与 test_pg_worker.py 相同，WSL/Anaconda 环境导入期 Segfault 规避）。
import sqlalchemy  # noqa: F401
import psycopg
import pytest
import yaml
from langchain_core.documents import Document

from agent_material.rag_engine import versioning
from agent_material.rag_engine.db.pg import (
    TaskRepository,
    VersionRepository,
    drop_schema,
    ensure_schema,
)
from agent_material.rag_engine.db.pg.schema import normalize_conn_str
from agent_material.rag_engine.indexing_service import (
    DirectWriteForbiddenError,
    RagIndexingService,
)

ROOT = Path(__file__).resolve().parents[2]
from agent_material.utils.config_handler import get_vector_store_config

CONN_STR = normalize_conn_str(get_vector_store_config()["connection_string"])
PREFIX = "test_ent_"
KB = "agent_test_ent"


def _pg_available() -> bool:
    try:
        with psycopg.connect(CONN_STR, connect_timeout=3):
            return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def pg():
    if not _pg_available():
        pytest.skip("PostgreSQL 127.0.0.1:5433 不可达")
    drop_schema(CONN_STR, PREFIX)
    ensure_schema(CONN_STR, PREFIX)
    yield CONN_STR
    drop_schema(CONN_STR, PREFIX)


@pytest.fixture()
def service(pg):
    """测试前缀的 RagIndexingService（不绑定 Worker，任务停留 pending）。"""
    return RagIndexingService(conn_str=pg, kb_id=KB, table_prefix=PREFIX)


def _count_test_embeddings(conn_str: str) -> int:
    """测试知识库 collection 中的向量条数（应始终为 0 —— 入口不得直写）。"""
    with psycopg.connect(conn_str) as conn:
        return conn.execute(
            r"""
            SELECT COUNT(*)
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
            WHERE c.name LIKE 'agent\_test\_ent%'
            """
        ).fetchone()[0]


# ── REST / Agent Tool 公共脚手架 ──────────────────────────

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.gateway.app import app  # noqa: E402
from app.gateway.auth.dependencies import get_current_user  # noqa: E402
from app.gateway.auth.schemas import UserInfo  # noqa: E402
from app.gateway.routers import rag as rag_router  # noqa: E402
from agent_material.agents.tools import rag_local_tools  # noqa: E402
from agent_material.utils import config_handler  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


def _fake_user() -> UserInfo:
    return UserInfo(
        user_id="test-u", username="tester", created_at=time.time(), is_admin=True
    )


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


def _pg_flag_on(monkeypatch):
    monkeypatch.setattr(
        config_handler,
        "vector_store_config",
        {"feature_flags": {"pg_control_plane": True}},
    )


class _RecordingStore:
    """记录型 fake store：写删方法委托给注入的 service，直写方法记调用。"""

    def __init__(self, svc=None):
        self._svc = svc
        self.add_documents_called = False
        self.deleted_sources = []
        self._next_id = 500

    # ── pg 模式统一入口 ──
    def enqueue_ingest_text(self, text, source):
        if self._svc is not None:
            return self._svc.enqueue_text(text, source)
        self._next_id += 1
        return self._next_id

    def enqueue_delete_by_source(self, source):
        if self._svc is not None:
            return self._svc.enqueue_delete(source)
        self._next_id += 1
        return {
            "source": source,
            "canonical_source": source,
            "task_id": self._next_id,
            "status": "accepted",
        }

    def get_document_count(self):
        return 7

    def load_document(self):
        return [1, 2]

    def get_task_status(self, task_id):
        return {"task_id": task_id, "status": "pending"}

    def get_task_summary(self):
        return {"pending": 1, "processing": 0, "completed": 2, "failed": 0}

    # ── 旧直写路径（pg 模式不得被调用）──
    def add_documents(self, docs):
        self.add_documents_called = True

    def delete_documents_by_source(self, source):
        self.deleted_sources.append(source)


# ================================================================
# ENT-002：统一 source 契约（canonicalizer 单测，无需 PG）
# ================================================================


class TestSourceCanonicalizer:
    def test_manual_source_gets_prefix(self):
        assert versioning.canonicalize_source("manual_0") == "manual://manual_0"
        assert versioning.canonicalize_source("  note_1 ") == "manual://note_1"
        # 已带 scheme 的逻辑标识原样保留
        assert versioning.canonicalize_source("manual://x") == "manual://x"

    def test_file_source_normalized(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert versioning.canonicalize_source("data/a.pdf") == (
            versioning.normalize_source_path(str(tmp_path / "data" / "a.pdf"))
        )

    def test_relative_and_absolute_same_document_id(self, tmp_path, monkeypatch):
        """P0 不接受同一来源因相对/绝对路径差异生成多个 document_id。"""
        monkeypatch.chdir(tmp_path)
        assert versioning.compute_document_id(
            "data/a.pdf", kb_id=KB
        ) == versioning.compute_document_id(str(tmp_path / "data" / "a.pdf"), kb_id=KB)

    def test_manual_source_not_cwd_dependent(self, tmp_path, monkeypatch):
        """manual:// 逻辑标识不得依赖当前工作目录。"""
        monkeypatch.chdir(tmp_path)
        id1 = versioning.compute_document_id("manual://n", kb_id=KB)
        monkeypatch.chdir(tmp_path.parent)
        id2 = versioning.compute_document_id("manual://n", kb_id=KB)
        assert id1 == id2

    def test_empty_source_rejected(self):
        with pytest.raises(ValueError, match="source 不能为空"):
            versioning.canonicalize_source("   ")


# ================================================================
# T-ENT-01：REST add 只创建 PG 任务、不直接增加向量
# ================================================================


class TestEnT01AddOnlyEnqueues:
    def test_service_enqueue_text_creates_pg_task_only(self, pg, service):
        task_id = service.enqueue_text("氧化镓热导率片段", "ent01_note")
        assert isinstance(task_id, int)
        row = TaskRepository(pg, PREFIX).get(task_id)
        assert row["task_type"] == "ingest_text"
        assert row["status"] == "pending"
        # source 经 canonicalizer 加 manual:// 前缀（ENT-002）
        assert row["payload"]["source"] == "manual://ent01_note"
        # dedup：同内容同来源不重复入队
        assert service.enqueue_text("氧化镓热导率片段", "ent01_note") is None
        # 不直接写 PGVector
        assert _count_test_embeddings(pg) == 0

    def test_rest_add_only_enqueues(self, authed_client, monkeypatch):
        _pg_flag_on(monkeypatch)
        store = _RecordingStore()
        monkeypatch.setattr(rag_router, "_get_vector_store", lambda: store)
        resp = authed_client.post(
            "/api/v1/rag/add-documents",
            json={"documents": ["文本一", "文本二"], "sources": ["s1", "s2"]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 200
        assert body["data"]["accepted"] == 2
        assert body["data"]["deduplicated"] == 0
        assert len(body["data"]["task_ids"]) == 2
        assert "已接收" in body["message"]
        # 兼容直写方法未被调用（ENT-003）
        assert store.add_documents_called is False


# ================================================================
# T-ENT-02：REST delete 返回 accepted/task_id、不同步物理删除
# ================================================================


class TestEnT02DeleteEnqueuesOnly:
    def test_service_enqueue_delete_no_physical_delete(self, pg, service):
        result = service.enqueue_delete("data/ent02_never_ingested.pdf")
        assert result["status"] == "accepted"
        assert isinstance(result["task_id"], int)
        row = TaskRepository(pg, PREFIX).get(result["task_id"])
        assert row["task_type"] == "delete_document"
        # 删除与入库使用同一 canonical source（ENT-002）
        assert row["payload"]["source"] == result["canonical_source"]
        assert row["dedup_key"] == (
            f"delete:{versioning.compute_document_id(result['canonical_source'], kb_id=KB)}"
        )
        # source 从未入库也允许创建幂等删除任务 —— 版本表无任何记录，
        # 且入口本身不做任何物理删除（由 Worker 异步 no-op 完成）
        document_id = versioning.compute_document_id(
            result["canonical_source"], kb_id=KB
        )
        assert VersionRepository(pg, PREFIX).get_document(document_id) is None
        assert _count_test_embeddings(pg) == 0

    def test_rest_delete_returns_accepted_with_task_id(self, authed_client, monkeypatch):
        _pg_flag_on(monkeypatch)
        store = _RecordingStore()
        monkeypatch.setattr(rag_router, "_get_vector_store", lambda: store)
        resp = authed_client.post(
            "/api/v1/rag/delete-by-source", json={"source": "data/x.pdf"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "已接收" in body["message"]
        assert "已删除" not in body["message"]
        assert body["data"]["status"] == "accepted"
        assert isinstance(body["data"]["task_id"], int)
        # 不同步物理删除
        assert store.deleted_sources == []


# ================================================================
# T-ENT-03：批量删除每个 source 独立结果和 task_id
# ================================================================


class TestEnT03BatchDelete:
    def test_batch_delete_independent_results(self, pg, service):
        results = service.enqueue_batch_delete(
            ["data/ent03_a.pdf", "data/ent03_b.pdf", "data/ent03_a.pdf", ""]
        )
        assert len(results) == 4
        accepted = [r for r in results if r["status"] == "accepted"]
        deduplicated = [r for r in results if r["status"] == "deduplicated"]
        failed = [r for r in results if r["status"] == "failed"]
        # 两个新 source 各自独立任务；重复 source 去重；空 source 参数失败
        assert len(accepted) == 2
        assert len(deduplicated) == 1
        assert deduplicated[0]["task_id"] is None
        assert len(failed) == 1
        assert failed[0]["error"]
        task_ids = {r["task_id"] for r in accepted}
        assert len(task_ids) == 2
        assert None not in task_ids
        # 批量入队同样不做物理删除
        assert _count_test_embeddings(pg) == 0


# ================================================================
# T-ENT-04：Agent Tool add/delete 与 REST 同一 RagIndexingService
# ================================================================


class TestEnT04AgentToolsShareService:
    def test_rest_and_agent_tool_same_pg_task_table(
        self, pg, service, authed_client, monkeypatch
    ):
        """REST 与 Agent Tool 的任务落在同一张 PG 任务表。"""
        store = _RecordingStore(svc=service)
        monkeypatch.setattr(rag_router, "_get_vector_store", lambda: store)
        monkeypatch.setattr(rag_local_tools, "_get_vector_store", lambda: store)
        _pg_flag_on(monkeypatch)

        # REST 入口
        resp = authed_client.post(
            "/api/v1/rag/add-documents",
            json={"documents": ["REST 入口文本"], "sources": ["ent04_rest"]},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["accepted"] == 1

        # Agent Tool 入口
        out = rag_local_tools.rag_add_documents.invoke(
            {"documents": ["Agent 工具文本"], "sources": ["ent04_tool"]}
        )
        assert "任务已接收" in out
        assert "task_ids=" in out
        out_del = rag_local_tools.rag_delete_by_source.invoke(
            {"source": "ent04_tool"}
        )
        assert "删除任务已接收" in out_del
        assert "task_id=" in out_del

        # 两个入口的任务都在同一张（测试前缀）PG 任务表
        repo = TaskRepository(pg, PREFIX)
        rows = [
            r
            for r in repo.list_by_status("pending", limit=1000)
            if "ent04" in str(r["payload"])
        ]
        assert sorted(r["task_type"] for r in rows) == [
            "delete_document",
            "ingest_text",
            "ingest_text",
        ]
        # 直写路径未被触发
        assert store.add_documents_called is False
        assert store.deleted_sources == []

    def test_vector_store_enqueue_delegates_to_indexing_service(self, monkeypatch):
        """VectorStore 的入队/查询方法全部委托 RagIndexingService（ENT-001）。"""
        import agent_material.rag_engine.vector_store as vs_module

        calls = []

        class _FakeService:
            def enqueue_text(self, text, source):
                calls.append(("text", text, source))
                return 7

            def enqueue_delete(self, source):
                calls.append(("delete", source))
                return {
                    "source": source,
                    "canonical_source": "manual://x",
                    "task_id": 9,
                    "status": "accepted",
                }

            def enqueue_batch_delete(self, sources):
                return [self.enqueue_delete(s) for s in sources]

            def get_task(self, task_id):
                return {"task_id": task_id, "status": "pending"}

            def get_task_summary(self):
                return {"pending": 1, "processing": 0, "completed": 0, "failed": 0}

        fake = _FakeService()
        monkeypatch.setattr(
            vs_module, "get_indexing_service", lambda ensure_worker=None: fake
        )
        monkeypatch.setattr(
            vs_module,
            "vector_store_config",
            {"feature_flags": {"pg_control_plane": True}},
        )
        vs = vs_module.VectorStore.__new__(vs_module.VectorStore)
        vs._indexing_service = None

        assert vs.enqueue_ingest_text("t", "s") == 7
        assert vs.enqueue_delete_by_source("s")["task_id"] == 9
        assert len(vs.enqueue_batch_delete(["a", "b"])) == 2
        assert vs.get_task_status(1)["status"] == "pending"
        assert vs.get_task_summary()["pending"] == 1
        assert calls == [
            ("text", "t", "s"),
            ("delete", "s"),
            ("delete", "a"),
            ("delete", "b"),
        ]


# ================================================================
# T-ENT-05：全链路不创建 RAG SQLite 控制面
# ================================================================


class TestEnT05NoSqliteQueueInPgMode:
    def test_agent_tools_and_store_never_touch_sqlite(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        store = _RecordingStore()
        monkeypatch.setattr(rag_local_tools, "_get_vector_store", lambda: store)

        # 全部写删/查询类工具跑一遍
        assert "任务已接收" in rag_local_tools.rag_add_documents.invoke(
            {"documents": ["x"], "sources": ["s"]}
        )
        assert "删除任务已接收" in rag_local_tools.rag_delete_by_source.invoke(
            {"source": "data/x.pdf"}
        )
        assert "队列状态" in rag_local_tools.rag_load_documents.invoke({})
        assert "pending" in rag_local_tools.rag_get_task_status.invoke({"task_id": 1})
        assert "pending" in rag_local_tools.rag_get_pending_tasks.invoke({})

        # 全程未创建 SQLite 队列库文件
        assert not (tmp_path / "rag_queue.db").exists()


# ================================================================
# T-ENT-06：兼容直写方法始终明确拒绝
# ================================================================


class TestEnT06DirectWriteForbidden:
    def test_direct_write_raises(self, monkeypatch):
        import agent_material.rag_engine.vector_store as vs_module

        vs = vs_module.VectorStore.__new__(vs_module.VectorStore)
        with pytest.raises(DirectWriteForbiddenError, match="ENT-005"):
            vs.add_documents([Document(page_content="x", metadata={"source": "s"})])
        with pytest.raises(DirectWriteForbiddenError, match="ENT-005"):
            vs.delete_documents_by_source("data/x.pdf")

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
