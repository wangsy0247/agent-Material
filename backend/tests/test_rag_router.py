"""RAG 路由（/api/v1/rag/*）端点测试。

覆盖 9 个端点：GET /stats、POST /search、POST /query（SSE 流式）、
POST /load-documents、POST /add-documents、POST /add-chunks、
GET /sources、POST /delete-by-source、POST /delete-sources。

mock 策略：
- 认证：dependency_overrides 注入 fake user（user_id="test-u"）
- rag.py 模块级单例 _get_vector_store / _get_rag_service 用 monkeypatch
  替换为返回 fake 对象的函数（端点内部通过模块全局名调用，补丁生效）
- /stats 读取的 vector_store_config（config_handler 模块级字典）同样
  用 monkeypatch 替换为纯 dict，不触碰真实配置或 PGVector
"""

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# backend/ 与 backend/packages 加入路径（pyproject 仅配置了 packages）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document  # noqa: E402

from app.gateway.app import app  # noqa: E402
from app.gateway.auth.dependencies import get_current_user  # noqa: E402
from app.gateway.auth.schemas import UserInfo  # noqa: E402
from app.gateway.routers import rag as rag_router  # noqa: E402
from agent_material.utils import config_handler  # noqa: E402


def _fake_user() -> UserInfo:
    return UserInfo(
        user_id="test-u",
        username="tester",
        created_at=time.time(),
        is_admin=True,
    )


def _fake_standard_user() -> UserInfo:
    return UserInfo(
        user_id="test-standard",
        username="standard",
        created_at=time.time(),
        is_admin=False,
    )


# ------------------------------------------------------------------
# fake 对象
# ------------------------------------------------------------------


class FakeVectorStore:
    """仅实现 rag.py 端点用到的 VectorStore 接口。"""

    def __init__(self, document_count: int = 42):
        self._document_count = document_count
        self.load_document_calls = 0
        self.enqueued_texts = []
        self.enqueued_files = []
        self.enqueued_deletes = []
        self.enqueue_fail_sources = set()
        self._next_task_id = 1000
        self.retried_tasks = []

    def get_document_count(self):
        return self._document_count

    def load_document(self):
        self.load_document_calls += 1
        self._document_count += 3  # 模拟加载了 3 条
        return [1, 2, 3]  # Spec V2：返回已入队 task_id 列表

    def enqueue_ingest_text(self, text, source):
        self._next_task_id += 1
        self.enqueued_texts.append((text, source))
        return self._next_task_id

    def enqueue_ingest_file(self, file_path):
        self._next_task_id += 1
        self.enqueued_files.append(file_path)
        return self._next_task_id

    def get_all_sources(self):
        return [
            {"source": "data/a.pdf", "count": 10},
            {"source": "data/b.pdf", "count": 5},
        ]

    def enqueue_delete_by_source(self, source):
        """pg 模式删除入口（ENT-003）：模拟创建 delete_document 任务。"""
        if source in self.enqueue_fail_sources:
            raise RuntimeError(f"创建删除任务失败: {source}")
        self._next_task_id += 1
        self.enqueued_deletes.append(source)
        return {
            "source": source,
            "canonical_source": f"manual://{source}",
            "task_id": self._next_task_id,
            "status": "accepted",
        }

    def enqueue_batch_delete(self, sources):
        results = []
        for s in sources:
            try:
                results.append(self.enqueue_delete_by_source(s))
            except Exception as e:
                results.append({
                    "source": s,
                    "canonical_source": None,
                    "task_id": None,
                    "status": "failed",
                    "error": str(e),
                })
        return results

    def retry_failed_task(self, task_id):
        self.retried_tasks.append(task_id)
        return {"task_id": task_id, "status": "pending", "retries": 0}


class FakeRagService:
    """仅实现 rag.py 端点用到的 RagService 接口。"""

    def __init__(self, docs=None, stream_chunks=None):
        self._docs = docs if docs is not None else [
            Document(
                page_content="β-Ga2O3 的热导率研究片段",
                metadata={"source": "data/ga2o3.pdf", "score": 0.85},
            )
        ]
        self._stream_chunks = stream_chunks if stream_chunks is not None else ["你好", "世界"]
        self.retriever_queries = []

    # ── 新路径（single_retrieval_sse 开启）──

    def prepare(self, query, enable_rewrite=False, rewrite_strategy=""):
        self.retriever_queries.append(query)
        return FakeRequestContext(query, self._docs)

    def stream_answer(self, ctx):
        yield from self._stream_chunks

    # ── 旧路径（single_retrieval_sse 关闭时回滚使用）──

    def retriever_docx(self, query):
        self.retriever_queries.append(query)
        return self._docs

    def get_summary_stream(self, query):
        yield from self._stream_chunks


class FakeRequestContext:
    """RequestContext 替身：携带 sources / resolved_parents / 冻结上下文。"""

    def __init__(self, query, docs):
        self.original_query = query
        self.route_decision = "rag"
        self.context_docs = list(docs)
        self.resolved_parents = list(docs)
        self.reranked_children = list(docs)
        self.degradation_events = []
        self.final_context = (
            "".join(f"[参考资料{i}] 来源：x\n{d.page_content}\n\n"
                    for i, d in enumerate(docs, start=1))
        )
        self.sources = [
            {
                "index": i,
                "source": d.metadata.get("source", "未知来源"),
                "source_filename": d.metadata.get("source", "未知来源"),
                "page_start": d.metadata.get("page_start"),
                "page_end": d.metadata.get("page_end"),
                "page_unknown": False,
                "heading_path": None,
                "parent_id": d.metadata.get("parent_id"),
                "score": d.metadata.get("score", 0.0),
                "snippet": d.page_content,
            }
            for i, d in enumerate(docs, start=1)
        ]

    def summary(self):
        return {"request_id": "fake"}


# ------------------------------------------------------------------
# fixtures
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    # TestClient 启动会初始化 RAG 组件，较慢，模块级只建一次
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
def standard_user_client(client):
    app.dependency_overrides[get_current_user] = _fake_standard_user
    try:
        yield client
    finally:
        app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture()
def fake_store(monkeypatch):
    """替换 _get_vector_store 单例工厂，返回 FakeVectorStore。"""
    store = FakeVectorStore()
    monkeypatch.setattr(rag_router, "_get_vector_store", lambda: store)
    return store


@pytest.fixture()
def fake_service(monkeypatch):
    """替换 _get_rag_service 单例工厂，返回 FakeRagService。"""
    service = FakeRagService()
    monkeypatch.setattr(rag_router, "_get_rag_service", lambda: service)
    return service


# ------------------------------------------------------------------
# 未认证一律 401
# ------------------------------------------------------------------


class TestRagAuth:
    def test_stats_unauthenticated_401(self, client):
        assert client.get("/api/v1/rag/stats").status_code == 401

    def test_search_unauthenticated_401(self, client):
        resp = client.post("/api/v1/rag/search", json={"query": "Ga2O3"})
        assert resp.status_code == 401

    def test_query_unauthenticated_401(self, client):
        resp = client.post("/api/v1/rag/query", json={"query": "Ga2O3"})
        assert resp.status_code == 401

    def test_load_documents_unauthenticated_401(self, client):
        assert client.post("/api/v1/rag/load-documents").status_code == 401

    def test_add_documents_unauthenticated_401(self, client):
        resp = client.post("/api/v1/rag/add-documents", json={"documents": ["x"]})
        assert resp.status_code == 401

    def test_add_chunks_unauthenticated_401(self, client):
        resp = client.post("/api/v1/rag/add-chunks", json={"chunks": ["x"]})
        assert resp.status_code == 401

    def test_sources_unauthenticated_401(self, client):
        assert client.get("/api/v1/rag/sources").status_code == 401

    def test_delete_by_source_unauthenticated_401(self, client):
        resp = client.post("/api/v1/rag/delete-by-source", json={"source": "a.pdf"})
        assert resp.status_code == 401

    def test_delete_sources_unauthenticated_401(self, client):
        resp = client.post("/api/v1/rag/delete-sources", json={"sources": ["a.pdf"]})
        assert resp.status_code == 401

    @pytest.mark.parametrize(
        ("method", "path", "kwargs"),
        [
            ("post", "/api/v1/rag/load-documents", {}),
            (
                "post",
                "/api/v1/rag/add-documents",
                {"json": {"documents": ["x"]}},
            ),
            (
                "post",
                "/api/v1/rag/delete-by-source",
                {"json": {"source": "a.pdf"}},
            ),
        ],
    )
    def test_knowledge_base_mutation_requires_admin(
        self, standard_user_client, method, path, kwargs
    ):
        response = getattr(standard_user_client, method)(path, **kwargs)
        assert response.status_code == 403


# ------------------------------------------------------------------
# GET /stats
# ------------------------------------------------------------------


class TestRagStats:
    def test_stats_response_shape(self, authed_client, fake_store, monkeypatch):
        monkeypatch.setattr(
            config_handler,
            "vector_store_config",
            {
                "connection_string": "postgresql://user:secret@dbhost:5432/mydb",
                "chunk_size": 500,
                "chunk_overlap": 50,
            },
        )
        resp = authed_client.get("/api/v1/rag/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 200
        data = body["data"]
        assert data["document_count"] == 42
        assert data["collection_name"] == "agent"  # character_name 缺省回退
        assert data["splitter_type"] == "recursive"
        assert data["chunk_size"] == 500
        assert data["chunk_overlap"] == 50
        assert data["vector_weight"] == 0.7
        assert data["bm25_weight"] == 0.3
        assert data["enable_reranker"] is True
        # 连接串密码被掩码
        assert data["connection_string_masked"] == "postgresql://user:****@dbhost:5432/mydb"
        assert "secret" not in data["connection_string_masked"]

    def test_stats_store_error_500(self, authed_client, monkeypatch):
        def _boom():
            raise RuntimeError("vector store init failed")

        monkeypatch.setattr(rag_router, "_get_vector_store", _boom)
        resp = authed_client.get("/api/v1/rag/stats")
        assert resp.status_code == 500
        assert "获取统计信息失败" in resp.json()["detail"]


# ------------------------------------------------------------------
# POST /search
# ------------------------------------------------------------------


class TestRagSearch:
    def test_search_with_results(self, authed_client, fake_service):
        resp = authed_client.post("/api/v1/rag/search", json={"query": "热导率"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 200
        # BaseResponse.total 字段已补齐，列表响应应携带总数
        assert body["total"] == 1
        assert len(body["data"]) == 1
        item = body["data"][0]
        assert item["index"] == 1
        assert item["source"] == "data/ga2o3.pdf"
        assert item["score"] == 0.85
        assert item["snippet"] == "β-Ga2O3 的热导率研究片段"
        assert item["metadata"]["source"] == "data/ga2o3.pdf"
        assert fake_service.retriever_queries == ["热导率"]

    def test_search_empty_results(self, authed_client, monkeypatch):
        service = FakeRagService(docs=[])
        monkeypatch.setattr(rag_router, "_get_rag_service", lambda: service)
        resp = authed_client.post("/api/v1/rag/search", json={"query": "不存在的内容"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 200
        assert body["message"] == "未检索到相关文档"
        assert body["data"] == []

    def test_search_service_error_500(self, authed_client, monkeypatch):
        class _ErrService(FakeRagService):
            def prepare(self, query, enable_rewrite=False, rewrite_strategy=""):
                raise RuntimeError("检索后端异常")

        monkeypatch.setattr(rag_router, "_get_rag_service", lambda: _ErrService())
        resp = authed_client.post("/api/v1/rag/search", json={"query": "x"})
        assert resp.status_code == 500
        assert "检索失败" in resp.json()["detail"]


# ------------------------------------------------------------------
# POST /query（SSE 流式）
# ------------------------------------------------------------------


class TestRagQuery:
    def test_query_sse_full_flow(self, authed_client, fake_service):
        resp = authed_client.post("/api/v1/rag/query", json={"query": "Ga2O3 热导率"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = resp.text
        # 事件序列：status(retrieving) -> start -> sources -> status(generating) -> token* -> done
        assert "event: status" in body
        assert '"phase": "retrieving"' in body or '"phase":"retrieving"' in body
        assert "event: start" in body
        assert "event: sources" in body
        assert "data/ga2o3.pdf" in body
        assert '"phase": "generating"' in body or '"phase":"generating"' in body
        assert "event: token" in body
        assert "你好" in body and "世界" in body
        assert "event: done" in body
        # token 帧带递增 index
        assert '"index": 1' in body or '"index":1' in body

    def test_query_no_docs_short_circuit(self, authed_client, monkeypatch):
        service = FakeRagService(docs=[])
        monkeypatch.setattr(rag_router, "_get_rag_service", lambda: service)
        resp = authed_client.post("/api/v1/rag/query", json={"query": "无关问题"})
        assert resp.status_code == 200
        body = resp.text
        assert "根据现有资料无法回答该问题。" in body
        assert "event: done" in body
        # 无文档时不进入生成阶段
        assert "generating" not in body

    def test_query_error_frame_on_exception(self, authed_client, monkeypatch):
        class _ErrService(FakeRagService):
            def prepare(self, query, enable_rewrite=False, rewrite_strategy=""):
                raise RuntimeError("检索爆炸")

        monkeypatch.setattr(rag_router, "_get_rag_service", lambda: _ErrService())
        resp = authed_client.post("/api/v1/rag/query", json={"query": "x"})
        assert resp.status_code == 200  # SSE 流内错误，HTTP 仍 200
        body = resp.text
        assert "event: error" in body
        assert "检索或生成失败" in body
        assert "检索爆炸" not in body
        assert "event: done" not in body


# ------------------------------------------------------------------
# POST /load-documents
# ------------------------------------------------------------------


class TestRagLoadDocuments:
    def test_load_documents_success(self, authed_client, fake_store):
        resp = authed_client.post("/api/v1/rag/load-documents")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 200
        # Spec V2 11："任务已接收"语义，不再暗示"加载完成"
        assert "加载完成" not in body["message"]
        data = body["data"]
        assert data["task_ids"] == [1, 2, 3]
        assert data["accepted"] == 3
        assert fake_store.load_document_calls == 1

    def test_load_documents_error_500(self, authed_client, monkeypatch):
        class _ErrStore(FakeVectorStore):
            def load_document(self):
                raise RuntimeError("磁盘读取失败")

        monkeypatch.setattr(rag_router, "_get_vector_store", lambda: _ErrStore())
        resp = authed_client.post("/api/v1/rag/load-documents")
        assert resp.status_code == 500
        assert "加载文档失败" in resp.json()["detail"]


# ------------------------------------------------------------------
# POST /add-documents
# ------------------------------------------------------------------


class TestRagAddDocuments:
    def test_add_documents_with_sources(self, authed_client, fake_store):
        resp = authed_client.post(
            "/api/v1/rag/add-documents",
            json={"documents": ["文本一", "文本二"], "sources": ["s1", "s2"]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 200
        assert body["data"]["accepted"] == 2
        assert body["data"]["deduplicated"] == 0
        assert fake_store.enqueued_texts == [("文本一", "s1"), ("文本二", "s2")]

    def test_add_documents_auto_sources(self, authed_client, fake_store):
        resp = authed_client.post(
            "/api/v1/rag/add-documents",
            json={"documents": ["文本一", "文本二"]},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["accepted"] == 2
        assert fake_store.enqueued_texts == [
            ("文本一", "manual_0"),
            ("文本二", "manual_1"),
        ]

    def test_add_documents_sources_length_mismatch_400(self, authed_client, fake_store):
        resp = authed_client.post(
            "/api/v1/rag/add-documents",
            json={"documents": ["a", "b"], "sources": ["only-one"]},
        )
        assert resp.status_code == 400
        assert "sources 长度必须与 documents 一致" in resp.json()["detail"]

    def test_add_documents_empty_list_422(self, authed_client):
        resp = authed_client.post("/api/v1/rag/add-documents", json={"documents": []})
        assert resp.status_code == 422


# ------------------------------------------------------------------
# POST /add-chunks
# ------------------------------------------------------------------


class TestRagAddChunks:
    def test_add_chunks_removed(self, authed_client, fake_store):
        resp = authed_client.post(
            "/api/v1/rag/add-chunks",
            json={
                "chunks": ["chunk-a", "chunk-b"],
                "sources": ["doc1.pdf", "doc1.pdf"],
                "metadatas": [{"page": 1}, {"page": 2}],
            },
        )
        assert resp.status_code == 410
        assert "已移除" in resp.json()["detail"]

    def test_add_chunks_empty_list_422(self, authed_client):
        resp = authed_client.post("/api/v1/rag/add-chunks", json={"chunks": []})
        assert resp.status_code == 422


class TestRagTaskRetry:
    def test_retry_failed_task_returns_pending(self, authed_client, fake_store):
        resp = authed_client.post("/api/v1/rag/tasks/77/retry")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "pending"
        assert fake_store.retried_tasks == [77]


# ------------------------------------------------------------------
# GET /sources
# ------------------------------------------------------------------


class TestRagSources:
    def test_sources_success(self, authed_client, fake_store):
        resp = authed_client.get("/api/v1/rag/sources")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 200
        assert body["total"] == 2
        assert len(body["data"]) == 2
        assert body["data"][0] == {"source": "data/a.pdf", "count": 10}

    def test_sources_error_500(self, authed_client, monkeypatch):
        class _ErrStore(FakeVectorStore):
            def get_all_sources(self):
                raise RuntimeError("查询失败")

        monkeypatch.setattr(rag_router, "_get_vector_store", lambda: _ErrStore())
        resp = authed_client.get("/api/v1/rag/sources")
        assert resp.status_code == 500
        assert "获取文献列表失败" in resp.json()["detail"]


# ------------------------------------------------------------------
# POST /delete-by-source
# ------------------------------------------------------------------


class TestRagDeleteBySource:
    def test_delete_by_source_accepted(self, authed_client, fake_store):
        """pg 模式：返回 accepted + task_id，不同步物理删除（T-ENT-02）。"""
        resp = authed_client.post(
            "/api/v1/rag/delete-by-source", json={"source": "data/a.pdf"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 200
        # INV-008：异步接口不误报完成 —— "已接收"而非"已删除"
        assert "已接收" in body["message"]
        assert "已删除" not in body["message"]
        data = body["data"]
        assert data["source"] == "data/a.pdf"
        assert data["status"] == "accepted"
        assert isinstance(data["task_id"], int)
        # 创建了删除任务，但未同步物理删除
        assert fake_store.enqueued_deletes == ["data/a.pdf"]

    def test_delete_by_source_empty_string_422(self, authed_client):
        resp = authed_client.post("/api/v1/rag/delete-by-source", json={"source": ""})
        assert resp.status_code == 422

    def test_delete_by_source_missing_field_422(self, authed_client):
        resp = authed_client.post("/api/v1/rag/delete-by-source", json={})
        assert resp.status_code == 422


# ------------------------------------------------------------------
# POST /delete-sources
# ------------------------------------------------------------------


class TestRagDeleteSources:
    def test_delete_sources_all_accepted(self, authed_client, fake_store):
        """pg 模式：每个 source 独立任务，返回明细 + task_id（T-ENT-03）。"""
        resp = authed_client.post(
            "/api/v1/rag/delete-sources",
            json={"sources": ["data/a.pdf", "data/b.pdf"]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 200
        assert "已接收" in body["message"]
        assert "已删除" not in body["message"]
        data = body["data"]
        assert data["accepted"] == 2
        assert data["deduplicated"] == 0
        assert data["failed"] == []
        results = data["results"]
        assert len(results) == 2
        assert {r["source"] for r in results} == {"data/a.pdf", "data/b.pdf"}
        # 每个 source 独立的 task_id
        task_ids = [r["task_id"] for r in results]
        assert all(isinstance(tid, int) for tid in task_ids)
        assert len(set(task_ids)) == 2
        # 不同步物理删除

    def test_delete_sources_partial_failure(self, authed_client, fake_store):
        """pg 模式：单个 source 创建任务失败不影响其他 source。"""
        fake_store.enqueue_fail_sources = {"data/bad.pdf"}
        resp = authed_client.post(
            "/api/v1/rag/delete-sources",
            json={"sources": ["data/good.pdf", "data/bad.pdf"]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 200
        data = body["data"]
        assert data["accepted"] == 1
        assert len(data["failed"]) == 1
        assert data["failed"][0]["source"] == "data/bad.pdf"
        assert data["failed"][0]["status"] == "failed"
        assert "1 个创建失败" in body["message"]

    def test_delete_sources_empty_list_422(self, authed_client):
        resp = authed_client.post("/api/v1/rag/delete-sources", json={"sources": []})
        assert resp.status_code == 422

    def test_delete_sources_missing_field_422(self, authed_client):
        resp = authed_client.post("/api/v1/rag/delete-sources", json={})
        assert resp.status_code == 422


# ------------------------------------------------------------------
# POST /upload（文件上传入库）
# ------------------------------------------------------------------


@pytest.fixture()
def upload_dir(tmp_path, monkeypatch):
    """把 document_directory 指到临时目录，allow_type 限定 .pdf。"""
    monkeypatch.setattr(
        config_handler,
        "vector_store_config",
        {
            "document_directory": str(tmp_path),
            "allow_type": [".pdf"],
        },
    )
    return tmp_path


class TestRagUpload:
    def test_upload_unauthenticated_401(self, client):
        resp = client.post(
            "/api/v1/rag/upload",
            files={"files": ("a.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
        assert resp.status_code == 401

    def test_upload_pdf_accepted(self, authed_client, fake_store, upload_dir):
        resp = authed_client.post(
            "/api/v1/rag/upload",
            files={"files": ("paper.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 200
        data = body["data"]
        assert data["accepted"] == 1
        f = data["files"][0]
        assert f["filename"] == "paper.pdf"
        assert f["status"] == "accepted"
        assert f["task_id"] is not None
        # 文件落盘到 document_directory 且入队路径一致
        saved = upload_dir / "paper.pdf"
        assert saved.read_bytes() == b"%PDF-1.4 fake"
        assert fake_store.enqueued_files == [str(saved)]

    def test_upload_rejects_bad_extension(self, authed_client, fake_store, upload_dir):
        resp = authed_client.post(
            "/api/v1/rag/upload",
            files={"files": ("notes.txt", b"hello", "text/plain")},
        )
        assert resp.status_code == 200
        f = resp.json()["data"]["files"][0]
        assert f["status"] == "rejected"
        assert "不支持的文件类型" in f["reason"]
        assert fake_store.enqueued_files == []
        assert not (upload_dir / "notes.txt").exists()

    def test_upload_rejects_empty_file(self, authed_client, fake_store, upload_dir):
        resp = authed_client.post(
            "/api/v1/rag/upload",
            files={"files": ("empty.pdf", b"", "application/pdf")},
        )
        assert resp.status_code == 200
        f = resp.json()["data"]["files"][0]
        assert f["status"] == "rejected"
        assert f["reason"] == "空文件"
        assert fake_store.enqueued_files == []

    def test_upload_rejects_fake_pdf(self, authed_client, fake_store, upload_dir):
        resp = authed_client.post(
            "/api/v1/rag/upload",
            files={"files": ("fake.pdf", b"not actually a pdf", "application/pdf")},
        )
        assert resp.status_code == 200
        f = resp.json()["data"]["files"][0]
        assert f["status"] == "rejected"
        assert f["reason"] == "文件内容不是有效的 PDF"
        assert fake_store.enqueued_files == []
        assert not (upload_dir / "fake.pdf").exists()

    def test_upload_sanitizes_filename(self, authed_client, fake_store, upload_dir):
        """路径穿越成分被剥离，文件只落在 document_directory 内。"""
        resp = authed_client.post(
            "/api/v1/rag/upload",
            files={"files": ("../../evil.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
        assert resp.status_code == 200
        f = resp.json()["data"]["files"][0]
        assert f["status"] == "accepted"
        assert f["filename"] == "evil.pdf"
        assert (upload_dir / "evil.pdf").exists()
        assert fake_store.enqueued_files == [str(upload_dir / "evil.pdf")]

    def test_upload_deduplicated(self, authed_client, fake_store, upload_dir, monkeypatch):
        """入队返回 None 时（内容未变化）状态为 deduplicated。"""
        monkeypatch.setattr(
            fake_store.__class__, "enqueue_ingest_file", lambda self, p: None
        )
        resp = authed_client.post(
            "/api/v1/rag/upload",
            files={"files": ("same.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["accepted"] == 0
        assert data["files"][0]["status"] == "deduplicated"

    def test_upload_restores_previous_file_when_enqueue_fails(
        self, authed_client, fake_store, upload_dir, monkeypatch
    ):
        previous = upload_dir / "paper.pdf"
        previous.write_bytes(b"%PDF-1.4 previous")

        def _raise(_self, _path):
            raise RuntimeError("queue unavailable")

        monkeypatch.setattr(fake_store.__class__, "enqueue_ingest_file", _raise)
        resp = authed_client.post(
            "/api/v1/rag/upload",
            files={"files": ("paper.pdf", b"%PDF-1.4 replacement", "application/pdf")},
        )

        assert resp.status_code == 200
        assert resp.json()["data"]["files"][0]["status"] == "failed"
        assert previous.read_bytes() == b"%PDF-1.4 previous"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
