"""版本化离线索引集成测试（OFF-002 / OFF-003 / OFF-004 / OFF-007 / ONL-009）。

全部使用 fake/mock：不连接真实 PG，不调用真实 Embedding API。
Worker 流程使用 tmp_path 下的真实 sqlite（registry / tracker / queue）。
"""

import hashlib
import threading
import uuid
from unittest import mock

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from agent_material.rag_engine.db.file_tracker import FileTracker
from agent_material.rag_engine.db.task_queue import TaskQueue, TaskType, Task
from agent_material.rag_engine.db.version_registry import DocumentVersionRegistry
from agent_material.rag_engine.db.worker import BackgroundWorker
from agent_material.rag_engine.indexing.multi_granularity import MultiGranularityIndex
from agent_material.rag_engine.indexing.multi_gran_retriever import (
    MultiGranularityRetriever,
)

# ── 公共 fake ─────────────────────────────────────────────


class _FakeStore:
    """记录 add_embeddings 调用的假 PGVector。"""

    def __init__(self):
        self.calls: list[dict] = []

    def add_embeddings(self, texts, embeddings, metadatas, ids=None):
        self.calls.append(
            {"texts": texts, "embeddings": embeddings, "metadatas": metadatas, "ids": ids}
        )


class _FakeEmbeddings:
    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class _FakeVectorRetriever(BaseRetriever):
    model_config = ConfigDict(extra="allow")

    def _get_relevant_documents(self, query, *, run_manager=None):
        return []


def _make_index():
    """构造不触 PG 的 MultiGranularityIndex（fake store + fake embedding）。"""
    index = MultiGranularityIndex(enabled=True)
    index._small_store = _FakeStore()
    index._large_store = _FakeStore()
    return index


def _versioned_chunks():
    """符合 document_pipeline version_context 契约的子块。"""
    return [
        Document(
            page_content=f"子块内容 {i}",
            metadata={
                "chunk_id": f"chunk-{i}",
                "parent_id": "parent-0" if i < 2 else "parent-1",
                "parent_doc": "父块内容 0" if i < 2 else "父块内容 1",
                "parent_metadata": {
                    "heading_path": ["第 1 章"],
                    "document_id": "doc-1",
                    "version_id": "ver-1",
                    "is_active": False,
                },
                "document_id": "doc-1",
                "version_id": "ver-1",
                "is_active": False,
                "indexed_at": "2026-01-01T00:00:00",
            },
        )
        for i in range(3)
    ]


# ── 任务 1：insert(versioned=True) ────────────────────────


class TestVersionedInsert:
    def test_deterministic_uuid5_ids(self, monkeypatch):
        index = _make_index()
        monkeypatch.setattr(
            "agent_material.rag_engine.indexing.multi_granularity.embedding_model",
            _FakeEmbeddings(),
        )
        chunks = _versioned_chunks()
        index.insert(chunks, source="datas/a.pdf", versioned=True)

        parent_ids = index._large_store.calls[0]["ids"]
        child_ids = index._small_store.calls[0]["ids"]
        assert parent_ids == [
            str(uuid.uuid5(uuid.NAMESPACE_URL, "agent-material:parent:parent-0")),
            str(uuid.uuid5(uuid.NAMESPACE_URL, "agent-material:parent:parent-1")),
        ]
        assert child_ids == [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"agent-material:chunk:chunk-{i}"))
            for i in range(3)
        ]

        # 重跑相同输入 → 相同 ID（幂等）
        index2 = _make_index()
        monkeypatch.setattr(
            "agent_material.rag_engine.indexing.multi_granularity.embedding_model",
            _FakeEmbeddings(),
        )
        index2.insert(_versioned_chunks(), source="datas/a.pdf", versioned=True)
        assert index2._large_store.calls[0]["ids"] == parent_ids
        assert index2._small_store.calls[0]["ids"] == child_ids

    def test_metadata_contract(self, monkeypatch):
        index = _make_index()
        monkeypatch.setattr(
            "agent_material.rag_engine.indexing.multi_granularity.embedding_model",
            _FakeEmbeddings(),
        )
        index.insert(_versioned_chunks(), source="datas/a.pdf", versioned=True)

        # 父块：parent_metadata 提升 + chunk_type/source/parent_id
        pmeta = index._large_store.calls[0]["metadatas"][0]
        assert pmeta["chunk_type"] == "parent"
        assert pmeta["source"] == "datas/a.pdf"
        assert pmeta["parent_id"] == "parent-0"
        assert pmeta["heading_path"] == ["第 1 章"]
        assert pmeta["document_id"] == "doc-1"
        assert pmeta["version_id"] == "ver-1"
        assert pmeta["is_active"] is False

        # 子块：剥离 parent_doc/parent_metadata，补 chunk_type/source
        cmeta = index._small_store.calls[0]["metadatas"][0]
        assert cmeta["chunk_type"] == "child"
        assert cmeta["source"] == "datas/a.pdf"
        assert cmeta["chunk_id"] == "chunk-0"
        assert cmeta["parent_id"] == "parent-0"
        assert "parent_doc" not in cmeta
        assert "parent_metadata" not in cmeta

    def test_skip_chunks_without_ids(self, monkeypatch):
        index = _make_index()
        monkeypatch.setattr(
            "agent_material.rag_engine.indexing.multi_granularity.embedding_model",
            _FakeEmbeddings(),
        )
        chunks = _versioned_chunks()
        bad = Document(page_content="无 ID 子块", metadata={"parent_doc": "x"})
        parent_ids = index.insert(chunks + [bad], source="s", versioned=True)
        # 坏块被跳过：只写入 3 个子块
        assert len(index._small_store.calls[0]["ids"]) == 3
        assert parent_ids == ["parent-0", "parent-1"]

    def test_all_chunks_invalid_returns_empty(self, monkeypatch):
        index = _make_index()
        monkeypatch.setattr(
            "agent_material.rag_engine.indexing.multi_granularity.embedding_model",
            _FakeEmbeddings(),
        )
        bad = [Document(page_content="x", metadata={})]
        assert index.insert(bad, source="s", versioned=True) == []
        assert index._small_store.calls == []
        assert index._large_store.calls == []

    def test_legacy_signature_unchanged(self, monkeypatch):
        """versioned=False 保持旧行为：不传 ids，父块用 base_meta。"""
        index = _make_index()
        monkeypatch.setattr(
            "agent_material.rag_engine.indexing.multi_granularity.embedding_model",
            _FakeEmbeddings(),
        )
        index.insert(_versioned_chunks(), source="datas/a.pdf")
        assert index._large_store.calls[0]["ids"] is None
        assert index._small_store.calls[0]["ids"] is None


# ── 任务 1：validate_version ──────────────────────────────


class _FakeCursor:
    def __init__(self, results):
        self._results = list(results)

    def execute(self, sql, params=None):
        pass

    def fetchone(self):
        return self._results.pop(0)

    def fetchall(self):
        return self._results.pop(0)

    def close(self):
        pass


class _FakeConn:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def close(self):
        pass


def _patch_pg(monkeypatch, fetchone_results):
    cursor = _FakeCursor(fetchone_results)
    monkeypatch.setattr("psycopg.connect", lambda *a, **kw: _FakeConn(cursor))
    return cursor


class TestValidateVersion:
    """fetchone 结果序列：父块(数,空文本) → 子块(数,缺pid,空文本) → dup_chunk → parent(总,缺失,重复)。"""

    def test_happy_path(self, monkeypatch):
        _patch_pg(monkeypatch, [(2, 0), (6, 0, 0), (0,), (2, 0, 0)])
        ok, stats = _make_index().validate_version("doc-1", "ver-1")
        assert ok is True
        assert stats["parent_count"] == 2
        assert stats["child_count"] == 6

    def test_zero_parents(self, monkeypatch):
        _patch_pg(monkeypatch, [(0, 0), (6, 0, 0), (0,), (0, 0, 0)])
        ok, stats = _make_index().validate_version("d", "v")
        assert ok is False
        assert stats["parent_count"] == 0

    def test_zero_children(self, monkeypatch):
        _patch_pg(monkeypatch, [(2, 0), (0, 0, 0), (0,), (0, 0, 0)])
        ok, _ = _make_index().validate_version("d", "v")
        assert ok is False

    def test_missing_parent_id(self, monkeypatch):
        _patch_pg(monkeypatch, [(2, 0), (6, 1, 0), (0,), (2, 0, 0)])
        ok, stats = _make_index().validate_version("d", "v")
        assert ok is False
        assert stats["missing_parent_id"] == 1

    def test_missing_parent_in_large(self, monkeypatch):
        _patch_pg(monkeypatch, [(2, 0), (6, 0, 0), (0,), (3, 1, 0)])
        ok, stats = _make_index().validate_version("d", "v")
        assert ok is False
        assert stats["missing_parent"] == 1

    def test_duplicate_parent_in_large(self, monkeypatch):
        _patch_pg(monkeypatch, [(2, 0), (6, 0, 0), (0,), (3, 0, 1)])
        ok, stats = _make_index().validate_version("d", "v")
        assert ok is False
        assert stats["duplicate_parent"] == 1

    def test_empty_text(self, monkeypatch):
        _patch_pg(monkeypatch, [(2, 1), (6, 0, 0), (0,), (2, 0, 0)])
        ok, stats = _make_index().validate_version("d", "v")
        assert ok is False
        assert stats["empty_text"] == 1

    def test_duplicate_chunk_id(self, monkeypatch):
        _patch_pg(monkeypatch, [(2, 0), (6, 0, 0), (2,), (2, 0, 0)])
        ok, stats = _make_index().validate_version("d", "v")
        assert ok is False
        assert stats["duplicate_chunk_id"] == 2


# ── 任务 2/3：Worker 版本化流程 ───────────────────────────


class _FakePipeline:
    def __init__(self, chunks):
        self._chunks = chunks
        self.calls = 0

    def process(self, file_path, version_context=None, **kwargs):
        self.calls += 1
        self.last_version_context = version_context
        return self._chunks


class _FakeMultiIndex:
    base_name = "agent"

    def __init__(self):
        self.insert_calls = 0
        self.delete_calls: list[tuple] = []
        self.set_active_calls: list[tuple] = []
        self.fail_store = False
        self.fail_set_active = False
        self.on_set_active = None  # 可选钩子（断言调用时 tracker 状态）

    def delete_by_version(self, document_id, version_id):
        self.delete_calls.append((document_id, version_id))

    def insert(self, chunks, source="", versioned=False):
        if self.fail_store:
            raise RuntimeError("store boom")
        self.insert_calls += 1
        assert versioned is True
        return ["pid-1"]

    def validate_version(self, document_id, version_id):
        return True, {"parent_count": 1, "child_count": 2}

    def set_active_version(self, document_id, new_version_id, old_version_id):
        if self.on_set_active:
            self.on_set_active()
        self.set_active_calls.append((document_id, new_version_id, old_version_id))
        if self.fail_set_active:
            raise RuntimeError("pg boom")


def _make_task(file_path, **extra_payload):
    return Task(
        id=1,
        task_type=TaskType.INGEST_FILE.value,
        payload={"file_path": str(file_path), **extra_payload},
        status="processing",
        retries=0,
        error_msg=None,
        created_at="",
    )


@pytest.fixture()
def worker_env(tmp_path):
    db_path = str(tmp_path / "q.db")
    queue = TaskQueue(db_path=db_path)
    tracker = FileTracker(db_path=db_path)
    registry = DocumentVersionRegistry(db_path=db_path)

    doc_file = tmp_path / "a.pdf"
    doc_file.write_bytes(b"hello v1")

    fake_index = _FakeMultiIndex()
    pipeline = _FakePipeline(_versioned_chunks())
    activated_calls = []

    worker = BackgroundWorker(
        task_queue=queue,
        file_tracker=tracker,
        on_activated=lambda: activated_calls.append(1),
        version_registry=registry,
    )
    worker._pipeline = pipeline
    worker._multi_index = fake_index

    return {
        "tmp_path": tmp_path,
        "file": doc_file,
        "queue": queue,
        "tracker": tracker,
        "registry": registry,
        "worker": worker,
        "pipeline": pipeline,
        "index": fake_index,
        "activated_calls": activated_calls,
    }


def _doc_ver(env):
    from agent_material.rag_engine import versioning

    data = env["file"].read_bytes()
    content_hash = versioning.compute_content_hash(data)
    document_id = versioning.compute_document_id(str(env["file"]), kb_id="agent")
    return document_id, versioning.compute_version_id(content_hash)


class TestWorkerVersionedIngest:
    def test_happy_path(self, worker_env):
        env = worker_env
        document_id, version_id = _doc_ver(env)

        # set_active 调用时 tracker 必须尚未记录（OFF-002：激活后才更新 FileTracker）
        file_path = str(env["file"])

        def _assert_tracker_not_recorded():
            state = env["tracker"].get_by_path(file_path)
            assert state is None or state["last_indexed_at"] is None

        env["index"].on_set_active = _assert_tracker_not_recorded

        env["worker"]._handle_ingest(_make_task(file_path))

        row = env["registry"].get(document_id, version_id)
        assert row["status"] == "active"
        assert row["parent_count"] == 1
        assert row["child_count"] == 2
        assert env["activated_calls"] == [1]

        # pipeline 收到 version_context
        assert env["pipeline"].last_version_context == {
            "document_id": document_id,
            "version_id": version_id,
        }

        # tracker 在激活后落库（md5/mtime/size + parent_ids）
        state = env["tracker"].get_by_path(file_path)
        assert state is not None
        assert state["last_indexed_at"] is not None
        assert state["md5_hash"] == hashlib.md5(b"hello v1").hexdigest()
        assert state["parent_ids"] == ["pid-1"]

        # store 阶段先清 staging 残留再写入
        assert env["index"].delete_calls == [(document_id, version_id)]

    def test_idempotent_skip_same_content(self, worker_env):
        env = worker_env
        file_path = str(env["file"])
        env["worker"]._handle_ingest(_make_task(file_path))
        env["worker"]._handle_ingest(_make_task(file_path))

        # 第二次：相同内容已激活 → 幂等跳过，pipeline 不再调用
        assert env["pipeline"].calls == 1
        assert env["index"].insert_calls == 1

    def test_store_failure_marks_failed_and_recovers(self, worker_env):
        env = worker_env
        file_path = str(env["file"])
        document_id, version_id = _doc_ver(env)

        env["index"].fail_store = True
        with pytest.raises(RuntimeError, match="store boom"):
            env["worker"]._handle_ingest(_make_task(file_path))

        row = env["registry"].get(document_id, version_id)
        assert row["status"] == "failed"
        assert row["failure_stage"] == "store"
        assert "store boom" in row["failure_reason"]
        # 失败时不得更新 tracker
        state = env["tracker"].get_by_path(file_path)
        assert state is None or state["last_indexed_at"] is None

        # 重新投递 → 从头安全重试并恢复
        env["index"].fail_store = False
        env["worker"]._handle_ingest(_make_task(file_path))
        assert env["registry"].get(document_id, version_id)["status"] == "active"
        assert env["index"].insert_calls == 1

    def test_activate_idempotent_recovery(self, worker_env):
        env = worker_env
        file_path = str(env["file"])
        document_id, version_id = _doc_ver(env)

        # 模拟崩溃：registry 已激活、PG 翻牌失败
        activate_spy = mock.patch.object(
            env["registry"], "activate", wraps=env["registry"].activate
        )
        env["index"].fail_set_active = True
        with activate_spy as spy:
            with pytest.raises(RuntimeError, match="pg boom"):
                env["worker"]._handle_ingest(_make_task(file_path))
            assert spy.call_count == 1

            # registry 已 active（旧版本继续在线服务的另一侧：新版本 registry 已翻牌）
            assert env["registry"].get_active(document_id)["version_id"] == version_id

            # 重试：不再调 registry.activate，直接补 PG 翻牌（old=None）
            env["index"].fail_set_active = False
            env["worker"]._handle_ingest(_make_task(file_path))
            assert spy.call_count == 1
            assert env["index"].set_active_calls[-1] == (document_id, version_id, None)

    def test_bm25_refresh_failure_does_not_fail_task(self, worker_env):
        env = worker_env

        def _boom():
            raise RuntimeError("bm25 boom")

        env["worker"]._on_activated = _boom
        # 不应抛出：BM25 刷新失败只记 error，任务仍完成
        env["worker"]._handle_ingest(_make_task(str(env["file"])))
        document_id, version_id = _doc_ver(env)
        assert env["registry"].get(document_id, version_id)["status"] == "active"

    def test_empty_chunks_not_failed(self, worker_env):
        env = worker_env
        env["pipeline"]._chunks = []
        env["worker"]._handle_ingest(_make_task(str(env["file"])))
        document_id, version_id = _doc_ver(env)
        # 空结果：不标 failed（保持 staging），不写 store
        row = env["registry"].get(document_id, version_id)
        assert row["status"] == "staging"
        assert env["index"].insert_calls == 0


# ── 任务 3：FileTracker 延迟落库 ──────────────────────────


class TestFileTrackerDefer:
    def test_scan_defer_does_not_upsert(self, tmp_path):
        tracker = FileTracker(db_path=str(tmp_path / "t.db"))
        docs = tmp_path / "docs"
        docs.mkdir()
        f = docs / "a.md"
        f.write_text("内容", encoding="utf-8")

        result = tracker.scan_directory(str(docs), [".md"], defer_state_update=True)
        assert result["new"] == [str(f)]
        # 延迟落库：file_states 不写，但 file_info 携带 md5/mtime/size
        assert tracker.get_by_path(str(f)) is None
        info = result["file_info"][str(f)]
        assert info["md5_hash"] == hashlib.md5("内容".encode()).hexdigest()
        assert info["file_size"] == f.stat().st_size

        # 再次扫描仍是 new（状态未落库）
        result2 = tracker.scan_directory(str(docs), [".md"], defer_state_update=True)
        assert result2["new"] == [str(f)]

    def test_scan_default_upserts(self, tmp_path):
        tracker = FileTracker(db_path=str(tmp_path / "t.db"))
        docs = tmp_path / "docs"
        docs.mkdir()
        f = docs / "a.md"
        f.write_text("内容", encoding="utf-8")

        result = tracker.scan_directory(str(docs), [".md"])
        assert result["new"] == [str(f)]
        assert tracker.get_by_path(str(f)) is not None

    def test_record_indexed_upserts_state(self, tmp_path):
        tracker = FileTracker(db_path=str(tmp_path / "t.db"))
        f = tmp_path / "a.md"
        f.write_text("内容", encoding="utf-8")
        tracker.record_indexed(
            str(f),
            ["pid-1"],
            md5_hash="abc",
            file_mtime=1.0,
            file_size=100,
        )
        state = tracker.get_by_path(str(f))
        assert state["md5_hash"] == "abc"
        assert state["file_size"] == 100
        assert state["last_indexed_at"] is not None
        assert state["parent_ids"] == ["pid-1"]

        # parent_ids=None 时保留已存 parent_ids
        tracker.record_indexed(str(f), None, md5_hash="abc", file_mtime=2.0, file_size=100)
        assert tracker.get_by_path(str(f))["parent_ids"] == ["pid-1"]


# ── 任务 4：refresh_documents 原子替换 ────────────────────


def _make_retriever(docs):
    retriever = MultiGranularityRetriever(
        small_store=object(), large_store=object(), all_documents=docs
    )
    retriever._vector_retriever = _FakeVectorRetriever()
    retriever._retrievers_initialized = True
    return retriever


class TestRefreshDocuments:
    def test_new_docs_searchable_after_refresh(self):
        retriever = _make_retriever([Document(page_content="旧文档 alpha", metadata={})])
        retriever.refresh_documents(
            [Document(page_content="zirconia-xyzzy 新材料", metadata={})]
        )
        assert retriever._bm25_retriever is not None
        results = retriever._bm25_retriever.invoke("zirconia-xyzzy")
        assert len(results) == 1
        assert "zirconia-xyzzy" in results[0].page_content
        # 旧文档已不在新索引中（BM25 单文档时任意查询都返回该文档，
        # 用旧正文缺席来证明旧实例已被替换）
        assert all("旧文档" not in d.page_content for d in results)

    def test_build_failure_keeps_old_instance(self, monkeypatch):
        old_docs = [Document(page_content="旧文档 alpha", metadata={})]
        retriever = _make_retriever(old_docs)
        retriever.refresh_documents(old_docs)
        old_bm25 = retriever._bm25_retriever
        old_hybrid = retriever._hybrid_retriever

        import agent_material.rag_engine.indexing.multi_gran_retriever as mgr

        monkeypatch.setattr(
            mgr.BM25Retriever,
            "from_documents",
            classmethod(lambda cls, docs, **kw: (_ for _ in ()).throw(RuntimeError("boom"))),
        )
        retriever.refresh_documents([Document(page_content="新文档", metadata={})])
        # 构建失败：保留旧实例
        assert retriever._bm25_retriever is old_bm25
        assert retriever._hybrid_retriever is old_hybrid

    def test_empty_docs_degrades_to_vector_only(self):
        retriever = _make_retriever([Document(page_content="旧文档 alpha", metadata={})])
        retriever.refresh_documents([])
        assert retriever._bm25_retriever is None
        assert retriever._hybrid_retriever is retriever._vector_retriever

    def test_concurrent_read_during_refresh(self):
        retriever = _make_retriever([Document(page_content="初始 alpha", metadata={})])
        retriever.refresh_documents([Document(page_content="初始 alpha", metadata={})])

        errors: list[Exception] = []
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                try:
                    retriever._hybrid_retriever.invoke("alpha")
                    retriever._bm25_retriever.invoke("alpha")
                except Exception as e:  # noqa: BLE001
                    errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for i in range(20):
            retriever.refresh_documents(
                [Document(page_content=f"刷新文档 beta-{i}", metadata={})]
            )
        stop.set()
        for t in threads:
            t.join()
        assert errors == []


# ── 任务 4：_fetch_parents 有序回溯 + missing 回退 ────────


class TestFetchParents:
    def _patch_pg_rows(self, monkeypatch, rows):
        cursor = _FakeCursor([rows])
        monkeypatch.setattr("psycopg.connect", lambda *a, **kw: _FakeConn(cursor))

    def test_order_restored(self, monkeypatch):
        # DB 返回顺序与输入不同 → 结果必须按 parent_ids 输入顺序重排
        self._patch_pg_rows(
            monkeypatch,
            [("父块1正文", {"parent_id": "p1"}), ("父块2正文", {"parent_id": "p2"})],
        )
        retriever = _make_retriever([])
        docs, missing = retriever._fetch_parents(["p2", "p1"])
        assert missing == []
        assert [d.metadata["parent_id"] for d in docs] == ["p2", "p1"]
        assert docs[0].page_content == "父块2正文"

    def test_missing_ids_reported(self, monkeypatch):
        self._patch_pg_rows(monkeypatch, [("父块2正文", {"parent_id": "p2"})])
        retriever = _make_retriever([])
        docs, missing = retriever._fetch_parents(["p2", "p3"])
        assert [d.metadata["parent_id"] for d in docs] == ["p2"]
        assert missing == ["p3"]

    def test_sql_error_returns_all_missing(self, monkeypatch):
        monkeypatch.setattr(
            "psycopg.connect",
            mock.Mock(side_effect=RuntimeError("conn boom")),
        )
        retriever = _make_retriever([])
        docs, missing = retriever._fetch_parents(["p1", "p2"])
        assert docs == []
        assert missing == ["p1", "p2"]

    def test_missing_parent_falls_back_to_child(self):
        retriever = _make_retriever([])
        child1 = Document(page_content="子块1", metadata={"parent_id": "p1"})
        child2 = Document(page_content="子块2", metadata={"parent_id": "p2"})
        parent1 = Document(page_content="父块1", metadata={"parent_id": "p1"})

        retriever._hybrid_search = lambda q: [child1, child2]
        retriever._fetch_parents = lambda ids: ([parent1], ["p2"])

        docs = retriever._get_relevant_documents("query")
        assert len(docs) == 2
        # 命中的父块保持顺序第一
        assert docs[0].page_content == "父块1"
        assert "parent_fallback" not in docs[0].metadata
        # 未命中的 p2 回退子块并打标
        assert docs[1].page_content == "子块2"
        assert docs[1].metadata["parent_fallback"] is True
