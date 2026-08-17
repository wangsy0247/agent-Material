"""BM25 生命周期与在线链路收口测试（Spec V3 11 / 12.3 / 15）。

覆盖：
- BM25-001 唯一构建状态机：冷启动=热刷新同一 rebuild 流程（T-BM25-01），
  先捕获 revision 再加载（T-BM25-02）
- BM25-002/INV-006 严格 active-only：缺 is_active 不进语料（T-BM25-05）
- BM25-003 加载失败显式化：SQL/连接/collection/契约失败抛 BM25LoadError，
  旧实例保留、revision 不推进（T-BM25-03）；合法空知识库成功构建并记
  revision（T-BM25-04）
- BM25-004 多进程语义：两个实例各自按 local revision 刷新（T-BM25-09）
- BM25-005 RRF 前 active 校验：缺 chunk_id 候选丢弃（T-BM25-06）、
  退役候选剔除、校验 SQL 失败降级 vector-only（T-BM25-07）
- 9.3 刷新失败：保留旧 BM25 实例 + 旧语料退役内容被二次校验过滤（T-BM25-08）
- 12.3 SSE 取消：_consume_stream 因取消标志退出、底层 stream 被关闭、cancelled 帧保留
- 15 可运维性：/rag/kb-state、/rag/consistency-check 端点与一致性检测 SQL

真实 PG（127.0.0.1:5433，连接串读 config/vector_store_config.yaml），
table_prefix="test_rag_" 隔离测试表，测试 collection test_bm25_small/
test_bm25_large 用后清理；PG 不可达时集成用例 skip。
BM25 用 rank-bm25 真实构建；不打 LLM/embedding API（Document 列表直接喂
refresh_documents，不走向量写入）。
"""

import asyncio
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

# backend/ 与 backend/packages 加入路径（pyproject 仅配置了 packages）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 注意 import 顺序（本环境 WSL+Anaconda 存在原生库符号冲突）：
# vector_store 必须先于 psycopg 导入——在 test_auth（jose/cryptography）
# 之后若以 psycopg → vector_store 顺序导入会稳定 segfault（收集期 139），
# 反向顺序则正常。请勿调整下列两组的先后关系。
from agent_material.rag_engine.vector_store import (  # noqa: E402
    BM25LoadError,
    VectorStore,
)

import psycopg  # noqa: E402
from psycopg.types.json import Jsonb  # noqa: E402

from langchain_core.documents import Document  # noqa: E402

from agent_material.rag_engine.db.pg import (  # noqa: E402
    KbStateRepository,
    drop_schema,
    ensure_schema,
)
from agent_material.rag_engine.db.pg.schema import normalize_conn_str  # noqa: E402
from agent_material.rag_engine.request_context import RequestContext  # noqa: E402
from agent_material.rag_engine.retrieval_pipeline import RetrievalPipeline  # noqa: E402
from agent_material.rag_engine.indexing import multi_gran_retriever as mgr_module  # noqa: E402
from agent_material.rag_engine.indexing.multi_gran_retriever import (  # noqa: E402
    MultiGranularityRetriever,
)
from agent_material.utils.config_handler import (  # noqa: E402
    get_vector_store_config,
    vector_store_config,
)

ROOT = Path(__file__).resolve().parents[2]
CONN_STR = normalize_conn_str(get_vector_store_config()["connection_string"])
PREFIX = "test_rag_"
SMALL = "test_bm25_small"
LARGE = "test_bm25_large"


# ------------------------------------------------------------------
# PG fixtures 与工具
# ------------------------------------------------------------------


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
                WHERE name LIKE 'test\_bm25\_%'
            )
            """
        )
        conn.execute(
            r"DELETE FROM langchain_pg_collection WHERE name LIKE 'test\_bm25\_%'"
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


def _insert_chunk(conn_str, collection, content, metadata):
    """直接 SQL 插入一条子块/父块（不走向量写入，embedding 列留空）。"""
    with psycopg.connect(conn_str) as conn:
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
            (str(uuid.uuid4()), collection, content, Jsonb(metadata)),
        )
        conn.commit()


@pytest.fixture()
def kb_id(monkeypatch) -> str:
    """pg 控制面 + 每个用例独立的 kb_id（经 character_name 注入），
    避免共享 test_rag_rag_kb_state 行导致 revision 跨用例累积。"""
    kid = f"test_kb_{uuid.uuid4().hex[:8]}"
    monkeypatch.setitem(vector_store_config, "character_name", kid)
    return kid


def _make_store(monkeypatch, kb_repo, interval: float = 0.0) -> VectorStore:
    store = VectorStore()
    monkeypatch.setattr(store, "_get_pg_repos", lambda: {"kb": kb_repo})
    monkeypatch.setitem(vector_store_config, "bm25_revision_check_interval", interval)
    return store


def _make_retriever(monkeypatch, docs=None) -> MultiGranularityRetriever:
    """构造真实 BM25 的多粒度检索器（关闭 reranker，不打任何 API）。"""
    monkeypatch.setitem(vector_store_config, "enable_reranker", False)
    retr = MultiGranularityRetriever(
        small_store=None, large_store=None, all_documents=docs or []
    )
    retr._init_retrievers()  # 真实构建 BM25（docs 为空则 bm25=None）
    return retr


# ------------------------------------------------------------------
# 9.1 冷启动
# ------------------------------------------------------------------


class TestColdStart:
    def test_cold_start_loads_active_only(self, collections):
        """冷启动无条件只加载显式 active 子块（staging/retired/缺 is_active 不进 BM25）。"""
        _insert_chunk(
            CONN_STR, SMALL, "active 子块内容",
            {
                "chunk_id": "c-active",
                "document_id": "d-1",
                "version_id": "v-1",
                "parent_id": "p-1",
                "is_active": True,
            },
        )
        _insert_chunk(
            CONN_STR, SMALL, "staging 子块内容",
            {
                "chunk_id": "c-staging",
                "document_id": "d-1",
                "version_id": "v-2",
                "parent_id": "p-2",
                "is_active": False,
            },
        )
        store = VectorStore()
        docs = store._load_documents_from_collection(
            SMALL, active_only=True, require_parent_id=True
        )
        chunk_ids = {d.metadata.get("chunk_id") for d in docs}
        assert "c-active" in chunk_ids
        assert "c-staging" not in chunk_ids

    def test_cold_start_records_revision(self, pg, collections, kb_id, monkeypatch):
        """T-BM25-01：冷启动 active_revision=N，走统一 rebuild 状态机，
        构建成功后 local_bm25_revision=N 且回写远端观测值。"""
        _insert_chunk(
            CONN_STR, SMALL, "active 子块内容",
            {
                "chunk_id": "c-1",
                "document_id": "d-1",
                "version_id": "v-1",
                "parent_id": "p-1",
                "is_active": True,
            },
        )
        kb = KbStateRepository(pg, PREFIX)
        kb.increment_active_revision(kb_id)
        kb.increment_active_revision(kb_id)

        store = _make_store(monkeypatch, kb)
        store._multi_gran_enabled = True
        monkeypatch.setattr(
            store, "_get_multi_index",
            lambda: SimpleNamespace(
                small_name=SMALL, large_name=LARGE,
                small_store=None, large_store=None,
            ),
        )

        store._init_multi_retriever()  # 冷启动 = refresh_bm25 同一状态机

        assert store._multi_retriever is not None
        assert len(store._multi_retriever.all_documents) == 1
        assert store._local_bm25_revision == 2
        assert kb.get(kb_id)["bm25_revision"] == 2

    def test_cold_start_load_failure_keeps_revision_zero(
        self, pg, collections, kb_id, monkeypatch
    ):
        """冷启动加载失败：挂空语料兜底实例（vector-only），revision 不推进，
        等待节流窗口重试（BM25-003）。"""
        kb = KbStateRepository(pg, PREFIX)
        kb.increment_active_revision(kb_id)
        store = _make_store(monkeypatch, kb)
        store._multi_gran_enabled = True
        monkeypatch.setattr(
            store, "_get_multi_index",
            lambda: SimpleNamespace(
                small_name=SMALL, large_name=LARGE,
                small_store=None, large_store=None,
            ),
        )

        def _boom(*a, **kw):
            raise BM25LoadError("模拟连接失败")

        monkeypatch.setattr(store, "_load_documents_from_collection", _boom)
        store._init_multi_retriever()

        assert store._multi_retriever is not None  # 兜底实例已挂
        assert store._multi_retriever.all_documents == []  # vector-only
        assert store._local_bm25_revision == 0  # 未推进
        assert kb.get(kb_id)["bm25_revision"] == 0


# ------------------------------------------------------------------
# 9.2 revision 检测刷新 + 节流
# ------------------------------------------------------------------


class TestRevisionRefresh:
    def test_revision_lag_triggers_refresh(self, pg, collections, kb_id, monkeypatch):
        """active_revision 领先 → 下次检索路径检测后 refresh_documents 被调、
        本地/远端 bm25_revision 推进。"""
        if not mgr_module.BM25_AVAILABLE:
            pytest.skip("rank-bm25 未安装")
        _insert_chunk(
            CONN_STR, SMALL, "氧化镓热导率研究",
            {
                "chunk_id": "c-1",
                "document_id": "d-1",
                "version_id": "v-1",
                "parent_id": "p-1",
                "is_active": True,
            },
        )
        _insert_chunk(
            CONN_STR, SMALL, "金刚石热导率研究",
            {
                "chunk_id": "c-2",
                "document_id": "d-1",
                "version_id": "v-1",
                "parent_id": "p-1",
                "is_active": True,
            },
        )
        kb = KbStateRepository(pg, PREFIX)

        store = _make_store(monkeypatch, kb, interval=0.0)
        store._multi_gran_enabled = True
        monkeypatch.setattr(
            store, "_get_multi_index",
            lambda: SimpleNamespace(small_name=SMALL, large_name=LARGE),
        )

        refresh_calls: list[int] = []
        retr = _make_retriever(monkeypatch)
        orig_refresh = retr.refresh_documents

        def spy_refresh(docs):
            refresh_calls.append(len(docs))
            return orig_refresh(docs)

        monkeypatch.setattr(retr, "refresh_documents", spy_refresh)
        store._multi_retriever = retr

        # revision 0 == local 0 → 不刷新
        store.maybe_refresh_bm25()
        assert refresh_calls == []

        # 模拟激活（active_revision +1）→ 下次检测触发刷新
        kb.increment_active_revision(kb_id)
        store.maybe_refresh_bm25()

        assert refresh_calls == [2]  # refresh_documents 被调，语料为 2 条 active
        assert store._local_bm25_revision == 1
        assert kb.get(kb_id)["bm25_revision"] == 1
        assert retr._bm25_retriever is not None  # 真实 BM25 已构建
        assert len(retr.all_documents) == 2

        # 再次检测：revision 已对齐 → 不再刷新
        store.maybe_refresh_bm25()
        assert refresh_calls == [2]

    def test_refresh_failure_keeps_revision_lagging(
        self, pg, collections, kb_id, monkeypatch
    ):
        """刷新失败不推进本地 revision（下个窗口自动重试）。"""
        kb = KbStateRepository(pg, PREFIX)
        store = _make_store(monkeypatch, kb, interval=0.0)
        store._multi_gran_enabled = True
        monkeypatch.setattr(
            store, "_get_multi_index",
            lambda: SimpleNamespace(small_name=SMALL, large_name=LARGE),
        )
        monkeypatch.setattr(store, "refresh_bm25", lambda: False)

        kb.increment_active_revision(kb_id)
        store.maybe_refresh_bm25()

        assert store._local_bm25_revision == 0  # 未推进
        assert kb.get(kb_id)["bm25_revision"] == 0

    def test_revision_check_throttled(self, kb_id, monkeypatch):
        """节流：间隔内连续两次检查只查一次 PG。"""

        class CountingKb:
            def __init__(self):
                self.get_calls = 0

            def get(self, kb_id):
                self.get_calls += 1
                return {"active_revision": 0, "bm25_revision": 0}

        kb = CountingKb()
        store = _make_store(monkeypatch, kb, interval=3600.0)

        store.maybe_refresh_bm25()
        store.maybe_refresh_bm25()
        store.maybe_refresh_bm25()
        assert kb.get_calls == 1

        # 窗口过后再次检查
        store._last_bm25_revision_check = 0.0
        store.maybe_refresh_bm25()
        assert kb.get_calls == 2


# ------------------------------------------------------------------
# T-BM25-02 捕获 revision 后发生激活：local 记捕获值，下轮追平
# ------------------------------------------------------------------


class TestCaptureThenActivate:
    def test_local_records_captured_not_latest(
        self, pg, collections, kb_id, monkeypatch
    ):
        """构建窗口内发生新激活：local 仍记捕获值（不误标新 revision），
        下一轮节流检测自动追平（BM25-001 / INV-007）。"""
        _insert_chunk(
            CONN_STR, SMALL, "active 子块",
            {
                "chunk_id": "c-1",
                "document_id": "d-1",
                "version_id": "v-1",
                "parent_id": "p-1",
                "is_active": True,
            },
        )
        kb = KbStateRepository(pg, PREFIX)
        kb.increment_active_revision(kb_id)  # active_revision = 1

        store = _make_store(monkeypatch, kb, interval=0.0)
        store._multi_gran_enabled = True
        monkeypatch.setattr(
            store, "_get_multi_index",
            lambda: SimpleNamespace(small_name=SMALL, large_name=LARGE),
        )
        store._multi_retriever = _make_retriever(monkeypatch)

        # 首次加载中途模拟一次新激活（active_revision 1 → 2），仅触发一次
        orig_load = store._load_documents_from_collection
        activated = {"done": False}

        def load_then_activate(*a, **kw):
            docs = orig_load(*a, **kw)
            if not activated["done"]:
                activated["done"] = True
                kb.increment_active_revision(kb_id)
            return docs

        monkeypatch.setattr(
            store, "_load_documents_from_collection", load_then_activate
        )

        assert store.refresh_bm25() is True
        assert store._local_bm25_revision == 1  # 记捕获值，而非加载后的最新值 2
        assert kb.get(kb_id)["bm25_revision"] == 1

        # 下一轮节流检测：active_revision=2 > local=1 → 再刷新追平
        store.maybe_refresh_bm25()
        assert store._local_bm25_revision == 2


# ------------------------------------------------------------------
# T-BM25-03/04/05 加载失败显式化与合法空知识库（BM25-002/003）
# ------------------------------------------------------------------


class TestLoadFailureSemantics:
    def test_load_failure_keeps_old_instance_and_revision(
        self, pg, collections, kb_id, monkeypatch
    ):
        """T-BM25-03：SQL 加载失败（≠构建失败）→ 返回 False，旧实例保留，
        local/远端 revision 均不推进，下个节流窗口重试。"""
        kb = KbStateRepository(pg, PREFIX)
        kb.increment_active_revision(kb_id)  # active_revision = 1
        store = _make_store(monkeypatch, kb, interval=0.0)
        store._multi_gran_enabled = True
        monkeypatch.setattr(
            store, "_get_multi_index",
            lambda: SimpleNamespace(small_name=SMALL, large_name=LARGE),
        )
        old_retr = _make_retriever(
            monkeypatch,
            docs=[Document(page_content="旧语料", metadata={"chunk_id": "c-old"})],
        )
        store._multi_retriever = old_retr

        # 开关式加载故障注入：开启时抛错，关闭后走真实加载
        orig_load = store._load_documents_from_collection
        fail = {"on": True}

        def gated_load(*a, **kw):
            if fail["on"]:
                raise BM25LoadError("模拟 SQL 执行失败")
            return orig_load(*a, **kw)

        monkeypatch.setattr(store, "_load_documents_from_collection", gated_load)
        assert store.refresh_bm25() is False

        assert store._multi_retriever is old_retr  # 旧实例保留
        assert store._local_bm25_revision == 0  # revision 不推进
        assert kb.get(kb_id)["bm25_revision"] == 0

        # 恢复加载后下个窗口自动重试成功
        fail["on"] = False
        store.maybe_refresh_bm25()
        assert store._local_bm25_revision == 1

    def test_load_nonexistent_collection_raises(self, pg):
        """BM25-003：collection 不存在是加载失败，抛 BM25LoadError 而非返回 []。"""
        store = VectorStore()
        with pytest.raises(BM25LoadError, match="collection 不存在"):
            store._load_documents_from_collection(
                "test_bm25_no_such_collection", active_only=True
            )

    def test_empty_kb_builds_empty_bm25_and_records_revision(
        self, pg, collections, kb_id, monkeypatch
    ):
        """T-BM25-04：合法空知识库（成功 SQL + 0 行）→ 成功构建空 BM25
        （vector-only 降级）并推进 revision。"""
        kb = KbStateRepository(pg, PREFIX)
        kb.increment_active_revision(kb_id)  # active_revision = 1，语料为空
        store = _make_store(monkeypatch, kb, interval=0.0)
        store._multi_gran_enabled = True
        monkeypatch.setattr(
            store, "_get_multi_index",
            lambda: SimpleNamespace(small_name=SMALL, large_name=LARGE),
        )
        retr = _make_retriever(monkeypatch)  # 空语料
        store._multi_retriever = retr

        assert store.refresh_bm25() is True
        assert store._local_bm25_revision == 1  # 空语料也推进 revision
        assert kb.get(kb_id)["bm25_revision"] == 1
        assert retr.all_documents == []
        assert retr._bm25_retriever is None  # vector-only

    def test_missing_is_active_not_loaded(self, collections):
        """T-BM25-05：metadata 缺 is_active 的行不进入 BM25 语料（INV-006，
        缺失不再视为 active）。"""
        _insert_chunk(
            CONN_STR, SMALL, "缺 is_active 的完整行",
            {
                "chunk_id": "c-no-active-flag",
                "document_id": "d-1",
                "version_id": "v-1",
                "parent_id": "p-1",
            },
        )
        _insert_chunk(
            CONN_STR, SMALL, "显式 active 行",
            {
                "chunk_id": "c-active",
                "document_id": "d-1",
                "version_id": "v-1",
                "parent_id": "p-1",
                "is_active": True,
            },
        )
        store = VectorStore()
        docs = store._load_documents_from_collection(
            SMALL, active_only=True, require_parent_id=True
        )
        chunk_ids = {d.metadata["chunk_id"] for d in docs}
        assert chunk_ids == {"c-active"}

    def test_contract_violation_raises(self, collections):
        """T-BM25-05：active 行 metadata 契约不全（缺 chunk_id）→ 加载失败
        抛 BM25LoadError（metadata 契约校验失败属加载失败，BM25-003）。"""
        _insert_chunk(
            CONN_STR, SMALL, "缺 chunk_id 的 active 行",
            {"document_id": "d-1", "version_id": "v-1", "is_active": True},
        )
        store = VectorStore()
        with pytest.raises(BM25LoadError, match="契约不全"):
            store._load_documents_from_collection(SMALL, active_only=True)

    def test_child_missing_parent_id_raises(self, collections):
        """T-BM25-05：多粒度子块缺 parent_id → metadata 契约失败抛错。"""
        _insert_chunk(
            CONN_STR, SMALL, "缺 parent_id 的 active 子块",
            {
                "chunk_id": "c-1",
                "document_id": "d-1",
                "version_id": "v-1",
                "is_active": True,
            },
        )
        store = VectorStore()
        with pytest.raises(BM25LoadError, match="parent_id"):
            store._load_documents_from_collection(
                SMALL, active_only=True, require_parent_id=True
            )


# ------------------------------------------------------------------
# T-BM25-09 多查询进程：各自按 local revision 独立刷新（BM25-004）
# ------------------------------------------------------------------


class TestMultiProcessSemantics:
    def test_two_instances_refresh_independently(
        self, pg, collections, kb_id, monkeypatch
    ):
        """两个查询进程（两个 VectorStore 实例）各自维护 local_bm25_revision：
        一个刷新不带动另一个，正确性只依赖 active_revision > local。"""
        _insert_chunk(
            CONN_STR, SMALL, "active 子块",
            {
                "chunk_id": "c-1",
                "document_id": "d-1",
                "version_id": "v-1",
                "parent_id": "p-1",
                "is_active": True,
            },
        )
        kb = KbStateRepository(pg, PREFIX)
        kb.increment_active_revision(kb_id)  # active_revision = 1

        stores = []
        for _ in range(2):
            store = _make_store(monkeypatch, kb, interval=0.0)
            store._multi_gran_enabled = True
            monkeypatch.setattr(
                store, "_get_multi_index",
                lambda: SimpleNamespace(small_name=SMALL, large_name=LARGE),
            )
            store._multi_retriever = _make_retriever(monkeypatch)
            stores.append(store)
        store_a, store_b = stores

        # A 检测刷新 → 仅 A 推进
        store_a.maybe_refresh_bm25()
        assert store_a._local_bm25_revision == 1
        assert store_b._local_bm25_revision == 0  # B 不受影响

        # B 下次检测自行追平
        store_b.maybe_refresh_bm25()
        assert store_b._local_bm25_revision == 1


# ------------------------------------------------------------------
# 9.3 / 10.2 RRF 前 active 二次校验
# ------------------------------------------------------------------


class TestActiveRecheck:
    def test_filter_drops_retired_candidates(self, collections, monkeypatch):
        """BM25 语料含已退役 chunk → 二次校验剔除（真实 PG 单条批量 SQL）。"""
        _insert_chunk(
            CONN_STR, SMALL, "现役内容",
            {
                "chunk_id": "c-active", "document_id": "d-1",
                "version_id": "v-1", "is_active": True,
            },
        )
        _insert_chunk(
            CONN_STR, SMALL, "已退役内容",
            {
                "chunk_id": "c-retired", "document_id": "d-1",
                "version_id": "v-0", "is_active": False,
            },
        )
        # filter_active_bm25 由 config 推导 small collection 名（test_bm25 + _small）
        monkeypatch.setitem(vector_store_config, "character_name", "test_bm25")
        retr = MultiGranularityRetriever(small_store=None, large_store=None)

        docs = [
            Document(
                page_content="现役内容",
                metadata={"chunk_id": "c-active", "document_id": "d-1", "version_id": "v-1"},
            ),
            Document(
                page_content="已退役内容",
                metadata={"chunk_id": "c-retired", "document_id": "d-1", "version_id": "v-0"},
            ),
            Document(page_content="无 chunk_id 旧数据", metadata={}),
        ]
        kept = retr.filter_active_bm25(docs)
        contents = [d.page_content for d in kept]
        assert "现役内容" in contents
        assert "已退役内容" not in contents
        # T-BM25-06：缺 chunk_id 的候选 RRF 前直接丢弃（BM25-005，不再兼容保留）
        assert "无 chunk_id 旧数据" not in contents

    def test_filter_all_missing_chunk_id_returns_empty(self, collections, monkeypatch):
        """T-BM25-06：全部候选缺 chunk_id → 全部丢弃，不打 SQL。"""
        monkeypatch.setitem(vector_store_config, "character_name", "test_bm25")
        retr = MultiGranularityRetriever(small_store=None, large_store=None)
        docs = [Document(page_content="无 chunk_id", metadata={})]
        assert retr.filter_active_bm25(docs) == []

    def test_filter_uses_chunk_document_version_identity(self, collections, monkeypatch):
        """相同 legacy chunk_id 不能让另一文档的 retired 候选复活。"""
        _insert_chunk(
            CONN_STR, SMALL, "A 的现役块",
            {
                "chunk_id": "legacy-shared", "document_id": "doc-a",
                "version_id": "v1", "is_active": True,
            },
        )
        monkeypatch.setitem(vector_store_config, "character_name", "test_bm25")
        retr = MultiGranularityRetriever(small_store=None, large_store=None)
        docs = [
            Document(
                page_content="A 的现役块",
                metadata={
                    "chunk_id": "legacy-shared", "document_id": "doc-a",
                    "version_id": "v1",
                },
            ),
            Document(
                page_content="B 的退役块",
                metadata={
                    "chunk_id": "legacy-shared", "document_id": "doc-b",
                    "version_id": "v0",
                },
            ),
        ]
        assert [d.page_content for d in retr.filter_active_bm25(docs)] == ["A 的现役块"]

    # ── pipeline 级（fake checker，不打 PG）──

    class _FakeRetriever:
        """search_children + 可控 filter_active_bm25 的检索器替身。"""

        def __init__(self, vector_docs, bm25_docs, check_error=None, keep_ids=None):
            self._vector_docs = vector_docs
            self._bm25_docs = bm25_docs
            self._bm25_retriever = object() if bm25_docs else None
            self._check_error = check_error
            self._keep_ids = keep_ids
            self.check_calls = 0

        def search_children(self, query, k=None, hyde_embedding=None):
            return list(self._vector_docs), list(self._bm25_docs)

        def filter_active_bm25(self, docs):
            self.check_calls += 1
            if self._check_error is not None:
                raise self._check_error
            return [
                d for d in docs
                if d.metadata.get("chunk_id") in (self._keep_ids or set())
            ]

    @staticmethod
    def _make_pipeline(retriever):
        cfg = {
            "k": 5,
            "vector_weight": 0.5,
            "bm25_weight": 0.5,
            "feature_flags": {},
        }
        return RetrievalPipeline(retriever_getter=lambda: retriever, config=cfg)

    @staticmethod
    def _make_ctx():
        ctx = RequestContext(original_query="热导率")
        ctx.generated_queries = [
            {"query": "热导率", "weight": 1.0, "type": "original"}
        ]
        return ctx

    def test_pipeline_recheck_filters_before_rrf(self):
        """二次校验在单查询 RRF 之前执行，退役候选不进分支结果。"""
        v = Document(page_content="向量路结果", metadata={"chunk_id": "v-1"})
        b_keep = Document(page_content="BM25 现役", metadata={"chunk_id": "b-1"})
        b_drop = Document(page_content="BM25 退役", metadata={"chunk_id": "b-2"})
        retr = self._FakeRetriever([v], [b_keep, b_drop], keep_ids={"b-1"})
        pipeline = self._make_pipeline(retr)
        ctx = self._make_ctx()

        pipeline.retrieve_branches(ctx)

        assert retr.check_calls == 1
        contents = [d.page_content for d in ctx.retrieved_children[0]]
        assert "BM25 现役" in contents
        assert "BM25 退役" not in contents
        assert "向量路结果" in contents
        assert not ctx.degradation_events

    def test_pipeline_recheck_sql_failure_vector_only(self):
        """校验 SQL 异常 → 当前请求 BM25 路整体弃用 + degradation event。"""
        v = Document(page_content="向量路结果", metadata={"chunk_id": "v-1"})
        b = Document(page_content="BM25 候选", metadata={"chunk_id": "b-1"})
        retr = self._FakeRetriever([v], [b], check_error=RuntimeError("PG 连接断开"))
        pipeline = self._make_pipeline(retr)
        ctx = self._make_ctx()

        pipeline.retrieve_branches(ctx)

        contents = [d.page_content for d in ctx.retrieved_children[0]]
        assert contents == ["向量路结果"]  # vector-only
        assert any(
            e["component"] == "bm25_active_check" and e["fallback"] == "vector_only"
            for e in ctx.degradation_events
        )

# ------------------------------------------------------------------
# 9.3 刷新失败保留旧实例 + 旧语料退役内容被过滤（冷启动 + 二次校验组合）
# ------------------------------------------------------------------


class TestRefreshFailureFallback:
    def test_refresh_failure_keeps_old_and_recheck_filters(
        self, collections, monkeypatch
    ):
        if not mgr_module.BM25_AVAILABLE:
            pytest.skip("rank-bm25 未安装")
        # PG 现状：c-old 已退役
        _insert_chunk(
            CONN_STR, SMALL, "旧版本内容",
            {"chunk_id": "c-old", "is_active": False},
        )
        monkeypatch.setitem(vector_store_config, "character_name", "test_bm25")

        # 旧 BM25 语料含 c-old（真实 rank-bm25 构建）
        old_docs = [Document(page_content="旧版本内容", metadata={"chunk_id": "c-old"})]
        retr = _make_retriever(monkeypatch, docs=old_docs)
        old_bm25 = retr._bm25_retriever
        assert old_bm25 is not None

        # 刷新时构建失败 → 保留旧实例，返回 False
        class _BoomBM25:
            @staticmethod
            def from_documents(docs, k=10):
                raise RuntimeError("模拟 BM25 构建失败")

        monkeypatch.setattr(mgr_module, "BM25Retriever", _BoomBM25)
        ok = retr.refresh_documents(
            [Document(page_content="新版本内容", metadata={"chunk_id": "c-new"})]
        )
        assert ok is False
        assert retr._bm25_retriever is old_bm25  # 旧实例保留

        # 旧语料候选经 RRF 前二次校验被剔除 → 退役内容不进最终上下文
        kept = retr.filter_active_bm25(list(old_docs))
        assert kept == []


# ------------------------------------------------------------------
# 12.3 SSE 取消后终止生成
# ------------------------------------------------------------------


class _SlowFakeService:
    """慢速流式生成替身：记录已产出 chunk 数与 close 是否被调。"""

    def __init__(self, total: int = 50, delay: float = 0.05):
        self.total = total
        self.delay = delay
        self.yielded = 0
        self.closed = False

    # 新路径
    def prepare(self, query, enable_rewrite=False, rewrite_strategy=""):
        return SimpleNamespace(
            route_decision="rag",
            final_context="冻结上下文",
            sources=[],
            summary=lambda: {"request_id": "fake"},
        )

    def stream_answer(self, ctx):
        return self._gen()

    # 旧路径
    def retriever_docx(self, query):
        return [Document(page_content="片段", metadata={"source": "s.pdf"})]

    def get_summary_stream(self, query):
        return self._gen()

    def _gen(self):
        try:
            for i in range(self.total):
                time.sleep(self.delay)  # 模拟慢速 LLM token
                self.yielded += 1
                yield f"chunk-{i}"
        finally:
            # close()（GeneratorExit）或正常耗尽都会走到
            self.closed = True


class TestSseCancel:
    @pytest.mark.parametrize("use_new", [True, False])
    def test_cancel_stops_consume_thread(self, monkeypatch, use_new):
        """取消后 _consume_stream 因取消标志退出：生成提前终止、
        底层 stream 被关闭、cancelled 帧语义保留。"""
        from app.gateway.routers import rag as rag_router

        service = _SlowFakeService()
        monkeypatch.setattr(rag_router, "_get_rag_service", lambda: service)
        monkeypatch.setattr(
            rag_router, "_single_retrieval_enabled", lambda: use_new
        )

        async def main():
            frames: list[str] = []

            async def drive():
                resp = await rag_router.rag_query(
                    rag_router.RAGQueryRequest(query="热导率"),
                    current_user=SimpleNamespace(user_id="u"),
                )
                async for frame in resp.body_iterator:
                    frames.append(frame)

            task = asyncio.create_task(drive())
            await asyncio.sleep(0.4)  # 等待进入生成阶段并产出若干 token
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=5)
            except (asyncio.CancelledError, TimeoutError):
                pass
            return "".join(frames)

        body = asyncio.run(main())

        assert service.yielded >= 1  # 确实进入了生成阶段
        assert service.yielded < service.total  # 生成被提前终止
        assert service.closed is True  # 底层 stream 已关闭
        assert "event: cancelled" in body  # cancelled 帧语义保留
        assert "event: done" not in body

    def test_normal_completion_unaffected(self, monkeypatch):
        """正常完成路径不受取消标志影响（回归保护）。"""
        from app.gateway.routers import rag as rag_router

        service = _SlowFakeService(total=3, delay=0.01)
        monkeypatch.setattr(rag_router, "_get_rag_service", lambda: service)
        monkeypatch.setattr(rag_router, "_single_retrieval_enabled", lambda: True)

        async def main():
            frames: list[str] = []
            resp = await rag_router.rag_query(
                rag_router.RAGQueryRequest(query="热导率"),
                current_user=SimpleNamespace(user_id="u"),
            )
            async for frame in resp.body_iterator:
                frames.append(frame)
            return "".join(frames)

        body = asyncio.run(main())
        assert "event: done" in body
        assert service.yielded == 3
        assert service.closed is True


# ------------------------------------------------------------------
# 15 可运维性：kb-state / consistency-check 端点 + 一致性 SQL
# ------------------------------------------------------------------


class TestKbStateConsistency:
    def test_kb_state_returns_revisions(self, pg, kb_id, monkeypatch):
        kb = KbStateRepository(pg, PREFIX)
        kb.increment_active_revision(kb_id)
        store = _make_store(monkeypatch, kb)

        state = store.get_kb_state()
        assert state["kb_id"] == kb_id
        assert state["active_revision"] == 1
        assert state["bm25_revision"] == 0
        assert state["local_bm25_revision"] == 0
        assert state["bm25_lagging"] is True

    def test_consistency_check_clean(self, collections, monkeypatch):
        """干净测试库两项检测均应为 0。"""
        monkeypatch.setitem(vector_store_config, "character_name", "test_bm25")
        _insert_chunk(
            CONN_STR, LARGE, "父块内容",
            {
                "source": "manual://clean", "parent_id": "p-1", "chunk_id": "p-1",
                "document_id": "d-1", "version_id": "v-1", "is_active": True,
            },
        )
        _insert_chunk(
            CONN_STR, SMALL, "子块内容",
            {
                "chunk_id": "c-1",
                "source": "manual://clean",
                "parent_id": "p-1",
                "document_id": "d-1",
                "version_id": "v-1",
                "is_active": True,
            },
        )
        store = VectorStore()
        result = store.run_consistency_check()
        assert result["multiple_active_version_documents"] == 0
        assert result["orphan_active_children"] == 0
        assert result["ok"] is True
        assert result["collections"] == [SMALL, LARGE]

    def test_consistency_check_detects_violations(self, collections, monkeypatch):
        """注入违规数据：多 active version + 孤儿 child 应被检出。"""
        monkeypatch.setitem(vector_store_config, "character_name", "test_bm25")
        # ① 同一 document_id 两个 active version
        for vid in ("v-1", "v-2"):
            _insert_chunk(
                CONN_STR, SMALL, f"{vid} 内容",
                {
                    "chunk_id": f"c-{vid}",
                    "document_id": "d-dup",
                    "version_id": vid,
                    "is_active": True,
                },
            )
        # ② 孤儿 active child（large 中无对应 parent）
        _insert_chunk(
            CONN_STR, SMALL, "孤儿子块",
            {"chunk_id": "c-orphan", "parent_id": "p-missing", "is_active": True},
        )
        store = VectorStore()
        result = store.run_consistency_check()
        assert result["multiple_active_version_documents"] == 1
        assert result["orphan_active_children"] == 1
        assert result["ok"] is False


# ------------------------------------------------------------------
# 端点（TestClient + fake store）
# ------------------------------------------------------------------


class TestKbStateEndpoints:
    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient

        from app.gateway.app import app

        with TestClient(app) as c:
            yield c

    @pytest.fixture()
    def authed_client(self, client):
        from app.gateway.app import app
        from app.gateway.auth.dependencies import get_current_user
        from app.gateway.auth.schemas import UserInfo

        app.dependency_overrides[get_current_user] = lambda: UserInfo(
            user_id="test-u", username="tester", created_at=time.time(), is_admin=True
        )
        try:
            yield client
        finally:
            app.dependency_overrides.pop(get_current_user, None)

    @staticmethod
    def _patch_store(monkeypatch, store):
        from app.gateway.routers import rag as rag_router

        monkeypatch.setattr(rag_router, "_get_vector_store", lambda: store)

    def test_kb_state_unauthenticated_401(self, client):
        assert client.get("/api/v1/rag/kb-state").status_code == 401

    def test_consistency_check_unauthenticated_401(self, client):
        assert client.get("/api/v1/rag/consistency-check").status_code == 401

    def test_kb_state_pg_mode(self, authed_client, monkeypatch):
        store = SimpleNamespace(
            get_kb_state=lambda: {
                "kb_id": "agent",
                "active_revision": 7,
                "bm25_revision": 7,
                "local_bm25_revision": 7,
                "bm25_lagging": False,
                "updated_at": "2026-08-10T00:00:00",
            }
        )
        self._patch_store(monkeypatch, store)
        resp = authed_client.get("/api/v1/rag/kb-state")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 200
        assert body["data"]["active_revision"] == 7
        assert body["data"]["bm25_revision"] == 7

    def test_consistency_check_success(self, authed_client, monkeypatch):
        store = SimpleNamespace(
            run_consistency_check=lambda: {
                "collections": ["agent_small", "agent_large"],
                "multiple_active_version_documents": 0,
                "orphan_active_children": 0,
                "ok": True,
            }
        )
        self._patch_store(monkeypatch, store)
        resp = authed_client.get("/api/v1/rag/consistency-check")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 200
        assert body["data"]["ok"] is True
        assert body["data"]["multiple_active_version_documents"] == 0
        assert body["data"]["orphan_active_children"] == 0

    def test_consistency_check_error_500(self, authed_client, monkeypatch):
        def _boom():
            raise RuntimeError("PG 不可达")

        store = SimpleNamespace(run_consistency_check=_boom)
        self._patch_store(monkeypatch, store)
        resp = authed_client.get("/api/v1/rag/consistency-check")
        assert resp.status_code == 500
        assert "一致性检测失败" in resp.json()["detail"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
