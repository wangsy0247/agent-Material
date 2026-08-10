"""请求级检索管道（RetrievalPipeline）与单次检索 SSE 链路测试。

覆盖 Spec ONL-001 ~ ONL-011 的在线链路行为：
- plan_queries：查询计划顺序/权重/上限/去重/改写降级/HyDE 请求级 embedding
- retrieve_branches：分支 RRF、BM25 降级、HyDE 参数传递（不写共享属性）
- fuse：跨查询 weighted RRF、候选截断
- global_rerank：恰好一次调用、rerank_score 回写、失败回退 RRF
- resolve_parents：有序回溯、missing 回退子块
- build_context：字符预算截断、sources 字段齐全、freeze 后不可写
- 并发隔离：20 线程 HyDE embedding 无串扰
- RagService.prepare + stream_answer：chat/irrelevant 不检索、rag 单次检索
- /rag/query SSE：事件序列、event_id/token_index 单调、单次 prepare
- /rag/search：溯源新字段存在

全部使用 fake/mock，不打真实 DashScope/PG/LLM API。
"""

import json
import sys
import threading
import time
from pathlib import Path

import pytest

# backend/ 与 backend/packages 加入路径（pyproject 仅配置了 packages）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document  # noqa: E402

from agent_material.rag_engine.fusion import doc_fusion_key  # noqa: E402
from agent_material.rag_engine.request_context import RequestContext  # noqa: E402
from agent_material.rag_engine.retrieval_pipeline import RetrievalPipeline  # noqa: E402


# ------------------------------------------------------------------
# fake 组件
# ------------------------------------------------------------------


def make_config(**overrides):
    """构造管道测试配置（深合并 overrides）。"""
    cfg = {
        "k": 5,
        "vector_weight": 0.5,
        "bm25_weight": 0.5,
        "reranker": {"top_n": 3},
        "context_budget_chars": 8000,
        "query_rewriting": {
            "default_strategy": "step_back+expand",
            "enable_hyde": False,
            "max_queries": 3,
            "fusion_top_n": 20,
            "weights": {"original": 1.0, "step_back": 0.8, "expand": 0.7},
        },
        "feature_flags": {
            "cross_query_rrf": True,
            "global_rerank": True,
            "request_scoped_hyde": True,
            "single_retrieval_sse": True,
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    return cfg


def make_doc(content, **metadata):
    return Document(page_content=content, metadata=metadata)


class FakeRewriter:
    """可控的 QueryRewriter 替身。"""

    def __init__(
        self,
        step_back_query="材料热导率的通用原理",
        expand_variants=("Ga2O3 导热性能", "氧化镓热导"),
        hyde_doc="假设文档内容",
        fail=(),
    ):
        self._sb = step_back_query
        self._variants = list(expand_variants)
        self._hyde_doc = hyde_doc
        self._fail = set(fail)
        self.calls = []

    def step_back(self, query):
        self.calls.append(("step_back", query))
        if "step_back" in self._fail:
            raise RuntimeError("step_back LLM 失败")
        return (query, self._sb)

    def expand(self, query):
        self.calls.append(("expand", query))
        if "expand" in self._fail:
            raise RuntimeError("expand LLM 失败")
        return [query, *self._variants]

    def hyde(self, query):
        self.calls.append(("hyde", query))
        if "hyde" in self._fail:
            raise RuntimeError("hyde LLM 失败")
        return self._hyde_doc


class FakeEmbedder:
    """生成与文本相关的确定性 embedding。"""

    def __init__(self):
        self.calls = []

    def embed_query(self, text):
        self.calls.append(text)
        return [float(len(text)), 1.0, 0.5]


class FakeBM25:
    def __init__(self, results=None):
        self._results = results if results is not None else {}
        self.calls = []

    def invoke(self, query):
        self.calls.append(query)
        return list(self._results.get(query, []))


class FakeMultiRetriever:
    """MultiGranularityRetriever 替身：记录 search_children 收到的参数。"""

    def __init__(
        self,
        vector_results=None,
        bm25_results=None,
        parents=None,
        vector_error=None,
        bm25_available=True,
    ):
        # vector_results / bm25_results: dict[query] -> list[Document]
        self._vector_results = vector_results if vector_results is not None else {}
        self._bm25_retriever = (
            FakeBM25(bm25_results) if bm25_available else None
        )
        self._parents = parents or {}  # parent_id -> Document
        self._vector_error = vector_error
        self.search_calls = []  # {"query", "k", "hyde_embedding", "thread"}
        self.fetch_calls = []

    def search_children(self, query, k=None, hyde_embedding=None):
        self.search_calls.append(
            {
                "query": query,
                "k": k,
                "hyde_embedding": hyde_embedding,
                "thread": threading.current_thread().name,
            }
        )
        if self._vector_error is not None:
            raise self._vector_error
        bm25_docs = (
            self._bm25_retriever.invoke(query) if self._bm25_retriever else []
        )
        return list(self._vector_results.get(query, [])), bm25_docs

    def _fetch_parents(self, parent_ids):
        """模拟真实实现：按输入顺序返回命中父块 + missing 列表。"""
        self.fetch_calls.append(list(parent_ids))
        ordered = [self._parents[pid] for pid in parent_ids if pid in self._parents]
        missing = [pid for pid in parent_ids if pid not in self._parents]
        return ordered, missing


class FakeReranker:
    """DashScopeReranker 替身：可按分数排序并回写 rerank_score，或抛错。"""

    def __init__(self, scores=None, error=None, top_n=3):
        # scores: dict[content] -> float；None 时全部 0.5
        self._scores = scores
        self._error = error
        self.top_n = top_n
        self.calls = []

    def compress_documents(self, documents, query, callbacks=None):
        self.calls.append({"query": query, "count": len(documents)})
        if self._error is not None:
            raise self._error
        scored = []
        for doc in documents:
            score = (
                self._scores.get(doc.page_content, 0.5)
                if self._scores is not None
                else 0.5
            )
            doc.metadata["rerank_score"] = score
            scored.append((doc, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[: self.top_n]]


def make_pipeline(retriever, rewriter=None, reranker=None, config=None):
    return RetrievalPipeline(
        retriever_getter=lambda: retriever,
        rewriter=rewriter,
        reranker=reranker,
        config=config or make_config(),
    )


def make_ctx(query="Ga2O3 的热导率是多少"):
    return RequestContext(original_query=query)


# ------------------------------------------------------------------
# plan_queries
# ------------------------------------------------------------------


class TestPlanQueries:
    def test_rewrite_disabled_single_original(self):
        pipeline = make_pipeline(FakeMultiRetriever(), rewriter=FakeRewriter())
        ctx = make_ctx()
        pipeline.plan_queries(ctx, enable_rewrite=False, strategy="")
        assert ctx.generated_queries == [
            {"query": ctx.original_query, "weight": 1.0, "type": "original"}
        ]
        assert "plan_queries" in ctx.stage_durations

    def test_rewrite_order_weights_and_cap(self):
        rewriter = FakeRewriter(
            expand_variants=["变体A", "变体B", "变体C"]  # 超过 max_queries
        )
        pipeline = make_pipeline(FakeMultiRetriever(), rewriter=rewriter)
        ctx = make_ctx()
        pipeline.plan_queries(
            ctx, enable_rewrite=True, strategy="step_back+expand"
        )
        queries = ctx.generated_queries
        # 确定顺序：original → step_back → expand；总数 ≤ 3
        assert len(queries) == 3
        assert queries[0]["type"] == "original"
        assert queries[0]["weight"] == 1.0
        assert queries[1] == {
            "query": "材料热导率的通用原理",
            "weight": 0.8,
            "type": "step_back",
        }
        assert queries[2] == {"query": "变体A", "weight": 0.7, "type": "expand"}
        # 原始 query 始终在列
        assert queries[0]["query"] == ctx.original_query

    def test_rewrite_dedup(self):
        # expand 返回的第一个变体与 step_back 结果相同 → 去重
        rewriter = FakeRewriter(
            step_back_query="重复查询", expand_variants=["重复查询", "新查询"]
        )
        pipeline = make_pipeline(FakeMultiRetriever(), rewriter=rewriter)
        ctx = make_ctx()
        pipeline.plan_queries(ctx, enable_rewrite=True, strategy="step_back+expand")
        texts = [q["query"] for q in ctx.generated_queries]
        assert len(texts) == len(set(texts))
        assert texts == [ctx.original_query, "重复查询", "新查询"]

    def test_rewrite_failure_keeps_original_with_degradation(self):
        rewriter = FakeRewriter(fail={"step_back", "expand"})
        pipeline = make_pipeline(FakeMultiRetriever(), rewriter=rewriter)
        ctx = make_ctx()
        pipeline.plan_queries(ctx, enable_rewrite=True, strategy="step_back+expand")
        # 改写全部失败 → 只剩原始 query，不中断
        assert [q["query"] for q in ctx.generated_queries] == [ctx.original_query]
        components = {e["component"] for e in ctx.degradation_events}
        assert "rewrite:step_back" in components
        assert "rewrite:expand" in components

    def test_hyde_embedding_stored_in_ctx(self, monkeypatch):
        from agent_material.rag_engine import models

        embedder = FakeEmbedder()
        monkeypatch.setattr(models, "embedding_model", embedder)
        config = make_config(
            query_rewriting={"enable_hyde": True},
        )
        pipeline = make_pipeline(
            FakeMultiRetriever(), rewriter=FakeRewriter(), config=config
        )
        ctx = make_ctx()
        pipeline.plan_queries(ctx, enable_rewrite=False, strategy="")
        assert ctx.hyde_embedding == [float(len("假设文档内容")), 1.0, 0.5]
        assert embedder.calls == ["假设文档内容"]

    def test_hyde_failure_fallback_with_degradation(self, monkeypatch):
        from agent_material.rag_engine import models

        embedder = FakeEmbedder()
        monkeypatch.setattr(models, "embedding_model", embedder)
        config = make_config(query_rewriting={"enable_hyde": True})
        rewriter = FakeRewriter(hyde_doc="")  # 空假设文档 → 失败
        pipeline = make_pipeline(
            FakeMultiRetriever(), rewriter=rewriter, config=config
        )
        ctx = make_ctx()
        pipeline.plan_queries(ctx, enable_rewrite=False, strategy="")
        assert ctx.hyde_embedding is None
        assert any(e["component"] == "hyde" for e in ctx.degradation_events)


# ------------------------------------------------------------------
# retrieve_branches
# ------------------------------------------------------------------


class TestRetrieveBranches:
    def test_branch_rrf_merges_vector_and_bm25(self):
        shared = make_doc("共享文档", chunk_id="c-shared")
        v_only = make_doc("向量独有", chunk_id="c-v")
        b_only = make_doc("BM25独有", chunk_id="c-b")
        query = "Ga2O3 的热导率是多少"
        retriever = FakeMultiRetriever(
            vector_results={query: [shared, v_only]},
            bm25_results={query: [shared, b_only]},
        )
        pipeline = make_pipeline(retriever)
        ctx = make_ctx()
        pipeline.plan_queries(ctx, enable_rewrite=False)
        pipeline.retrieve_branches(ctx)

        assert len(ctx.retrieved_children) == 1
        branch = ctx.retrieved_children[0]
        # 两路都命中的文档 RRF 分数最高，排第一
        assert branch[0].metadata["chunk_id"] == "c-shared"
        assert len(branch) == 3
        assert all("rrf_score" in d.metadata for d in branch)

    def test_bm25_unavailable_vector_only_with_degradation(self):
        query = "Ga2O3 的热导率是多少"
        retriever = FakeMultiRetriever(
            vector_results={query: [make_doc("v", chunk_id="c1")]},
            bm25_available=False,
        )
        pipeline = make_pipeline(retriever)
        ctx = make_ctx()
        pipeline.plan_queries(ctx, enable_rewrite=False)
        pipeline.retrieve_branches(ctx)
        assert len(ctx.retrieved_children[0]) == 1
        assert any(e["component"] == "bm25" for e in ctx.degradation_events)

    def test_hyde_embedding_passed_as_parameter_only(self, monkeypatch):
        from agent_material.rag_engine import models

        embedder = FakeEmbedder()
        monkeypatch.setattr(models, "embedding_model", embedder)
        config = make_config(query_rewriting={"enable_hyde": True})
        retriever = FakeMultiRetriever()
        pipeline = make_pipeline(retriever, rewriter=FakeRewriter(), config=config)
        ctx = make_ctx()
        pipeline.plan_queries(ctx, enable_rewrite=True, strategy="step_back")
        pipeline.retrieve_branches(ctx)

        assert len(retriever.search_calls) == 2
        # 仅 original 分支收到 hyde embedding，且与 ctx 中一致
        assert retriever.search_calls[0]["hyde_embedding"] is ctx.hyde_embedding
        assert retriever.search_calls[1]["hyde_embedding"] is None
        # 不写任何共享属性
        assert not hasattr(retriever, "embedding_override")
        assert getattr(retriever, "hyde_embedding", None) is None

    def test_vector_error_falls_back_to_bm25_only(self):
        query = "Ga2O3 的热导率是多少"
        retriever = FakeMultiRetriever(
            bm25_results={query: [make_doc("b", chunk_id="c-b")]},
            vector_error=RuntimeError("PGVector 连接失败"),
        )
        pipeline = make_pipeline(retriever)
        ctx = make_ctx()
        pipeline.plan_queries(ctx, enable_rewrite=False)
        pipeline.retrieve_branches(ctx)
        assert [d.metadata["chunk_id"] for d in ctx.retrieved_children[0]] == ["c-b"]
        assert any(
            e["component"] == "vector_search" for e in ctx.degradation_events
        )

    def test_vector_and_bm25_both_unavailable_raises(self):
        retriever = FakeMultiRetriever(
            vector_error=RuntimeError("PGVector 连接失败"),
            bm25_available=False,
        )
        pipeline = make_pipeline(retriever)
        ctx = make_ctx()
        pipeline.plan_queries(ctx, enable_rewrite=False)
        with pytest.raises(RuntimeError, match="无法完成召回"):
            pipeline.retrieve_branches(ctx)


# ------------------------------------------------------------------
# fuse
# ------------------------------------------------------------------


class TestFuse:
    def _ctx_with_branches(self, branches, queries):
        ctx = make_ctx()
        ctx.retrieved_children = branches
        ctx.generated_queries = queries
        return ctx

    def test_cross_query_rrf_dedup_and_weight(self):
        shared = make_doc("共享", chunk_id="c-shared")
        first_only = make_doc("仅第一分支", chunk_id="c-first")
        second_only = make_doc("仅第二分支", chunk_id="c-second")
        ctx = self._ctx_with_branches(
            branches=[[first_only, shared], [shared, second_only]],
            queries=[
                {"query": "q1", "weight": 1.0, "type": "original"},
                {"query": "q2", "weight": 0.8, "type": "step_back"},
            ],
        )
        make_pipeline(FakeMultiRetriever()).fuse(ctx)
        fused = ctx.fused_children
        # 共享文档跨查询累加贡献 → 排第一；同 chunk_id 只保留一条
        assert fused[0].metadata["chunk_id"] == "c-shared"
        keys = [doc_fusion_key(d) for d in fused]
        assert len(keys) == len(set(keys)) == 3
        # 第一分支（原始 query，权重 1.0）的独有文档 > 第二分支（0.8）独有文档
        assert keys.index("c-first") < keys.index("c-second")

    def test_fuse_top_n_truncation(self):
        branches = [
            [make_doc(f"doc{i}", chunk_id=f"c{i}") for i in range(5)]
        ]
        ctx = self._ctx_with_branches(
            branches=branches,
            queries=[{"query": "q", "weight": 1.0, "type": "original"}],
        )
        config = make_config(query_rewriting={"fusion_top_n": 2})
        make_pipeline(FakeMultiRetriever(), config=config).fuse(ctx)
        assert len(ctx.fused_children) == 2

    def test_fuse_flag_off_stable_dedup(self):
        a = make_doc("A", chunk_id="c-a")
        a_dup = make_doc("A", chunk_id="c-a")
        b = make_doc("B", chunk_id="c-b")
        ctx = self._ctx_with_branches(
            branches=[[a], [a_dup, b]],
            queries=[
                {"query": "q1", "weight": 1.0, "type": "original"},
                {"query": "q2", "weight": 0.8, "type": "step_back"},
            ],
        )
        config = make_config(feature_flags={"cross_query_rrf": False})
        make_pipeline(FakeMultiRetriever(), config=config).fuse(ctx)
        # 开关关闭：按分支顺序拼接 + chunk_id 稳定去重（非随机 hash）
        assert [d.metadata["chunk_id"] for d in ctx.fused_children] == ["c-a", "c-b"]


# ------------------------------------------------------------------
# global_rerank
# ------------------------------------------------------------------


class TestGlobalRerank:
    def _ctx_with_fused(self, contents):
        ctx = make_ctx()
        ctx.fused_children = [make_doc(c, chunk_id=f"c-{c}") for c in contents]
        return ctx

    def test_rerank_success_sorts_and_writes_score(self):
        reranker = FakeReranker(scores={"低": 0.1, "高": 0.9, "中": 0.5})
        pipeline = make_pipeline(FakeMultiRetriever(), reranker=reranker)
        ctx = self._ctx_with_fused(["低", "中", "高"])
        pipeline.global_rerank(ctx)
        assert [d.page_content for d in ctx.reranked_children] == ["高", "中", "低"]
        assert ctx.reranked_children[0].metadata["rerank_score"] == 0.9
        # reranker query 必须是 original_query
        assert reranker.calls[0]["query"] == ctx.original_query

    def test_rerank_called_exactly_once(self):
        reranker = FakeReranker()
        pipeline = make_pipeline(FakeMultiRetriever(), reranker=reranker)
        ctx = self._ctx_with_fused(["a", "b"])
        pipeline.global_rerank(ctx)
        assert len(reranker.calls) == 1

    def test_rerank_failure_falls_back_to_rrf_order(self):
        reranker = FakeReranker(error=RuntimeError("DashScope 限流"))
        pipeline = make_pipeline(FakeMultiRetriever(), reranker=reranker)
        ctx = self._ctx_with_fused(["一", "二", "三"])
        pipeline.global_rerank(ctx)
        assert [d.page_content for d in ctx.reranked_children] == ["一", "二", "三"]
        assert any(e["component"] == "reranker" for e in ctx.degradation_events)

    def test_rerank_flag_off_uses_rrf_order(self):
        reranker = FakeReranker()
        config = make_config(feature_flags={"global_rerank": False})
        pipeline = make_pipeline(FakeMultiRetriever(), reranker=reranker, config=config)
        ctx = self._ctx_with_fused(["x", "y"])
        pipeline.global_rerank(ctx)
        assert reranker.calls == []
        assert [d.page_content for d in ctx.reranked_children] == ["x", "y"]

    def test_rerank_top_n_from_config(self):
        reranker = FakeReranker(top_n=2)
        config = make_config(reranker={"top_n": 2})
        pipeline = make_pipeline(FakeMultiRetriever(), reranker=reranker, config=config)
        ctx = self._ctx_with_fused(["a", "b", "c", "d"])
        pipeline.global_rerank(ctx)
        assert len(ctx.reranked_children) == 2


# ------------------------------------------------------------------
# resolve_parents
# ------------------------------------------------------------------


class TestResolveParents:
    def test_ordered_resolve_with_missing_fallback(self):
        children = [
            make_doc("子块1", chunk_id="c1", parent_id="p1", rrf_score=0.9),
            make_doc("子块2", chunk_id="c2", parent_id="p2", rrf_score=0.8),
            make_doc("子块1b", chunk_id="c1b", parent_id="p1"),  # 同父块去重
            make_doc("子块3", chunk_id="c3", parent_id="p3", rrf_score=0.7),
        ]
        parents = {
            "p2": make_doc("父块2", parent_id="p2"),
            "p1": make_doc("父块1", parent_id="p1"),
            # p3 缺失 → missing 回退子块
        }
        retriever = FakeMultiRetriever(parents=parents)
        pipeline = make_pipeline(retriever)
        ctx = make_ctx()
        ctx.reranked_children = children
        pipeline.resolve_parents(ctx)

        # _fetch_parents 收到的 parent_id 顺序与子块相关性顺序一致且去重
        assert retriever.fetch_calls == [["p1", "p2", "p3"]]
        resolved = ctx.resolved_parents
        assert [d.metadata.get("parent_id") for d in resolved] == ["p1", "p2", "p3"]
        assert resolved[0].page_content == "父块1"
        assert resolved[1].page_content == "父块2"
        # missing 回退子块并标记 parent_fallback
        assert resolved[2].page_content == "子块3"
        assert resolved[2].metadata["parent_fallback"] is True
        assert any(
            e["component"] == "parent_fetch" for e in ctx.degradation_events
        )

    def test_no_parent_id_returns_children(self):
        children = [make_doc("无父块", chunk_id="c1")]
        pipeline = make_pipeline(FakeMultiRetriever())
        ctx = make_ctx()
        ctx.reranked_children = children
        pipeline.resolve_parents(ctx)
        assert ctx.resolved_parents == children


# ------------------------------------------------------------------
# build_context
# ------------------------------------------------------------------


class TestBuildContext:
    def _parent(self, content, **meta):
        base = {"parent_id": "p1", "source": "data/ga2o3.pdf"}
        base.update(meta)
        return make_doc(content, **base)

    def test_budget_truncation_in_order(self):
        docs = [
            self._parent("甲" * 100, parent_id="p1"),
            self._parent("乙" * 100, parent_id="p2"),
            self._parent("丙" * 100, parent_id="p3"),
        ]
        # 预算只够放下第一块（每块约 123 字符）
        config = make_config(context_budget_chars=200)
        pipeline = make_pipeline(FakeMultiRetriever(), config=config)
        ctx = make_ctx()
        ctx.reranked_children = [
            make_doc("子1", parent_id="p1", rerank_score=0.9),
        ]
        ctx.resolved_parents = docs
        pipeline.build_context(ctx)

        assert ctx.context_frozen
        assert "甲" in ctx.final_context
        assert "乙" not in ctx.final_context
        assert "丙" not in ctx.final_context
        # sources 与上下文同源
        assert len(ctx.sources) == len(ctx.context_docs) == 1

    def test_sources_fields_complete(self):
        doc = self._parent(
            "正文内容",
            parent_id="p1",
            source_filename="ga2o3.pdf",
            page_start=3,
            page_end=5,
            page_unknown=False,
            heading_path="第2章/热导率",
        )
        pipeline = make_pipeline(FakeMultiRetriever())
        ctx = make_ctx()
        ctx.reranked_children = [
            make_doc("子1", parent_id="p1", rerank_score=0.87)
        ]
        ctx.resolved_parents = [doc]
        pipeline.build_context(ctx)

        src = ctx.sources[0]
        assert src["index"] == 1
        assert src["source"] == "data/ga2o3.pdf"
        assert src["source_filename"] == "ga2o3.pdf"
        assert src["page_start"] == 3
        assert src["page_end"] == 5
        assert src["page_unknown"] is False
        assert src["heading_path"] == "第2章/热导率"
        assert src["parent_id"] == "p1"
        # score 来自子块的 rerank_score
        assert src["score"] == 0.87
        assert "页码：3-5" in ctx.final_context
        assert "[参考资料1] 来源：ga2o3.pdf" in ctx.final_context

    def test_page_unknown_omits_page_line(self):
        doc = self._parent("正文", page_unknown=True, page_start=None)
        pipeline = make_pipeline(FakeMultiRetriever())
        ctx = make_ctx()
        ctx.resolved_parents = [doc]
        pipeline.build_context(ctx)
        assert "页码" not in ctx.final_context
        assert ctx.sources[0]["page_unknown"] is True

    def test_freeze_twice_raises(self):
        pipeline = make_pipeline(FakeMultiRetriever())
        ctx = make_ctx()
        ctx.resolved_parents = [self._parent("正文")]
        pipeline.build_context(ctx)
        with pytest.raises(RuntimeError, match="已冻结"):
            ctx.freeze_context("x", [], [])


# ------------------------------------------------------------------
# 并发隔离（ONL-008：20 线程 HyDE 无串扰）
# ------------------------------------------------------------------


class TestConcurrencyIsolation:
    def test_20_threads_hyde_no_crosstalk(self):
        retriever = FakeMultiRetriever()
        config = make_config(query_rewriting={"enable_hyde": False})
        pipeline = make_pipeline(retriever, rewriter=FakeRewriter(), config=config)

        n_threads = 20
        errors: list[str] = []
        barrier = threading.Barrier(n_threads)

        def worker(i: int):
            try:
                ctx = make_ctx(query=f"线程{i}的查询")
                embedding = [float(i), float(i) + 0.1]
                ctx.hyde_embedding = embedding
                ctx.generated_queries = [
                    {"query": ctx.original_query, "weight": 1.0, "type": "original"}
                ]
                barrier.wait()  # 尽量同时进入检索
                pipeline.retrieve_branches(ctx)
            except Exception as e:  # noqa: BLE001
                errors.append(f"线程{i} 异常: {e}")

        threads = [
            threading.Thread(target=worker, args=(i,), name=f"t{i}")
            for i in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(retriever.search_calls) == n_threads
        for call in retriever.search_calls:
            tid = int(call["thread"][1:])
            # 每次调用收到的 embedding 必须与发起线程自己的一致
            assert call["hyde_embedding"] == [float(tid), float(tid) + 0.1]
            assert call["query"] == f"线程{tid}的查询"
        # 共享检索器上没有写入任何请求级状态
        assert not hasattr(retriever, "embedding_override")
        assert getattr(retriever, "hyde_embedding", None) is None


# ------------------------------------------------------------------
# RagService.prepare + stream_answer
# ------------------------------------------------------------------


class _FakeChain:
    """RAG 生成链替身（prompt | model | parser 的最终形态）。"""

    def __init__(self, chunks=("回答", "内容")):
        self.chunks = list(chunks)
        self.stream_inputs = []

    def stream(self, inputs, config=None):
        self.stream_inputs.append(inputs)
        yield from self.chunks


def _make_service(monkeypatch, classification, retriever, reranker=None, config=None):
    """构造全 fake 的 RagService（不打真实 LLM/PG/DashScope）。"""
    from agent_material.rag_engine import rag_service as rag_service_module
    from agent_material.rag_engine.query_router import QueryClassification

    config = config or make_config()
    # 模块级配置替换：single_retrieval_sse 开启 + 路由开启
    module_cfg = dict(config)
    module_cfg["query_routing"] = {
        "enabled": True,
        "chat_detection": True,
        "irrelevant_detection": True,
        "confidence_threshold": 0.6,
    }
    monkeypatch.setattr(rag_service_module, "vector_store_config", module_cfg)
    # 避免真实 VectorStore / prompt 加载
    monkeypatch.setattr(rag_service_module, "VectorStore", lambda: object())
    monkeypatch.setattr(rag_service_module, "get_prompts", lambda path: "{input}\n{context}")
    monkeypatch.setattr(
        rag_service_module, "prompt_config", {"rag_prompt_path": "x"}
    )

    service = rag_service_module.RagService()

    class _FakeRouter:
        def classify(self, query):
            return QueryClassification(**classification)

    service._router = _FakeRouter()
    service._pipeline = RetrievalPipeline(
        retriever_getter=lambda: retriever,
        rewriter=FakeRewriter(),
        reranker=reranker or FakeReranker(),
        config=config,
    )
    return service


class TestPrepareAndStreamAnswer:
    def test_chat_skips_retrieval(self, monkeypatch):
        retriever = FakeMultiRetriever()
        service = _make_service(
            monkeypatch,
            classification={
                "is_chat": True,
                "is_irrelevant": False,
                "question_type": "unknown",
                "confidence": 0.95,
                "reasoning": "闲聊",
            },
            retriever=retriever,
        )
        ctx = service.prepare("你好")
        assert ctx.route_decision == "chat"
        assert retriever.search_calls == []  # 未检索
        assert not ctx.context_frozen

    def test_irrelevant_skips_retrieval_and_refuses(self, monkeypatch):
        retriever = FakeMultiRetriever()
        service = _make_service(
            monkeypatch,
            classification={
                "is_chat": False,
                "is_irrelevant": True,
                "question_type": "unknown",
                "confidence": 0.9,
                "reasoning": "无关",
            },
            retriever=retriever,
        )
        ctx = service.prepare("今天天气怎么样")
        assert ctx.route_decision == "irrelevant"
        assert retriever.search_calls == []
        answer = "".join(service.stream_answer(ctx))
        assert "无关" in answer
        assert retriever.search_calls == []  # 流式生成也不检索

    def test_rag_retrieves_exactly_once_and_stream_reuses(self, monkeypatch):
        query = "Ga2O3 的热导率是多少"
        child = make_doc("子块", chunk_id="c1", parent_id="p1")
        parent = make_doc("父块正文", parent_id="p1", source="data/ga2o3.pdf")
        retriever = FakeMultiRetriever(
            vector_results={query: [child]},
            bm25_results={query: []},
            parents={"p1": parent},
        )
        reranker = FakeReranker()
        service = _make_service(
            monkeypatch,
            classification={
                "is_chat": False,
                "is_irrelevant": False,
                "question_type": "factual",
                "confidence": 0.9,
                "reasoning": "事实性问题",
            },
            retriever=retriever,
            reranker=reranker,
        )
        ctx = service.prepare(query)

        # 只检索一次（一个分支一次 search_children），reranker 恰好一次
        assert len(retriever.search_calls) == 1
        assert len(reranker.calls) == 1
        assert ctx.context_frozen
        assert ctx.route_decision == "rag"
        assert "父块正文" in ctx.final_context
        # question_type 仅记录不改变策略（ONL-003）
        assert ctx.route_result.question_type == "factual"

        # stream_answer 不再次检索
        service._chain = _FakeChain()
        answer = "".join(service.stream_answer(ctx))
        assert answer == "回答内容"
        assert len(retriever.search_calls) == 1
        assert len(reranker.calls) == 1
        # prompt 使用冻结上下文
        assert service._chain.stream_inputs[0]["context"] == ctx.final_context

    def test_rag_empty_context_answer(self, monkeypatch):
        service = _make_service(
            monkeypatch,
            classification={
                "is_chat": False,
                "is_irrelevant": False,
                "question_type": "factual",
                "confidence": 0.9,
                "reasoning": "x",
            },
            retriever=FakeMultiRetriever(),
        )
        ctx = service.prepare("没有任何结果的查询")
        assert ctx.context_frozen
        assert ctx.final_context == ""
        answer = "".join(service.stream_answer(ctx))
        assert answer == "根据现有资料无法回答该问题。"


# ------------------------------------------------------------------
# /rag/query SSE 与 /rag/search（FastAPI 路由层）
# ------------------------------------------------------------------


def _parse_sse(body: str):
    """解析 SSE 帧为 [(event, data_dict, event_id)] 列表（忽略心跳注释帧）。"""
    frames = []
    for raw in body.split("\n\n"):
        raw = raw.strip()
        if not raw or raw.startswith(":"):
            continue
        event = None
        event_id = None
        data = None
        for line in raw.splitlines():
            if line.startswith("id: "):
                event_id = int(line[4:])
            elif line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        frames.append((event, data, event_id))
    return frames


class TestRagQuerySSE:
    @pytest.fixture()
    def authed_client(self, monkeypatch):
        from fastapi.testclient import TestClient
        from app.gateway.app import app
        from app.gateway.auth.dependencies import get_current_user
        from app.gateway.auth.schemas import UserInfo
        from app.gateway.routers import rag as rag_router

        docs = [
            make_doc(
                "父块内容",
                parent_id="p1",
                source="data/ga2o3.pdf",
                source_filename="ga2o3.pdf",
                page_start=1,
                page_end=2,
            )
        ]

        class FakeCtx:
            def __init__(self, query):
                self.original_query = query
                self.route_decision = "rag"
                self.final_context = "[参考资料1] 来源：ga2o3.pdf\n父块内容\n\n"
                self.context_docs = docs
                self.sources = [
                    {
                        "index": 1,
                        "source": "data/ga2o3.pdf",
                        "source_filename": "ga2o3.pdf",
                        "page_start": 1,
                        "page_end": 2,
                        "page_unknown": False,
                        "heading_path": None,
                        "parent_id": "p1",
                        "score": 0.9,
                        "snippet": "父块内容",
                    }
                ]
                self.resolved_parents = docs
                self.reranked_children = []
                self.degradation_events = []

            def summary(self):
                return {"request_id": "fake-req"}

        class FakeService:
            def __init__(self):
                self.prepare_calls = 0
                self.stream_ctxs = []

            def prepare(self, query, enable_rewrite=False, rewrite_strategy=""):
                self.prepare_calls += 1
                return FakeCtx(query)

            def stream_answer(self, ctx):
                self.stream_ctxs.append(ctx)
                yield "回答"
                yield "完毕"

        service = FakeService()
        monkeypatch.setattr(rag_router, "_get_rag_service", lambda: service)
        # 强制走新路径（不依赖真实 yaml 开关状态）
        monkeypatch.setattr(rag_router, "_single_retrieval_enabled", lambda: True)
        app.dependency_overrides[get_current_user] = lambda: UserInfo(
            user_id="test-u", username="tester", created_at=time.time()
        )
        try:
            with TestClient(app) as client:
                yield client, service
        finally:
            app.dependency_overrides.pop(get_current_user, None)

    def test_sse_event_sequence_and_monotonic_ids(self, authed_client):
        client, service = authed_client
        resp = client.post("/api/v1/rag/query", json={"query": "Ga2O3 热导率"})
        assert resp.status_code == 200
        frames = _parse_sse(resp.text)

        events = [f[0] for f in frames]
        # 严格事件序列：status(retrieving) → start → sources → status(generating) → token* → done
        assert events[0] == "status"
        assert frames[0][1]["content"] == {"phase": "retrieving"}
        assert events[1] == "start"
        assert events[2] == "sources"
        assert events[3] == "status"
        assert frames[3][1]["content"] == {"phase": "generating"}
        assert events[-1] == "done"
        assert events[4:-1] == ["token", "token"]

        # event_id 单调递增
        ids = [f[2] for f in frames]
        assert ids == sorted(ids) and len(set(ids)) == len(ids)

        # token_index 从 1 单调递增
        token_indexes = [f[1]["index"] for f in frames if f[0] == "token"]
        assert token_indexes == [1, 2]

        # 全程只调一次 prepare，sources 与 token 来自同一 ctx
        assert service.prepare_calls == 1
        assert len(service.stream_ctxs) == 1

        sources_frame = frames[2][1]["content"]
        assert sources_frame[0]["source_filename"] == "ga2o3.pdf"
        assert sources_frame[0]["parent_id"] == "p1"
        assert sources_frame[0]["score"] == 0.9
        token_text = "".join(f[1]["content"] for f in frames if f[0] == "token")
        assert token_text == "回答完毕"

    def test_sse_empty_sources_short_circuit(self, authed_client, monkeypatch):
        client, service = authed_client
        # 构造空结果 ctx
        original_prepare = service.prepare

        def empty_prepare(query, enable_rewrite=False, rewrite_strategy=""):
            ctx = original_prepare(query)
            ctx.final_context = ""
            ctx.sources = []
            return ctx

        service.prepare = empty_prepare
        resp = client.post("/api/v1/rag/query", json={"query": "无结果"})
        frames = _parse_sse(resp.text)
        events = [f[0] for f in frames]
        assert events == ["status", "start", "sources", "token", "done"]
        assert frames[2][1]["content"] == []  # 空 sources 数组
        assert frames[3][1]["content"] == "根据现有资料无法回答该问题。"
        assert "generating" not in resp.text


class TestRagSearchNewFields:
    @pytest.fixture()
    def authed_client(self, monkeypatch):
        from fastapi.testclient import TestClient
        from app.gateway.app import app
        from app.gateway.auth.dependencies import get_current_user
        from app.gateway.auth.schemas import UserInfo
        from app.gateway.routers import rag as rag_router

        doc = make_doc(
            "父块内容片段",
            parent_id="p1",
            chunk_id="c9",
            document_id="d1",
            version_id="v1",
            source="data/ga2o3.pdf",
            source_filename="ga2o3.pdf",
            page_start=3,
            page_end=4,
            page_unknown=False,
            heading_path="第1章",
            rerank_score=0.77,
        )

        class FakeCtx:
            route_decision = "rag"
            resolved_parents = [doc]
            reranked_children = []
            degradation_events = [
                {"component": "bm25", "reason": "不可用", "fallback": "vector_only"}
            ]

            def summary(self):
                return {}

        class FakeService:
            def prepare(self, query, enable_rewrite=False, rewrite_strategy=""):
                return FakeCtx()

        monkeypatch.setattr(rag_router, "_get_rag_service", lambda: FakeService())
        monkeypatch.setattr(rag_router, "_single_retrieval_enabled", lambda: True)
        app.dependency_overrides[get_current_user] = lambda: UserInfo(
            user_id="test-u", username="tester", created_at=time.time()
        )
        try:
            with TestClient(app) as client:
                yield client
        finally:
            app.dependency_overrides.pop(get_current_user, None)

    def test_search_returns_traceability_fields(self, authed_client):
        resp = authed_client.post("/api/v1/rag/search", json={"query": "热导率"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 200
        item = body["data"][0]
        assert item["chunk_id"] == "c9"
        assert item["parent_id"] == "p1"
        assert item["document_id"] == "d1"
        assert item["version_id"] == "v1"
        assert item["source"] == "data/ga2o3.pdf"
        assert item["page_start"] == 3
        assert item["page_end"] == 4
        assert item["page_unknown"] is False
        assert item["heading_path"] == "第1章"
        assert item["rank"] == 1
        assert item["rerank_score"] == 0.77
        assert item["score"] == 0.77
        # 响应级降级事件
        assert body["degradation_events"][0]["component"] == "bm25"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
