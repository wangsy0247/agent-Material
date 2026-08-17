"""融合模块（weighted RRF）与请求级上下文的单元测试。

对应 Spec：ONL-001 / ONL-002 / ONL-005 / ONL-006 / ONL-010。
"""

import pytest
from langchain_core.documents import Document

from agent_material.rag_engine.fusion import doc_fusion_key, weighted_rrf
from agent_material.rag_engine.request_context import RequestContext


def _doc(content: str, chunk_id: str = "") -> Document:
    meta = {"chunk_id": chunk_id} if chunk_id else {}
    return Document(page_content=content, metadata=meta)


class TestDocFusionKey:
    def test_prefers_chunk_id(self):
        assert doc_fusion_key(_doc("abc", "c1")) == "c1"

    def test_fallback_content_sha256_stable(self):
        k1 = doc_fusion_key(_doc("相同内容"))
        k2 = doc_fusion_key(_doc("相同内容"))
        assert k1 == k2 and k1.startswith("content:")

    def test_fallback_differs_for_different_content(self):
        assert doc_fusion_key(_doc("A")) != doc_fusion_key(_doc("B"))


class TestWeightedRrf:
    def test_single_branch_order_preserved(self):
        docs = [_doc("a", "c1"), _doc("b", "c2"), _doc("c", "c3")]
        fused = weighted_rrf([(docs, 1.0)])
        assert [d.metadata["chunk_id"] for d in fused] == ["c1", "c2", "c3"]

    def test_weight_changes_ranking(self):
        # c2 在高权分支排第一，c1 在低权分支排第一 → 高权获胜
        heavy = [_doc("x", "c2"), _doc("y", "c9")]
        light = [_doc("z", "c1")]
        fused = weighted_rrf([(light, 0.1), (heavy, 1.0)])
        assert fused[0].metadata["chunk_id"] == "c2"

    def test_duplicate_chunk_accumulates_contribution(self):
        # c1 出现在两个分支 → 分数应高于仅出现一次且排名相同的 c2
        branch_a = [_doc("a", "c1"), _doc("b", "c3")]
        branch_b = [_doc("a-dup", "c1"), _doc("c", "c2")]
        fused = weighted_rrf([(branch_a, 1.0), (branch_b, 1.0)])
        ids = [d.metadata["chunk_id"] for d in fused]
        assert ids.count("c1") == 1  # 去重只保留一条
        assert ids[0] == "c1"        # 累加贡献后排第一
        assert fused[0].metadata["rrf_score"] > fused[1].metadata["rrf_score"]

    def test_input_order_determinism(self):
        # 相同分支以相同顺序输入，多次运行结果一致
        b1 = [_doc("a", "c1"), _doc("b", "c2")]
        b2 = [_doc("c", "c3"), _doc("a", "c1")]
        r1 = [d.metadata["chunk_id"] for d in weighted_rrf([(b1, 1.0), (b2, 0.8)])]
        r2 = [d.metadata["chunk_id"] for d in weighted_rrf([(b1, 1.0), (b2, 0.8)])]
        assert r1 == r2

    def test_tie_break_by_original_query_rank_then_chunk_id(self):
        # 两个分支权重相同、chunk 互不重叠且排名相同 → 分数相同；
        # 出现在原始分支（第一个）的 chunk 排前；同分支内按排名。
        original = [_doc("a", "c2")]
        other = [_doc("b", "c1")]
        fused = weighted_rrf([(original, 1.0), (other, 1.0)])
        assert [d.metadata["chunk_id"] for d in fused] == ["c2", "c1"]

    def test_empty_branches(self):
        assert weighted_rrf([([], 1.0), ([], 0.8)]) == []

    def test_rrf_score_written(self):
        fused = weighted_rrf([([_doc("a", "c1")], 1.0)], rrf_k=60)
        assert fused[0].metadata["rrf_score"] == pytest.approx(1.0 / 61.0)


class TestRequestContext:
    def test_freeze_context_immutable(self):
        ctx = RequestContext(original_query="q")
        ctx.freeze_context("ctx", [_doc("a")], [{"index": 1}])
        assert ctx.context_frozen
        with pytest.raises(RuntimeError):
            ctx.freeze_context("other", [], [])

    def test_add_degradation(self):
        ctx = RequestContext(original_query="q")
        ctx.add_degradation("reranker", "timeout", "cross_query_rrf")
        assert ctx.degradation_events[0]["component"] == "reranker"
        assert ctx.degradation_events[0]["fallback"] == "cross_query_rrf"

    def test_summary_contains_required_fields(self):
        ctx = RequestContext(original_query="什么是 DFT？")
        ctx.freeze_context("ctx", [_doc("a")], [{"index": 1}])
        s = ctx.summary()
        for field in (
            "request_id",
            "query_hash",
            "query_count",
            "fused_count",
            "reranked_count",
            "parent_count",
            "final_context_chars",
            "degradation_events",
            "stage_durations",
        ):
            assert field in s
        # 日志摘要不包含原始查询明文
        assert "什么是 DFT？" not in str(s)

    def test_record_duration(self):
        import time

        ctx = RequestContext(original_query="q")
        t = time.time()
        ctx.record_duration("retrieve", t)
        assert ctx.stage_durations["retrieve"] >= 0
