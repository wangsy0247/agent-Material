"""统一只检索入口的契约测试。

验证 RagService.search、Agent rag_search 与问答包装器都只通过
RagService.prepare 执行检索，不再直接调用 legacy retriever 方法。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_material.agents.tools import rag_local_tools  # noqa: E402
from agent_material.rag_engine.rag_service import (  # noqa: E402
    RagSearchResult,
    RagService,
)
from agent_material.rag_engine.request_context import RequestContext  # noqa: E402


class _Logger:
    def info(self, *args, **kwargs):
        pass


def _context(query: str) -> RequestContext:
    kept = Document(
        page_content="父块正文",
        metadata={
            "chunk_id": "c-1",
            "parent_id": "p-1",
            "document_id": "d-1",
            "version_id": "v-1",
            "source": "data/material.pdf",
            "page_start": 2,
            "page_end": 3,
            "rerank_score": 0.91,
        },
    )
    dropped = Document(
        page_content="因上下文预算被丢弃的尾部父块",
        metadata={"parent_id": "p-2", "source": "data/other.pdf"},
    )
    ctx = RequestContext(original_query=query)
    ctx.route_decision = "rag"
    ctx.generated_queries = [{"query": query, "weight": 1.0, "type": "original"}]
    ctx.resolved_parents = [kept, dropped]
    ctx.degradation_events = [{"component": "bm25", "reason": "不可用", "fallback": "vector_only"}]
    ctx.stage_durations = {"retrieve_total": 12.5}
    ctx.context_truncated = True
    ctx.freeze_context(
        "[参考资料1] 来源：material.pdf\n父块正文\n\n",
        [kept],
        [
            {
                "index": 1,
                "source": "data/material.pdf",
                "source_filename": "material.pdf",
                "parent_id": "p-1",
                "chunk_id": "c-1",
                "document_id": "d-1",
                "version_id": "v-1",
                "score": 0.91,
                "snippet": "父块正文",
            }
        ],
    )
    return ctx


def test_service_search_uses_frozen_context_and_forwards_options():
    service = RagService.__new__(RagService)
    service._logger = _Logger()
    calls = []

    def prepare(query, enable_rewrite=False, rewrite_strategy=""):
        calls.append((query, enable_rewrite, rewrite_strategy))
        return _context(query)

    service.prepare = prepare
    result = service.search("Ga2O3 热导率", True, "step_back+expand")

    assert calls == [("Ga2O3 热导率", True, "step_back+expand")]
    assert result.count == 1
    assert [doc.page_content for doc in result.documents] == ["父块正文"]
    assert result.results[0]["chunk_id"] == "c-1"
    assert result.results[0]["document_id"] == "d-1"
    assert result.results[0]["version_id"] == "v-1"
    assert result.results[0]["rank"] == 1
    assert result.results[0]["score"] == 0.91
    assert result.context_truncated is True
    assert result.degradation_events[0]["component"] == "bm25"
    # resolved_parents 中被预算丢弃的文档不得出现在统一搜索结果中。
    assert all("尾部父块" not in item["page_content"] for item in result.results)


def test_search_serialization_exposes_visual_evidence_fields():
    document = Document(
        page_content="[图表] FIG. 4",
        metadata={
            "content_type": "figure",
            "artifact_id": "artifact-4",
            "caption": "FIG. 4",
            "asset_uri": "rag-asset://doc/v1/figure.png",
            "bbox": [1, 2, 3, 4],
            "visual_status": "completed",
            "visual_summary": "累积热导率曲线",
            "visual_model": "fake-vl",
            "visual_confidence": 0.88,
            "visual_prompt_version": "visual-chart-v1",
            "matched_panel_id": "b",
            "matched_visual_entity_id": "artifact-4:panel:b",
            "matched_visual_asset_uri": "rag-asset://doc/v1/panel-b.png",
            "matched_evidence_bbox": [10, 20, 300, 400],
            "matched_evidence_bbox_coordinate_space": "image_pixels",
            "retrieval_channels": ["vector", "visual"],
        },
    )

    item = RagService._serialize_search_document(document, 1)

    assert item["content_type"] == "figure"
    assert item["artifact_id"] == "artifact-4"
    assert item["asset_uri"].startswith("rag-asset://")
    assert item["visual_status"] == "completed"
    assert item["visual_confidence"] == 0.88
    assert item["evidence_locator"]["panel_id"] == "b"
    assert item["evidence_locator"]["asset_uri"].endswith("panel-b.png")
    assert item["evidence_locator"]["bbox"] == [10, 20, 300, 400]
    assert item["evidence_locator"]["retrieval_channels"] == ["vector", "visual"]


def test_agent_rag_search_calls_only_unified_service(monkeypatch):
    ctx = _context("Ga2O3 热导率")
    item = RagService._serialize_search_document(ctx.context_docs[0], 1)
    result = RagSearchResult(
        request_id=ctx.request_id,
        original_query=ctx.original_query,
        route_decision=ctx.route_decision,
        question_type="factual",
        documents=tuple(ctx.context_docs),
        results=(item,),
        sources=tuple(ctx.sources),
        generated_queries=tuple(ctx.generated_queries),
        degradation_events=tuple(ctx.degradation_events),
        stage_durations=dict(ctx.stage_durations),
        context_truncated=ctx.context_truncated,
    )

    class FakeService:
        def __init__(self):
            self.calls = []

        def search(self, query, enable_rewrite=False, rewrite_strategy=""):
            self.calls.append((query, enable_rewrite, rewrite_strategy))
            return result

        def retriever_docx(self, query):
            raise AssertionError("Agent 不得调用 legacy retriever_docx")

        def _retrieve_with_strategy(self, query, strategy):
            raise AssertionError("Agent 不得调用 legacy _retrieve_with_strategy")

    service = FakeService()
    monkeypatch.setattr(rag_local_tools, "_get_rag_service", lambda: service)

    raw = rag_local_tools.rag_search.invoke(
        {
            "query": "Ga2O3 热导率",
            "enable_rewrite": True,
            "rewrite_strategy": "step_back+expand",
        }
    )
    payload = json.loads(raw)

    assert service.calls == [("Ga2O3 热导率", True, "step_back+expand")]
    assert payload["count"] == 1
    assert payload["results"][0]["chunk_id"] == "c-1"
    assert payload["sources"][0]["document_id"] == "d-1"
    assert payload["degradation_events"][0]["component"] == "bm25"


def test_summary_wrappers_always_prepare_once():
    service = RagService.__new__(RagService)
    service._logger = _Logger()
    ctx = _context("问题")
    calls = []

    def prepare(query, enable_rewrite=False, rewrite_strategy=""):
        calls.append((query, enable_rewrite, rewrite_strategy))
        return ctx

    service.prepare = prepare
    service.stream_answer = lambda frozen_ctx: iter(["回答", "完成"])

    assert service.get_summary("问题", True, "expand") == "回答完成"
    assert list(service.get_summary_stream("问题", False, "")) == ["回答", "完成"]
    assert calls == [("问题", True, "expand"), ("问题", False, "")]
