"""请求级检索执行上下文（ONL-001 / ONL-002）。

每次 RAG 请求拥有独立的 ``RequestContext``，记录从路由到生成的全部
中间产物。禁止通过全局对象临时属性传递请求数据。

``final_context`` 一经 ``freeze_context()`` 冻结即不可变：sources 事件与
LLM Prompt 必须来自同一份冻结上下文（ONL-002）。
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document


@dataclass
class RequestContext:
    """单次 RAG 请求的执行上下文。"""

    original_query: str
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    # 路由与改写
    route_result: Any = None  # QueryClassification 或 None
    route_decision: str = ""  # "rag" | "chat" | "irrelevant"（路由决策）
    rewrite_strategy: str = ""
    generated_queries: list[dict] = field(default_factory=list)
    retrieval_plan: dict[str, Any] = field(default_factory=dict)
    # 每项: {"query": str, "weight": float, "type": "original"|"step_back"|"expand"|"hyde"}

    # 请求级 HyDE embedding（ONL-008）：仅作为检索函数参数使用，
    # 禁止写入共享 Retriever/VectorStore 属性
    hyde_embedding: Any = None  # list[float] 或 None

    # 检索各阶段产物
    retrieved_children: list[list[Document]] = field(default_factory=list)  # 各分支候选
    fused_children: list[Document] = field(default_factory=list)  # 跨查询 RRF 后
    reranked_children: list[Document] = field(default_factory=list)  # 全局 Rerank 后
    resolved_parents: list[Document] = field(default_factory=list)  # 有序父块回溯后

    # 冻结上下文与来源
    final_context: str = ""
    context_docs: list[Document] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    context_truncated: bool = False  # 上下文因预算被截断（V2 Spec 10.5）

    # 降级与耗时
    degradation_events: list[dict] = field(default_factory=list)
    stage_durations: dict[str, float] = field(default_factory=dict)  # stage -> ms
    answer_mode: str = ""  # multimodal | text_rag | text_rag_fallback | direct 等

    _frozen: bool = field(default=False, repr=False)

    # ── 上下文冻结 ──────────────────────────────────────────

    def freeze_context(self, context: str, docs: list[Document], sources: list[dict]):
        """写入最终上下文并冻结。之后任何修改尝试都会抛异常。"""
        if self._frozen:
            raise RuntimeError("final_context 已冻结，禁止重复写入")
        self.final_context = context
        self.context_docs = list(docs)
        self.sources = list(sources)
        self._frozen = True

    @property
    def context_frozen(self) -> bool:
        return self._frozen

    def visual_sources(self, *, limit: int = 2, require_completed: bool = False) -> list[dict]:
        """从冻结来源中提取可用于多模态生成的图表证据。"""
        if not self._frozen:
            raise RuntimeError("final_context 尚未冻结，不能读取视觉证据")
        if limit <= 0:
            return []
        evidence: list[dict] = []
        seen: set[str] = set()
        for source in self.sources:
            artifact_id = str(source.get("artifact_id") or "")
            locator = source.get("evidence_locator") or {}
            asset_uri = (
                locator.get("asset_uri")
                or source.get("matched_visual_asset_uri")
                or source.get("visual_asset_uri")
                or source.get("asset_uri")
            )
            if (
                source.get("content_type") != "figure"
                or not asset_uri
                or not artifact_id
                or artifact_id in seen
            ):
                continue
            if require_completed and source.get("visual_status") != "completed":
                continue
            seen.add(artifact_id)
            evidence.append(dict(source))
            if len(evidence) >= limit:
                break
        return evidence

    # ── 降级事件 ────────────────────────────────────────────

    def add_degradation(self, component: str, reason: str, fallback: str):
        """记录一次降级事件（组件、原因、回退策略）。"""
        self.degradation_events.append(
            {
                "component": component,
                "reason": reason[:200],
                "fallback": fallback,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )

    # ── 阶段耗时 ────────────────────────────────────────────

    def record_duration(self, stage: str, start: float):
        """记录阶段耗时（毫秒），start 为 time.time() 起点。"""
        self.stage_durations[stage] = round((time.time() - start) * 1000, 2)

    # ── 日志摘要（不输出完整正文）─────────────────────────────

    def summary(self) -> dict:
        """请求级结构化日志摘要（ONL-001 / 9.1 必备字段）。"""
        query_hash = hashlib_sha256(self.original_query)
        return {
            "request_id": self.request_id,
            "query_hash": query_hash,
            "route": getattr(self.route_result, "question_type", None),
            "retrieval_route": self.retrieval_plan.get("route_type"),
            "retrieval_channels": {
                "text_vector": self.retrieval_plan.get("enable_text_vector", True),
                "bm25": self.retrieval_plan.get("enable_bm25", True),
                "visual": self.retrieval_plan.get("enable_visual", False),
            },
            "rewrite_strategy": self.rewrite_strategy,
            "query_count": len(self.generated_queries),
            "branch_result_counts": [len(b) for b in self.retrieved_children],
            "fused_count": len(self.fused_children),
            "reranked_count": len(self.reranked_children),
            "parent_count": len(self.resolved_parents),
            "final_context_chars": len(self.final_context),
            "context_truncated": self.context_truncated,
            "degradation_events": self.degradation_events,
            "stage_durations": self.stage_durations,
            "answer_mode": self.answer_mode,
        }


def hashlib_sha256(text: str) -> str:
    """返回文本的 sha256 前 16 位（日志脱敏用）。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
