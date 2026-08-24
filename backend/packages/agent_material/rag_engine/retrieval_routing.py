"""Deterministic P2 retrieval routing for text and visual evidence.

This planner complements the LLM chat/irrelevant classifier. It does not make an
external call, so routing remains available when the model is slow or unavailable.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


_VISUAL_TERMS = re.compile(
    r"(?:图\s*\d|figure\s*\d|fig\.?(?:ure)?\s*\d|panel|子图|曲线|图例|"
    r"横轴|纵轴|坐标轴|散点图|柱状图|折线图|热图|峰值|交点)",
    re.IGNORECASE,
)
_PANEL = re.compile(
    r"(?:panel\s*|子图\s*|图\s*\d*\s*)[（(]?\s*([a-z])\s*[)）]?",
    re.IGNORECASE,
)
_NUMERIC = re.compile(
    r"(?:多少|数值|读数|具体值|精确值|约为|达到|等于|"
    r"\d+(?:\.\d+)?\s*(?:k|℃|°c|w/?m(?:·?k)?|ev|gpa|mpa|nm|μm|um))",
    re.IGNORECASE,
)
_TREND = re.compile(r"(?:趋势|变化|升高|降低|增加|减小|随.+变化|最大|最小)")
_COMPARE = re.compile(r"(?:比较|对比|差异|相比|分别|哪条|哪个方向|与.+相比)")
_MULTI_FIGURE = re.compile(
    r"(?:图\s*\d+.*(?:和|与|及|、).*图\s*\d+|"
    r"fig(?:ure)?\.?\s*\d+.*(?:and|vs\.?).*fig(?:ure)?\.?\s*\d+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RetrievalRoutePlan:
    route_type: str
    enable_text_vector: bool = True
    enable_bm25: bool = True
    enable_visual: bool = False
    visual_requested: bool = False
    panel_id: str | None = None
    numeric_fact_required: bool = False
    decomposition_recommended: bool = False
    reason_codes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["reason_codes"] = list(self.reason_codes)
        return result


def plan_retrieval_route(
    query: str,
    *,
    visual_available: bool,
    question_type: str | None = None,
) -> RetrievalRoutePlan:
    """Choose a stable retrieval plan from query evidence and feature availability."""

    text = query.strip()
    visual_requested = bool(_VISUAL_TERMS.search(text))
    panel_match = _PANEL.search(text) if visual_requested else None
    panel_id = panel_match.group(1).lower() if panel_match else None
    numeric = visual_requested and bool(_NUMERIC.search(text))
    trend = visual_requested and bool(_TREND.search(text))
    comparison = visual_requested and (
        bool(_COMPARE.search(text)) or question_type == "comparison"
    )
    multi_figure = visual_requested and bool(_MULTI_FIGURE.search(text))

    reasons: list[str] = []
    if visual_requested:
        reasons.append("visual_reference")
    if panel_id:
        reasons.append("panel_target")
    if numeric:
        reasons.append("numeric_fact")
    if trend:
        reasons.append("trend")
    if comparison:
        reasons.append("comparison")
    if multi_figure:
        reasons.append("multi_figure")
    if visual_requested and not visual_available:
        reasons.append("visual_feature_disabled")

    if not visual_requested:
        route_type = "text_only"
    elif numeric:
        route_type = "numeric_visual"
    elif multi_figure:
        route_type = "multi_figure_visual"
    elif panel_id:
        route_type = "panel_visual"
    elif trend or comparison:
        route_type = "chart_reasoning"
    else:
        route_type = "artifact_visual"

    return RetrievalRoutePlan(
        route_type=route_type,
        enable_visual=visual_requested and visual_available,
        visual_requested=visual_requested,
        panel_id=panel_id,
        numeric_fact_required=numeric,
        decomposition_recommended=multi_figure,
        reason_codes=tuple(reasons),
    )
