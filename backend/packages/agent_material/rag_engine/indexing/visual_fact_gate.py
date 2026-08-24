"""视觉描述事实门禁。

模型输出可以通过 Schema 校验但仍然读错图例或数值。本模块采用保守策略：模型
自报置信度不构成验证；多序列趋势和精确数值只有在人工或可信解析器显式核验后
才能进入可信检索文本与精确数值回答。
"""

from __future__ import annotations

import re
from typing import Any

from agent_material.utils.config_handler import vector_store_config


TRUSTED_VERIFICATION_SOURCES = frozenset({"human", "trusted_parser", "gold_eval"})
NUMERIC_GUARDRAIL_REPLY = (
    "当前命中的图表尚未通过图例映射与坐标数值核验，因此不能可靠给出精确数值。"
    "我可以说明图表主题和坐标含义；如需精确读数，请先完成高清图表的人工或可信解析器核验。"
)


def numeric_fact_gate_config() -> dict[str, Any]:
    structured = vector_store_config.get("structured_index", {}) or {}
    visual = structured.get("visual_enrichment", {}) or {}
    return dict(visual.get("numeric_fact_gate", {}) or {})


def _verification_source(metadata: dict[str, Any], field: str) -> str:
    return str(metadata.get(field) or "").strip().lower()


def _readability(metadata: dict[str, Any]) -> str:
    cfg = numeric_fact_gate_config()
    status = str(metadata.get("visual_preprocess_status") or "unknown")
    width = int(metadata.get("visual_width_px") or 0)
    height = int(metadata.get("visual_height_px") or 0)
    min_width = int(cfg.get("min_readable_width_px", 1000))
    min_height = int(cfg.get("min_readable_height_px", 700))
    if status == "failed" or width <= 0 or height <= 0:
        return "unknown"
    if width >= min_width and height >= min_height:
        return "readable"
    return "limited"


def _contains_physical_value(text: str) -> bool:
    units = (
        r"K|THz|GHz|Hz|W\s*(?:m|cm)?(?:\^?-?1|⁻¹)?(?:\s*K(?:\^?-?1|⁻¹)?)?"
        r"|eV|meV|GPa|MPa|Pa|nm|μm|um|Å|A|%"
    )
    return bool(re.search(rf"\b\d+(?:\.\d+)?\s*(?:{units})\b", text, re.IGNORECASE))


def _safe_non_numeric(values: list[str]) -> list[str]:
    return [value for value in values if not _contains_physical_value(value)]


def evaluate_visual_fact_gate(
    description: Any, metadata: dict[str, Any]
) -> dict[str, Any]:
    """返回可持久化的门禁结果；不信任模型自报的验证字段或置信度。"""
    legend = [str(value) for value in getattr(description, "legend", [])]
    trends = [str(value) for value in getattr(description, "trends", [])]
    key_values = [str(value) for value in getattr(description, "key_values", [])]
    summary = str(getattr(description, "visual_summary", "") or "")
    readability = _readability(metadata)

    multi_series = len(legend) > 1
    legend_source = _verification_source(metadata, "legend_verification_source")
    legend_verified = bool(metadata.get("legend_mapping_verified")) and (
        legend_source in TRUSTED_VERIFICATION_SOURCES
    )
    legend_status = (
        "not_applicable" if not multi_series else "verified" if legend_verified else "blocked"
    )

    numeric_source = _verification_source(metadata, "numeric_verification_source")
    verified_values = [str(value) for value in metadata.get("verified_numeric_facts") or []]
    numeric_verified = (
        bool(metadata.get("numeric_values_verified"))
        and numeric_source in TRUSTED_VERIFICATION_SOURCES
        and bool(key_values)
        and set(key_values).issubset(set(verified_values))
        and readability == "readable"
        and legend_status != "blocked"
    )
    numeric_status = (
        "not_applicable" if not key_values else "verified" if numeric_verified else "blocked"
    )

    # 多序列图未核验图例时，不索引模型生成的趋势和摘要；它们最容易携带错误
    # 的颜色/方向绑定。数值未核验时，仅保留不带物理量读数的安全文本。
    safe_trends = trends if legend_status != "blocked" else []
    safe_summary = summary if legend_status != "blocked" else ""
    if numeric_status == "blocked":
        safe_trends = _safe_non_numeric(safe_trends)
        if _contains_physical_value(safe_summary):
            safe_summary = ""

    reasons: list[str] = []
    if readability != "readable":
        reasons.append(f"readability_{readability}")
    if legend_status == "blocked":
        reasons.append("legend_mapping_unverified")
    if numeric_status == "blocked":
        reasons.append("numeric_values_unverified")
    return {
        "readability_status": readability,
        "legend_mapping_status": legend_status,
        "numeric_fact_status": numeric_status,
        "verified_key_values": key_values if numeric_status == "verified" else [],
        "safe_trends": safe_trends,
        "safe_visual_summary": safe_summary,
        "fact_gate_reasons": reasons,
    }


def requires_precise_numeric_answer(query: str) -> bool:
    """识别需要从图中读取精确或近似数值的请求。"""
    return bool(
        re.search(
            r"多少|分别(?:是|为)|数值|读数|精确|近似值|约为|峰值(?:是|为|多少)|"
            r"how\s+much|what\s+(?:is|are)\s+the\s+value",
            query,
            flags=re.IGNORECASE,
        )
    )


def numeric_answer_allowed(sources: list[dict[str, Any]]) -> bool:
    return any(
        source.get("numeric_fact_status") == "verified"
        and source.get("legend_mapping_status") != "blocked"
        for source in sources
    )


def should_block_numeric_answer(query: str, sources: list[dict[str, Any]]) -> bool:
    return requires_precise_numeric_answer(query) and not numeric_answer_allowed(sources)
