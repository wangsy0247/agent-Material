"""P1 图表 RAG 的确定性评测口径与运行器。"""

from __future__ import annotations

import math
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class VisualEvalCase(BaseModel):
    """一条经过审核的图表问题与 gold artifact。"""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    gold_artifact_ids: list[str] = Field(min_length=1)
    required_answer_terms: list[list[str]] = Field(default_factory=list)
    gold_status: Literal["verified", "needs_review"] = "needs_review"

    @field_validator("gold_artifact_ids")
    @classmethod
    def _unique_gold_ids(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values if value.strip()]
        if not cleaned:
            raise ValueError("gold_artifact_ids 不能为空")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("gold_artifact_ids 不能重复")
        return cleaned

    @field_validator("required_answer_terms")
    @classmethod
    def _valid_term_groups(cls, values: list[list[str]]) -> list[list[str]]:
        result = []
        for group in values:
            cleaned = [term.strip() for term in group if term.strip()]
            if not cleaned:
                raise ValueError("答案事实组必须至少包含一个可接受术语")
            result.append(cleaned)
        return result


class VisualPrediction(BaseModel):
    """系统输出；不包含 gold，防止评测输入与预测结果混写。"""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    retrieved_artifact_ids: list[str] = Field(default_factory=list)
    retrieved_visual_statuses: list[str] = Field(default_factory=list)
    answer: str | None = None
    answer_mode: str = "retrieval_only"
    latency_ms: float = Field(default=0.0, ge=0.0)
    degradation_events: list[dict[str, Any]] = Field(default_factory=list)


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 2)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 2)
    fraction = position - lower
    return round(
        ordered[lower] + (ordered[upper] - ordered[lower]) * fraction,
        2,
    )


def _answer_facts(answer: str | None, groups: list[list[str]]) -> tuple[int, int]:
    if answer is None or not groups:
        return 0, 0
    normalized = answer.casefold()
    matched = sum(
        any(term.casefold() in normalized for term in alternatives)
        for alternatives in groups
    )
    return matched, len(groups)


def evaluate_visual_predictions(
    cases: list[VisualEvalCase],
    predictions: list[VisualPrediction],
) -> dict[str, Any]:
    """按 verified gold 计算图表召回、排序、事实答案和降级可靠性。"""
    prediction_by_id = {item.question_id: item for item in predictions}
    verified = [case for case in cases if case.gold_status == "verified"]
    details: list[dict[str, Any]] = []
    hit_1 = hit_3 = 0
    reciprocal_ranks: list[float] = []
    evidence_recalls: list[float] = []
    fact_matched = fact_total = answer_questions = 0
    statuses: list[str] = []
    latencies: list[float] = []
    multimodal_attempts = multimodal_successes = 0
    fallback_attempts = fallback_successes = 0

    for case in verified:
        prediction = prediction_by_id.get(case.question_id)
        retrieved = _deduplicate(
            prediction.retrieved_artifact_ids if prediction else []
        )
        gold = set(case.gold_artifact_ids)
        first_rank = next(
            (rank for rank, artifact_id in enumerate(retrieved, 1) if artifact_id in gold),
            None,
        )
        hit_1 += int(first_rank == 1)
        hit_3 += int(first_rank is not None and first_rank <= 3)
        reciprocal_ranks.append(1 / first_rank if first_rank and first_rank <= 3 else 0.0)
        evidence_recalls.append(len(gold.intersection(retrieved[:3])) / len(gold))

        matched, total = _answer_facts(
            prediction.answer if prediction else None,
            case.required_answer_terms,
        )
        if total:
            answer_questions += 1
            fact_matched += matched
            fact_total += total
        if prediction:
            statuses.extend(prediction.retrieved_visual_statuses)
            latencies.append(prediction.latency_ms)
            if prediction.answer_mode in {"multimodal", "text_rag_fallback"}:
                multimodal_attempts += 1
                multimodal_successes += int(prediction.answer_mode == "multimodal")
            if prediction.answer_mode == "text_rag_fallback":
                fallback_attempts += 1
                fallback_successes += int(bool((prediction.answer or "").strip()))
        details.append(
            {
                "question_id": case.question_id,
                "first_gold_rank": first_rank,
                "hit_at_3": bool(first_rank and first_rank <= 3),
                "gold_count": len(gold),
                "retrieved_count": len(retrieved),
                "answer_facts_matched": matched,
                "answer_facts_total": total,
                "answer_mode": prediction.answer_mode if prediction else "missing",
            }
        )

    denominator = len(verified)
    completed = sum(status == "completed" for status in statuses)
    summary = {
        "verified_questions": denominator,
        "questions_missing_prediction": sum(
            case.question_id not in prediction_by_id for case in verified
        ),
        "artifact_hit_at_1": hit_1 / denominator if denominator else None,
        "artifact_hit_at_3": hit_3 / denominator if denominator else None,
        "artifact_mrr_at_3": (
            sum(reciprocal_ranks) / denominator if denominator else None
        ),
        "artifact_evidence_recall_at_3": (
            sum(evidence_recalls) / denominator if denominator else None
        ),
        "answer_fact_accuracy": fact_matched / fact_total if fact_total else None,
        "answer_questions_scored": answer_questions,
        "visual_description_completion_rate": (
            completed / len(statuses) if statuses else None
        ),
        "multimodal_success_rate": (
            multimodal_successes / multimodal_attempts
            if multimodal_attempts
            else None
        ),
        "text_fallback_success_rate": (
            fallback_successes / fallback_attempts if fallback_attempts else None
        ),
        "latency_p50_ms": _percentile(latencies, 0.5),
        "latency_p95_ms": _percentile(latencies, 0.95),
    }
    return {
        "schema_version": "visual-rag-eval-v1",
        "summary": summary,
        "details": details,
    }


def run_visual_cases(
    cases: list[VisualEvalCase],
    *,
    service: Any,
    include_answers: bool = False,
) -> list[VisualPrediction]:
    """通过统一 RagService 入口运行评测，不复制检索逻辑。"""
    predictions = []
    for case in cases:
        started = time.perf_counter()
        ctx = service.prepare(case.question, False, "")
        sources = list(ctx.sources)
        answer = "".join(service.stream_answer(ctx)) if include_answers else None
        predictions.append(
            VisualPrediction(
                question_id=case.question_id,
                retrieved_artifact_ids=_deduplicate(
                    [str(source.get("artifact_id") or "") for source in sources]
                ),
                retrieved_visual_statuses=[
                    str(source.get("visual_status") or "not_requested")
                    for source in sources
                    if source.get("content_type") == "figure"
                ],
                answer=answer,
                answer_mode=(ctx.answer_mode or "retrieval_only"),
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                degradation_events=list(ctx.degradation_events),
            )
        )
    return predictions


def render_visual_report(result: dict[str, Any]) -> str:
    """生成不包含问题正文、答案正文和本机路径的 Markdown 报告。"""
    summary = result["summary"]

    def value(name: str) -> str:
        raw = summary.get(name)
        return "N/A" if raw is None else f"{raw:.4f}"

    return "\n".join(
        [
            "# P1 视觉 RAG 评测报告",
            "",
            "> 口径：仅统计 `gold_status=verified`；相关性以 `artifact_id` 精确匹配。",
            "",
            "## 核心指标",
            "",
            f"- Verified questions：{summary['verified_questions']}",
            f"- Artifact Hit@1：{value('artifact_hit_at_1')}",
            f"- Artifact Hit@3：{value('artifact_hit_at_3')}",
            f"- Artifact MRR@3：{value('artifact_mrr_at_3')}",
            f"- Artifact evidence Recall@3：{value('artifact_evidence_recall_at_3')}",
            f"- Answer fact accuracy：{value('answer_fact_accuracy')}",
            f"- Visual description completion：{value('visual_description_completion_rate')}",
            f"- Multimodal success：{value('multimodal_success_rate')}",
            f"- Text fallback success：{value('text_fallback_success_rate')}",
            f"- Latency p50 / p95：{summary['latency_p50_ms']} / "
            f"{summary['latency_p95_ms']} ms",
            "",
            "## 解释边界",
            "",
            "- 答案事实得分采用人工审核的可接受术语组，不是 LLM-as-a-judge。",
            "- `needs_review` 样本不进入分母；缺少预测按未命中处理。",
            "- 报告不保存问题、回答、主机路径、连接串或密钥。",
            "- 在真实图片外发未获授权时，应只运行 retrieval-only 评测。",
        ]
    )
