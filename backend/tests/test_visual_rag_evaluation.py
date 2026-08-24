from types import SimpleNamespace

import pytest

from agent_material.rag_engine.evaluation.visual_rag import (
    VisualEvalCase,
    VisualPrediction,
    evaluate_visual_predictions,
    render_visual_report,
    run_visual_cases,
)


def _cases():
    return [
        VisualEvalCase(
            question_id="q1",
            question="图一的峰值和趋势是什么？",
            gold_artifact_ids=["figure-a"],
            required_answer_terms=[["峰值"], ["上升", "增加"]],
            gold_status="verified",
        ),
        VisualEvalCase(
            question_id="q2",
            question="图二比较了什么？",
            gold_artifact_ids=["figure-b"],
            required_answer_terms=[["方法甲"], ["方法乙"]],
            gold_status="verified",
        ),
        VisualEvalCase(
            question_id="q3",
            question="待审核问题",
            gold_artifact_ids=["figure-c"],
            gold_status="needs_review",
        ),
    ]


def test_visual_metrics_use_verified_exact_artifact_ids():
    predictions = [
        VisualPrediction(
            question_id="q1",
            retrieved_artifact_ids=["figure-a"],
            retrieved_visual_statuses=["completed"],
            answer="峰值之后继续上升。",
            answer_mode="multimodal",
            latency_ms=100,
        ),
        VisualPrediction(
            question_id="q2",
            retrieved_artifact_ids=["noise-1", "noise-2", "figure-b"],
            retrieved_visual_statuses=["failed"],
            answer="只提到了方法甲。",
            answer_mode="text_rag_fallback",
            latency_ms=300,
        ),
    ]

    result = evaluate_visual_predictions(_cases(), predictions)
    summary = result["summary"]

    assert summary["verified_questions"] == 2
    assert summary["artifact_hit_at_1"] == 0.5
    assert summary["artifact_hit_at_3"] == 1.0
    assert summary["artifact_mrr_at_3"] == pytest.approx(2 / 3)
    assert summary["answer_fact_accuracy"] == 0.75
    assert summary["visual_description_completion_rate"] == 0.5
    assert summary["multimodal_success_rate"] == 0.5
    assert summary["text_fallback_success_rate"] == 1.0
    assert summary["latency_p50_ms"] == 200
    assert summary["latency_p95_ms"] == 290
    assert [item["question_id"] for item in result["details"]] == ["q1", "q2"]


def test_visual_report_does_not_include_question_or_answer_text():
    result = evaluate_visual_predictions(
        _cases(),
        [
            VisualPrediction(
                question_id="q1",
                retrieved_artifact_ids=["figure-a"],
                answer="峰值之后上升。",
                answer_mode="multimodal",
            )
        ],
    )
    report = render_visual_report(result)

    assert "图一的峰值" not in report
    assert "峰值之后上升" not in report
    assert "Artifact Hit@3" in report


def test_live_runner_uses_unified_service_entry_and_frozen_sources():
    class FakeService:
        def __init__(self):
            self.calls = []

        def prepare(self, question, enable_rewrite, rewrite_strategy):
            self.calls.append((question, enable_rewrite, rewrite_strategy))
            return SimpleNamespace(
                sources=[
                    {
                        "content_type": "figure",
                        "artifact_id": "figure-a",
                        "visual_status": "completed",
                    }
                ],
                answer_mode="",
                degradation_events=[],
            )

        @staticmethod
        def stream_answer(ctx):
            ctx.answer_mode = "multimodal"
            yield "回答"

    service = FakeService()
    predictions = run_visual_cases(
        [_cases()[0]], service=service, include_answers=True
    )

    assert service.calls == [("图一的峰值和趋势是什么？", False, "")]
    assert predictions[0].retrieved_artifact_ids == ["figure-a"]
    assert predictions[0].answer == "回答"
    assert predictions[0].answer_mode == "multimodal"
