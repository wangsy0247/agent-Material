"""RAG 离线评测工具。"""

from .visual_rag import (
    VisualEvalCase,
    VisualPrediction,
    evaluate_visual_predictions,
    render_visual_report,
    run_visual_cases,
)

__all__ = [
    "VisualEvalCase",
    "VisualPrediction",
    "evaluate_visual_predictions",
    "render_visual_report",
    "run_visual_cases",
]
