#!/usr/bin/env python3
"""运行或评分 P1 视觉 RAG 评测集，并生成脱敏 JSON/Markdown 报告。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ROOT / "backend" / "packages"
if str(PACKAGES) not in sys.path:
    sys.path.insert(0, str(PACKAGES))

from agent_material.rag_engine.evaluation.visual_rag import (
    VisualEvalCase,
    VisualPrediction,
    evaluate_visual_predictions,
    render_visual_report,
    run_visual_cases,
)

def _items(path: Path, key: str) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    values = payload.get(key)
    if not isinstance(values, list):
        raise ValueError(f"{path.name} 缺少列表字段 {key}")
    return values


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--run-live", action="store_true")
    parser.add_argument("--include-answers", action="store_true")
    parser.add_argument("--allow-external-images", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    cases = [VisualEvalCase.model_validate(item) for item in _items(args.dataset, "cases")]
    if args.run_live == bool(args.predictions):
        parser.error("必须且只能选择 --run-live 或 --predictions")

    if args.run_live:
        from agent_material.rag_engine.multimodal_answer import (
            multimodal_answer_enabled,
        )
        from agent_material.rag_engine.rag_service import RagService

        if (
            args.include_answers
            and multimodal_answer_enabled()
            and not args.allow_external_images
        ):
            parser.error(
                "在线多模态已启用；外发图片前必须显式添加 --allow-external-images"
            )
        predictions = run_visual_cases(
            cases,
            service=RagService(),
            include_answers=args.include_answers,
        )
    else:
        predictions = [
            VisualPrediction.model_validate(item)
            for item in _items(args.predictions, "predictions")
        ]

    result = evaluate_visual_predictions(cases, predictions)
    result["dataset"] = args.dataset.name
    result["prediction_count"] = len(predictions)
    _write_json(args.output, result)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_visual_report(result) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
