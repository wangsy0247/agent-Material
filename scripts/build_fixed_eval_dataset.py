#!/usr/bin/env python3
"""从公开分片重建固定评测集，并校验题型分层。"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "evaluation_data"
PART_DIR = EVAL_DIR / "dataset_parts"
PARTS = (
    "base_40.json",
    "terminology_formula.json",
    "summary_comparison.json",
    "multi_hop_unanswerable.json",
)
STRATA_MIN = {
    "factual": 15,
    "terminology": 10,
    "summary": 10,
    "comparison": 10,
    "multi_hop": 10,
    "formula_table": 10,
    "unanswerable": 10,
}


def main() -> int:
    dataset: list[dict] = []
    for name in PARTS:
        entries = json.loads((PART_DIR / name).read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            raise TypeError(f"{name} 必须是 JSON 数组")
        dataset.extend(entries)

    required = {"question", "ground_truth", "question_type"}
    questions: set[str] = set()
    for index, entry in enumerate(dataset):
        missing = required - set(entry)
        if missing:
            raise ValueError(f"第 {index} 条缺少字段: {sorted(missing)}")
        question = entry["question"].strip()
        if question in questions:
            raise ValueError(f"发现重复问题: {question}")
        questions.add(question)

    counts = Counter(entry["question_type"] for entry in dataset)
    unmet = {
        question_type: (counts.get(question_type, 0), minimum)
        for question_type, minimum in STRATA_MIN.items()
        if counts.get(question_type, 0) < minimum
    }
    if unmet:
        print(f"分层下限未满足: {unmet}", file=sys.stderr)
        return 1

    output = EVAL_DIR / "fixed_eval_dataset.json"
    output.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"已生成 {output.relative_to(ROOT).as_posix()}，共 {len(dataset)} 题")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
