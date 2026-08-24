#!/usr/bin/env python3
"""为评测集生成可追溯的 gold evidence 候选并分级。

这是一个确定性的自动标注工具，不会把自动结果冒充为人工审核。
它使用参考答案/问题中的英文、数字和中文 n-gram，结合语料库 IDF
找到证据块；低覆盖率样本进入 review queue，不参与严格检索指标。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "packages"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from agent_material.rag_engine.db.pg.schema import normalize_conn_str  # noqa: E402
from agent_material.utils.config_handler import vector_store_config  # noqa: E402

ASCII_TOKEN = re.compile(r"[a-zA-Z][a-zA-Z0-9_.+/-]*|\d+(?:\.\d+)?(?:e[+-]?\d+)?")
CHINESE_RUN = re.compile(r"[\u4e00-\u9fff]+")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tokens(text: str) -> list[str]:
    normalized = text.lower()
    tokens = [token for token in ASCII_TOKEN.findall(normalized) if len(token) > 1]
    for run in CHINESE_RUN.findall(normalized):
        tokens.extend(run[i : i + 2] for i in range(max(0, len(run) - 1)))
        tokens.extend(run[i : i + 3] for i in range(max(0, len(run) - 2)))
    return tokens


def _weighted_coverage(target: set[str], found: set[str], idf: dict[str, float]) -> float:
    denominator = sum(idf.get(token, 1.0) for token in target)
    if denominator == 0:
        return 0.0
    return sum(idf.get(token, 1.0) for token in target & found) / denominator


def _load_chunks(conn_str: str, collection: str) -> list[dict]:
    with psycopg.connect(conn_str) as conn:
        rows = conn.execute(
            """
            SELECT e.document, e.cmetadata
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON c.uuid = e.collection_id
            WHERE c.name = %s AND e.cmetadata->>'is_active' = 'true'
            ORDER BY e.cmetadata->>'chunk_id'
            """,
            (collection,),
        ).fetchall()
    chunks = []
    for passage, metadata in rows:
        token_counts = Counter(_tokens(passage or ""))
        chunks.append(
            {
                "passage": passage or "",
                "metadata": metadata or {},
                "token_counts": token_counts,
                "token_set": set(token_counts),
            }
        )
    if not chunks:
        raise RuntimeError(f"评测 collection 为空: {collection}")
    return chunks


def _idf(chunks: list[dict]) -> dict[str, float]:
    document_frequency: Counter[str] = Counter()
    for chunk in chunks:
        document_frequency.update(chunk["token_set"])
    total = len(chunks)
    return {
        token: math.log((total + 1) / (frequency + 1)) + 1.0
        for token, frequency in document_frequency.items()
    }


def _rank_chunks(
    question: str,
    ground_truth: str,
    chunks: list[dict],
    idf: dict[str, float],
) -> list[dict]:
    truth_tokens = set(_tokens(ground_truth))
    question_tokens = set(_tokens(question))
    ranked = []
    for chunk in chunks:
        found = chunk["token_set"]
        truth_coverage = _weighted_coverage(truth_tokens, found, idf)
        question_coverage = _weighted_coverage(question_tokens, found, idf)
        matched_truth = truth_tokens & found
        anchor_score = sum(
            idf.get(token, 1.0)
            for token in matched_truth
            if idf.get(token, 1.0) >= 4.0
            or any(char.isdigit() for char in token)
            or bool(re.search(r"[a-z]", token))
        )
        length_penalty = 1.0 / (1.0 + max(0, len(chunk["passage"]) - 1600) / 3200)
        score = (0.72 * truth_coverage + 0.23 * question_coverage) * length_penalty
        score += min(anchor_score, 20.0) * 0.0025
        ranked.append(
            {
                **chunk,
                "score": score,
                "truth_coverage": truth_coverage,
                "question_coverage": question_coverage,
                "anchor_score": anchor_score,
            }
        )
    return sorted(
        ranked,
        key=lambda item: (
            -item["score"],
            str(item["metadata"].get("chunk_id", "")),
        ),
    )


def _select_evidence(ranked: list[dict], truth_tokens: set[str], idf: dict[str, float]):
    selected = []
    covered: set[str] = set()
    seen_parents: set[str] = set()
    for item in ranked[:30]:
        parent_id = str(item["metadata"].get("parent_id", ""))
        contribution = (truth_tokens & item["token_set"]) - covered
        if selected and not contribution:
            continue
        if parent_id in seen_parents and len(selected) >= 1:
            continue
        selected.append(item)
        seen_parents.add(parent_id)
        covered.update(truth_tokens & item["token_set"])
        if _weighted_coverage(truth_tokens, covered, idf) >= 0.72 or len(selected) == 3:
            break
    coverage = _weighted_coverage(truth_tokens, covered, idf)
    return selected, coverage


def build(
    dataset_path: Path,
    output_path: Path,
    review_path: Path,
    collection: str,
    minimum_coverage: float,
) -> dict:
    conn_str = normalize_conn_str(vector_store_config["connection_string"])
    chunks = _load_chunks(conn_str, collection)
    idf = _idf(chunks)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    review_queue = []
    status_counts: Counter[str] = Counter()

    for index, sample in enumerate(dataset):
        if sample.get("question_type") == "unanswerable":
            sample["gold_evidence"] = []
            sample["gold_status"] = "not_applicable"
            sample["gold_verification"] = {
                "method": "dataset_label",
                "answerable": False,
                "strict_metric_eligible": False,
            }
            status_counts["not_applicable"] += 1
            continue

        ranked = _rank_chunks(
            sample["question"], sample["ground_truth"], chunks, idf
        )
        truth_tokens = set(_tokens(sample["ground_truth"]))
        selected, coverage = _select_evidence(ranked, truth_tokens, idf)
        anchor_score = sum(item["anchor_score"] for item in selected)
        verified = coverage >= minimum_coverage and anchor_score >= 2.0
        status = "auto_verified" if verified else "needs_review"
        status_counts[status] += 1
        sample["gold_evidence"] = [
            {
                "chunk_id": item["metadata"].get("chunk_id"),
                "parent_id": item["metadata"].get("parent_id"),
                "document_id": item["metadata"].get("document_id"),
                "version_id": item["metadata"].get("version_id"),
                "source": item["metadata"].get("source"),
                "passage": item["passage"],
                "lexical_score": round(item["score"], 6),
            }
            for item in selected
        ]
        sample["gold_status"] = status
        sample["gold_verification"] = {
            "method": "deterministic_reference_lexical_alignment_v1",
            "answerable": True,
            "weighted_reference_coverage": round(coverage, 6),
            "anchor_score": round(anchor_score, 6),
            "strict_metric_eligible": verified,
            "human_verified": False,
        }
        if not verified:
            review_queue.append(
                {
                    "sample_index": index,
                    "question": sample["question"],
                    "ground_truth": sample["ground_truth"],
                    "coverage": round(coverage, 6),
                    "anchor_score": round(anchor_score, 6),
                    "top_candidates": [
                        {
                            "chunk_id": item["metadata"].get("chunk_id"),
                            "parent_id": item["metadata"].get("parent_id"),
                            "passage": item["passage"],
                            "score": round(item["score"], 6),
                        }
                        for item in ranked[:5]
                    ],
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(
        json.dumps(review_queue, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "input_dataset": str(dataset_path),
        "input_sha256": _sha256(dataset_path),
        "output_dataset": str(output_path),
        "output_sha256": _sha256(output_path),
        "collection": collection,
        "corpus_chunks": len(chunks),
        "minimum_coverage": minimum_coverage,
        "status_counts": dict(status_counts),
        "review_queue": str(review_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "evaluation_data/fixed_eval_dataset.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evaluation_data/fixed_eval_dataset_v2.json",
    )
    parser.add_argument(
        "--review-output",
        type=Path,
        default=ROOT / "evaluation_data/gold_review_queue_v2.json",
    )
    parser.add_argument("--collection", default="agent_eval_small")
    parser.add_argument("--minimum-coverage", type=float, default=0.58)
    args = parser.parse_args()
    result = build(
        args.dataset.resolve(),
        args.output.resolve(),
        args.review_output.resolve(),
        args.collection,
        args.minimum_coverage,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
