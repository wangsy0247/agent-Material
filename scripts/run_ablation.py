#!/usr/bin/env python3
"""基于 gold chunk id 的可复现检索消融实验。

正式实验只使用 ``gold_status=auto_verified`` 的可回答样本计算
严格检索指标；``needs_review`` 和 unanswerable 不混入分母。单路消融和
父块回溯都使用生产配置开关，不做 monkeypatch。
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import random
import statistics
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "packages"))
sys.path.insert(0, str(ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from agent_material.rag_engine.rag_service import RagService  # noqa: E402
from agent_material.utils.config_handler import vector_store_config  # noqa: E402
from eval_reproducibility import collect, repo_relative_path  # noqa: E402

DEFAULT_DATASET = ROOT / "evaluation_data/fixed_eval_dataset_v2.json"
DEFAULT_OUTPUT = ROOT / "evaluation_data/ablation_results_v2.json"
SEED = 20260820
LOG_DIR = ROOT / "logs"
log = logging.getLogger("ablation_v2")

PROFILES = [
    {
        "name": "dense_only",
        "description": "Dense only",
        "flags": {
            "enable_vector_search": True,
            "enable_bm25_search": False,
            "global_rerank": False,
            "parent_backtrack": False,
        },
        "enable_rewrite": False,
    },
    {
        "name": "bm25_only",
        "description": "BM25 only",
        "flags": {
            "enable_vector_search": False,
            "enable_bm25_search": True,
            "global_rerank": False,
            "parent_backtrack": False,
        },
        "enable_rewrite": False,
    },
    {
        "name": "hybrid_rrf",
        "description": "Dense + BM25 + weighted RRF",
        "flags": {
            "enable_vector_search": True,
            "enable_bm25_search": True,
            "global_rerank": False,
            "parent_backtrack": False,
        },
        "enable_rewrite": False,
    },
    {
        "name": "hybrid_rerank",
        "description": "Hybrid + global rerank",
        "flags": {
            "enable_vector_search": True,
            "enable_bm25_search": True,
            "global_rerank": True,
            "parent_backtrack": False,
        },
        "enable_rewrite": False,
    },
    {
        "name": "hybrid_rerank_parent",
        "description": "Hybrid + global rerank + parent backtrack",
        "flags": {
            "enable_vector_search": True,
            "enable_bm25_search": True,
            "global_rerank": True,
            "parent_backtrack": True,
        },
        "enable_rewrite": False,
    },
    {
        "name": "full_rewrite",
        "description": "Hybrid + rerank + parent + step-back/expand",
        "flags": {
            "enable_vector_search": True,
            "enable_bm25_search": True,
            "global_rerank": True,
            "parent_backtrack": True,
        },
        "enable_rewrite": True,
        "rewrite_strategy": "step_back+expand",
    },
]


def setup_logging() -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / f"ablation_v2_{datetime.now().strftime('%Y%m%d_%H%M')}.log"
    log.setLevel(logging.INFO)
    log.propagate = False
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    handlers = (logging.StreamHandler(), logging.FileHandler(path, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)
        log.addHandler(handler)
    return path


def _doc_id(doc) -> str:
    return str(doc.metadata.get("chunk_id") or "")


def _parent_id(doc) -> str:
    return str(doc.metadata.get("parent_id") or "")


def _metric_at(retrieved: list[str], gold: set[str], k: int) -> dict[str, float]:
    ranked = retrieved[:k]
    relevant = [1 if item in gold else 0 for item in ranked]
    hit = float(any(relevant))
    recall = len(set(ranked) & gold) / len(gold) if gold else 0.0
    reciprocal_rank = next(
        (1.0 / rank for rank, value in enumerate(relevant, start=1) if value), 0.0
    )
    dcg = sum(value / math.log2(rank + 1) for rank, value in enumerate(relevant, 1))
    ideal_count = min(len(gold), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return {
        "hit": hit,
        "recall": recall,
        "rr": reciprocal_rank,
        "ndcg": dcg / idcg if idcg else 0.0,
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * percentile) - 1))
    return round(ordered[index], 2)


@contextmanager
def profile_config(profile: dict):
    flags = vector_store_config.setdefault("feature_flags", {})
    backup = copy.deepcopy(flags)
    flags.update(profile["flags"])
    flags["cross_query_rrf"] = True
    flags["active_version_filter"] = True
    try:
        yield
    finally:
        flags.clear()
        flags.update(backup)


def _snapshot(profile: dict) -> dict:
    flags = vector_store_config.get("feature_flags", {})
    return {
        "description": profile["description"],
        "collection_prefix": vector_store_config.get("character_name"),
        "k": vector_store_config.get("k"),
        "score_threshold": vector_store_config.get("score_threshold"),
        "vector_weight": vector_store_config.get("vector_weight"),
        "bm25_weight": vector_store_config.get("bm25_weight"),
        "reranker": copy.deepcopy(vector_store_config.get("reranker", {})),
        "feature_flags": copy.deepcopy(flags),
        "enable_rewrite": profile["enable_rewrite"],
        "rewrite_strategy": profile.get("rewrite_strategy", ""),
    }


def run_profile(service: RagService, profile: dict, dataset: list[dict]) -> dict:
    details = []
    failures = []
    with profile_config(profile):
        snapshot = _snapshot(profile)
        for index, sample in enumerate(dataset):
            started = time.perf_counter()
            try:
                ctx = service.prepare(
                    sample["question"],
                    enable_rewrite=profile["enable_rewrite"],
                    rewrite_strategy=profile.get("rewrite_strategy", ""),
                )
            except Exception as error:
                failures.append(
                    {
                        "index": index,
                        "question": sample["question"],
                        "error": str(error)[:500],
                    }
                )
                continue
            latency_ms = (time.perf_counter() - started) * 1000
            gold_chunks = {
                str(evidence.get("chunk_id"))
                for evidence in sample.get("gold_evidence", [])
                if evidence.get("chunk_id")
            }
            gold_parents = {
                str(evidence.get("parent_id"))
                for evidence in sample.get("gold_evidence", [])
                if evidence.get("parent_id")
            }
            candidate_ids = [_doc_id(doc) for doc in ctx.fused_children if _doc_id(doc)]
            final_ids = [
                _doc_id(doc) for doc in ctx.reranked_children if _doc_id(doc)
            ]
            resolved_parent_ids = [
                _parent_id(doc) for doc in ctx.resolved_parents if _parent_id(doc)
            ]
            eligible = bool(
                sample.get("gold_verification", {}).get("strict_metric_eligible")
            )
            metrics = None
            if eligible:
                candidate_5 = _metric_at(candidate_ids, gold_chunks, 5)
                candidate_10 = _metric_at(candidate_ids, gold_chunks, 10)
                final_1 = _metric_at(final_ids, gold_chunks, 1)
                final_3 = _metric_at(final_ids, gold_chunks, 3)
                parent_3 = _metric_at(resolved_parent_ids, gold_parents, 3)
                metrics = {
                    "candidate_hit_at_5": candidate_5["hit"],
                    "candidate_recall_at_5": candidate_5["recall"],
                    "candidate_hit_at_10": candidate_10["hit"],
                    "candidate_recall_at_10": candidate_10["recall"],
                    "candidate_mrr_at_10": candidate_10["rr"],
                    "candidate_ndcg_at_10": candidate_10["ndcg"],
                    "final_hit_at_1": final_1["hit"],
                    "final_hit_at_3": final_3["hit"],
                    "final_recall_at_3": final_3["recall"],
                    "final_mrr_at_3": final_3["rr"],
                    "final_ndcg_at_3": final_3["ndcg"],
                    "parent_hit_at_3": parent_3["hit"],
                }
            details.append(
                {
                    "index": index,
                    "question_type": sample.get("question_type"),
                    "gold_status": sample.get("gold_status"),
                    "strict_metric_eligible": eligible,
                    "gold_chunk_ids": sorted(gold_chunks),
                    "candidate_chunk_ids": candidate_ids[:10],
                    "final_chunk_ids": final_ids[:3],
                    "resolved_parent_ids": resolved_parent_ids[:3],
                    "metrics": metrics,
                    "latency_ms": round(latency_ms, 2),
                    "stage_durations": dict(ctx.stage_durations),
                    "degradation_events": ctx.degradation_events,
                }
            )
            if (index + 1) % 10 == 0:
                log.info("[%s] %s/%s", profile["name"], index + 1, len(dataset))

    strict = [detail for detail in details if detail["strict_metric_eligible"]]
    latencies = [detail["latency_ms"] for detail in details]

    def average(key: str) -> float | None:
        values = [detail["metrics"][key] for detail in strict if detail["metrics"]]
        return round(statistics.fmean(values), 4) if values else None

    unanswerable = [
        detail for detail in details if detail["question_type"] == "unanswerable"
    ]
    summary = {
        "questions_total": len(dataset),
        "questions_succeeded": len(details),
        "questions_failed": len(failures),
        "strict_gold_questions": len(strict),
        "candidate_hit_at_5": average("candidate_hit_at_5"),
        "candidate_recall_at_5": average("candidate_recall_at_5"),
        "candidate_hit_at_10": average("candidate_hit_at_10"),
        "candidate_recall_at_10": average("candidate_recall_at_10"),
        "candidate_mrr_at_10": average("candidate_mrr_at_10"),
        "candidate_ndcg_at_10": average("candidate_ndcg_at_10"),
        "final_hit_at_1": average("final_hit_at_1"),
        "final_hit_at_3": average("final_hit_at_3"),
        "final_recall_at_3": average("final_recall_at_3"),
        "final_mrr_at_3": average("final_mrr_at_3"),
        "final_ndcg_at_3": average("final_ndcg_at_3"),
        "parent_hit_at_3": average("parent_hit_at_3"),
        "unanswerable_nonempty_retrieval_rate": (
            round(
                statistics.fmean(
                    bool(detail["candidate_chunk_ids"]) for detail in unanswerable
                ),
                4,
            )
            if unanswerable
            else None
        ),
        "latency_p50_ms": _percentile(latencies, 0.50),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "degradation_events_total": sum(
            len(detail["degradation_events"]) for detail in details
        ),
    }
    return {
        "config_snapshot": snapshot,
        "summary": summary,
        "failures": failures,
        "per_question": details,
    }


def _save(results: dict, output: Path, partial: bool) -> None:
    results["partial"] = partial
    results["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _names(value: str) -> set[str]:
    return {name.strip() for name in value.split(",") if name.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--only", default="")
    parser.add_argument("--skip", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    setup_logging()
    random.seed(SEED)
    dataset_path = args.dataset.resolve()
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if args.limit:
        dataset = dataset[: args.limit]
    only = _names(args.only)
    skip = _names(args.skip)
    profiles = [
        profile
        for profile in PROFILES
        if (not only or profile["name"] in only) and profile["name"] not in skip
    ]
    if not profiles:
        raise SystemExit("没有可运行的消融组")

    collection_backup = vector_store_config.get("character_name")
    routing = vector_store_config.setdefault("query_routing", {})
    routing_backup = copy.deepcopy(routing)
    vector_store_config["character_name"] = "agent_eval"
    routing["enabled"] = False
    artifacts = [
        ROOT / "evaluation_data/index_manifest_v2.json",
        ROOT / "evaluation_data/index_audit_v2.json",
    ]
    results = {
        "schema_version": 2,
        "metric_policy": {
            "strict_gold_status": "auto_verified",
            "candidate_ranking": "fused_children",
            "final_ranking": "reranked_children (production top_n)",
            "gold_identity": "exact chunk_id",
            "unanswerable_excluded_from_retrieval_quality": True,
        },
        "dataset": repo_relative_path(dataset_path, ROOT),
        "num_questions": len(dataset),
        "profiles": PROFILES,
        "reproducibility": collect(
            root=ROOT,
            dataset=dataset_path,
            config=vector_store_config,
            seed=SEED,
            artifacts=artifacts,
        ),
        "configurations": {},
    }
    output = args.output.resolve()
    if args.resume and output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        results["configurations"].update(previous.get("configurations", {}))

    try:
        service = RagService()
        service._get_pipeline()
        for profile in profiles:
            if args.resume and profile["name"] in results["configurations"]:
                log.info("跳过已完成组: %s", profile["name"])
                continue
            started = time.perf_counter()
            log.info("开始: %s", profile["name"])
            results["configurations"][profile["name"]] = run_profile(
                service, profile, dataset
            )
            elapsed = time.perf_counter() - started
            summary = results["configurations"][profile["name"]]["summary"]
            log.info(
                "完成: %s %.1fs | final Hit@3=%s candidate Recall@10=%s",
                profile["name"],
                elapsed,
                summary["final_hit_at_3"],
                summary["candidate_recall_at_10"],
            )
            _save(results, output, partial=True)
    finally:
        vector_store_config["character_name"] = collection_backup
        routing.clear()
        routing.update(routing_backup)

    ordered = {}
    for profile in PROFILES:
        if profile["name"] in results["configurations"]:
            ordered[profile["name"]] = results["configurations"][profile["name"]]
    results["configurations"] = ordered
    _save(results, output, partial=False)
    log.info("结果: %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
