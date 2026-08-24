#!/usr/bin/env python3
"""在固定 RAG 管线上扫描 Vector/BM25 Weighted-RRF 权重。"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ROOT / "backend/packages"
for path in (ROOT, PACKAGES):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent_material.rag_engine.rag_service import RagService  # noqa: E402
from agent_material.utils.config_handler import vector_store_config  # noqa: E402
from eval_reproducibility import collect, repo_relative_path  # noqa: E402
from run_ablation import run_profile, setup_logging  # noqa: E402

DEFAULT_DATASET = ROOT / "evaluation_data/fixed_eval_dataset_v2.json"
DEFAULT_OUTPUT = ROOT / "evaluation_data/weight_sweep_results_v1.json"
DEFAULT_VECTOR_WEIGHTS = "0.5,0.6,0.7,0.8,0.9,1.0"
SEED = 20260820


def _parse_weights(value: str) -> list[float]:
    weights: list[float] = []
    for raw in value.split(","):
        weight = round(float(raw.strip()), 4)
        if not 0.0 <= weight <= 1.0:
            raise argparse.ArgumentTypeError("权重必须位于 [0, 1]")
        if weight not in weights:
            weights.append(weight)
    if not weights:
        raise argparse.ArgumentTypeError("至少提供一个 Vector 权重")
    return weights


def _profile(vector_weight: float) -> dict:
    bm25_weight = round(1.0 - vector_weight, 4)
    vector_label = round(vector_weight * 100)
    bm25_label = round(bm25_weight * 100)
    return {
        "name": f"vector_{vector_label:03d}_bm25_{bm25_label:03d}",
        "description": (
            "Hybrid + weighted RRF + global rerank + parent backtrack | "
            f"vector={vector_weight:.2f}, bm25={bm25_weight:.2f}"
        ),
        "flags": {
            "enable_vector_search": True,
            "enable_bm25_search": True,
            "global_rerank": True,
            "parent_backtrack": True,
        },
        "enable_rewrite": False,
        "vector_weight": vector_weight,
        "bm25_weight": bm25_weight,
    }


@contextmanager
def _weight_config(profile: dict):
    vector_backup = vector_store_config.get("vector_weight")
    bm25_backup = vector_store_config.get("bm25_weight")
    vector_store_config["vector_weight"] = profile["vector_weight"]
    vector_store_config["bm25_weight"] = profile["bm25_weight"]
    try:
        yield
    finally:
        vector_store_config["vector_weight"] = vector_backup
        vector_store_config["bm25_weight"] = bm25_backup


def _save(results: dict, output: Path, *, partial: bool) -> None:
    results["partial"] = partial
    results["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--vector-weights", default=DEFAULT_VECTOR_WEIGHTS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    random.seed(SEED)
    vector_weights = _parse_weights(args.vector_weights)
    profiles = [_profile(weight) for weight in vector_weights]
    dataset_path = args.dataset.resolve()
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if args.limit:
        dataset = dataset[: args.limit]

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
        "schema_version": 1,
        "experiment": "weighted_rrf_vector_bm25_sweep",
        "dataset": repo_relative_path(dataset_path, ROOT),
        "num_questions": len(dataset),
        "fixed_conditions": {
            "collection_prefix": "agent_eval",
            "query_rewrite": False,
            "vector_search": True,
            "bm25_search": True,
            "global_rerank": True,
            "parent_backtrack": True,
            "weight_sum": 1.0,
            "note": "BM25 分支始终执行；权重 0 仅表示其 RRF 贡献为 0。",
        },
        "profiles": profiles,
        "reproducibility": collect(
            root=ROOT,
            dataset=dataset_path,
            config=vector_store_config,
            seed=SEED,
            artifacts=artifacts,
        ),
        "configurations": {},
    }
    setup_logging()
    output = args.output.resolve()
    if args.resume and output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        results["configurations"].update(previous.get("configurations", {}))

    try:
        service = RagService()
        service._get_pipeline()
        for profile in profiles:
            name = profile["name"]
            if args.resume and name in results["configurations"]:
                continue
            started = time.perf_counter()
            with _weight_config(profile):
                result = run_profile(service, profile, dataset)
            result["elapsed_seconds"] = round(time.perf_counter() - started, 2)
            results["configurations"][name] = result
            _save(results, output, partial=True)
    finally:
        vector_store_config["character_name"] = collection_backup
        routing.clear()
        routing.update(routing_backup)

    results["configurations"] = {
        profile["name"]: results["configurations"][profile["name"]]
        for profile in profiles
        if profile["name"] in results["configurations"]
    }
    _save(results, output, partial=False)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
