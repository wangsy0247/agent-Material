"""Read-only BM25 smoke test against the configured active PostgreSQL corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "backend" / "packages"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from agent_material.rag_engine.bm25_retriever import build_material_bm25_retriever
from agent_material.rag_engine.vector_store import VectorStore
from agent_material.utils.config_handler import vector_store_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", action="append", default=[])
    args = parser.parse_args()
    queries = args.query or [
        "β-Ga₂O₃ 热导率随温度的变化",
        "deep neural network interatomic potential",
        "今天天气预报",
    ]

    store = VectorStore()
    started = time.perf_counter()
    multi_config = vector_store_config.get("multi_granularity", {})
    collection_name = (
        f"{vector_store_config.get('character_name', 'agent')}"
        f"{multi_config.get('small_collection_suffix', '_small')}"
        if multi_config.get("enabled", False)
        else vector_store_config.get("character_name", "agent")
    )
    documents = store._load_documents_from_collection(
        collection_name,
        active_only=bool(
            vector_store_config.get("feature_flags", {}).get(
                "active_version_filter", True
            )
        ),
        require_parent_id=bool(multi_config.get("enabled", False)),
    )
    load_seconds = time.perf_counter() - started

    started = time.perf_counter()
    retriever = build_material_bm25_retriever(
        documents,
        k=int(vector_store_config.get("k", 10)),
        config=vector_store_config,
    )
    build_seconds = time.perf_counter() - started

    report = {
        "documents": len(documents),
        "load_seconds": round(load_seconds, 3),
        "build_seconds": round(build_seconds, 3),
        "queries": [],
    }
    for query in queries:
        results = retriever.invoke(query)
        report["queries"].append(
            {
                "query": query,
                "hits": len(results),
                "top_scores": [
                    round(float(doc.metadata.get("bm25_score", 0.0)), 4)
                    for doc in results[:3]
                ],
                "top_term_overlap": [
                    int(doc.metadata.get("bm25_term_overlap", 0)) for doc in results[:3]
                ],
            }
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
