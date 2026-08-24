"""Run an explicitly authorised live cross-modal embedding smoke test."""

from __future__ import annotations

import argparse
import json
import math
import time

from dotenv import load_dotenv

from agent_material.rag_engine.indexing.visual_vector_index import (
    DashScopeMultimodalEmbeddings,
)


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _cosine(left: list[float], right: list[float]) -> float:
    left_norm = _norm(left)
    right_norm = _norm(right)
    if left_norm == 0 or right_norm == 0:
        raise RuntimeError("多模态向量不得为零向量")
    return sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset_uri", help="Authorised logical rag-asset:// URI")
    parser.add_argument("query", help="Relevant text query")
    parser.add_argument(
        "--negative-query",
        default="一只猫在草地上奔跑",
        help="Unrelated query used for a minimal ranking sanity check",
    )
    parser.add_argument("--model", default="qwen3-vl-embedding")
    parser.add_argument("--dimension", type=int, default=1024)
    parser.add_argument(
        "--allow-external-image-processing",
        action="store_true",
        help="Required acknowledgement that the image is sent to the provider",
    )
    args = parser.parse_args()
    if not args.allow_external_image_processing:
        parser.error("必须显式传入 --allow-external-image-processing")

    load_dotenv()
    embedder = DashScopeMultimodalEmbeddings(
        {
            "model": args.model,
            "dimension": args.dimension,
            "timeout_seconds": 60,
            "max_retries": 1,
        }
    )
    started = time.time()
    image_vector = embedder.embed_image_uris([args.asset_uri])[0]
    related_vector, negative_vector = embedder.embed_documents(
        [args.query, args.negative_query]
    )
    related_cosine = _cosine(image_vector, related_vector)
    negative_cosine = _cosine(image_vector, negative_vector)

    if len(image_vector) != args.dimension:
        raise RuntimeError("图片向量维度与配置不一致")
    if related_cosine <= negative_cosine:
        raise RuntimeError("相关查询相似度未高于无关查询")

    print(
        json.dumps(
            {
                "status": "passed",
                "model": embedder.model,
                "dimension": len(image_vector),
                "image_norm": round(_norm(image_vector), 6),
                "related_text_norm": round(_norm(related_vector), 6),
                "negative_text_norm": round(_norm(negative_vector), 6),
                "related_cosine": round(related_cosine, 6),
                "negative_cosine": round(negative_cosine, 6),
                "cosine_margin": round(related_cosine - negative_cosine, 6),
                "elapsed_seconds": round(time.time() - started, 3),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
