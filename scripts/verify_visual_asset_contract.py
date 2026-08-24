"""Offline validation for a logical RAG visual asset.

This command never calls a model provider. It verifies that a real local image
can become a P2 visual entity and an external-safe data URI without exposing its
physical host path in the payload.
"""

from __future__ import annotations

import argparse
import json

from langchain_core.documents import Document

from agent_material.rag_engine.indexing.visual_vector_index import (
    DashScopeMultimodalEmbeddings,
    build_visual_index_documents,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset_uri", help="A logical rag-asset:// URI")
    args = parser.parse_args()

    document = Document(
        page_content="visual asset contract validation",
        metadata={
            "content_type": "figure",
            "artifact_id": "visual-contract-check",
            "document_id": "visual-contract-check",
            "version_id": "local",
            "caption": "visual asset contract validation",
            "visual_asset_uri": args.asset_uri,
            "page_unknown": True,
        },
    )
    entities = build_visual_index_documents([document])
    if len(entities) != 1:
        raise RuntimeError("真实图片未生成唯一主图视觉实体")

    embedder = DashScopeMultimodalEmbeddings({"dimension": 1024})
    data_uri = embedder._image_data_uri(args.asset_uri)
    lower_payload = data_uri.lower()
    result = {
        "status": "passed",
        "visual_entities": len(entities),
        "entity_id": entities[0].metadata["visual_entity_id"],
        "logical_uri_only": str(
            entities[0].metadata["visual_asset_uri"]
        ).startswith("rag-asset://"),
        "payload_is_data_uri": lower_payload.startswith("data:image/"),
        "host_path_in_payload": "/mnt/" in lower_payload or ":\\" in data_uri,
        "payload_chars": len(data_uri),
    }
    if not result["logical_uri_only"] or not result["payload_is_data_uri"]:
        raise RuntimeError("视觉资源契约验证失败")
    if result["host_path_in_payload"]:
        raise RuntimeError("视觉请求负载包含主机路径")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
