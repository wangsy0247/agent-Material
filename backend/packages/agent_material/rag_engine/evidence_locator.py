"""Stable, JSON-safe evidence locators for unified RAG search results.

The locator deliberately exposes logical ``rag-asset://`` URIs only.  Physical
host paths remain an implementation detail of :class:`ArtifactStore` and must
never cross the REST or Agent-tool boundary.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


ASSET_SCHEME = "rag-asset://"
LOCATOR_VERSION = "visual-evidence-v1"


def _logical_asset_uri(value: Any) -> str | None:
    uri = str(value or "").strip()
    return uri if uri.startswith(ASSET_SCHEME) else None


def _bbox(value: Any) -> list[int | float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    result: list[int | float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        if not math.isfinite(float(item)):
            return None
        result.append(item)
    return result


def _channels(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        value = [value] if value else []
    result: list[str] = []
    for item in value:
        channel = str(item or "").strip()
        if channel and channel not in result:
            result.append(channel)
    return result


def build_evidence_locator(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return a versioned locator shared by REST search, Agent tools and QA.

    A cross-modal panel match takes precedence over the parent figure location.
    Text evidence still receives document/page/chunk coordinates, but has no image
    URI or visual bounding box.
    """

    content_type = str(metadata.get("content_type") or "text")
    artifact_id = metadata.get("artifact_id")
    panel_id = metadata.get("matched_panel_id") or metadata.get("panel_id")
    visual_entity_id = metadata.get("matched_visual_entity_id") or metadata.get("visual_entity_id")
    evidence_id = (
        visual_entity_id or artifact_id or metadata.get("chunk_id") or metadata.get("parent_id")
    )

    asset_uri = _logical_asset_uri(
        metadata.get("matched_visual_asset_uri")
        or metadata.get("visual_asset_uri")
        or metadata.get("asset_uri")
    )
    bbox = _bbox(
        metadata.get("matched_evidence_bbox")
        or metadata.get("evidence_bbox")
        or metadata.get("bbox")
    )
    bbox_coordinate_space = (
        metadata.get("matched_evidence_bbox_coordinate_space")
        or metadata.get("evidence_bbox_coordinate_space")
        or metadata.get("bbox_coordinate_space")
    )

    numeric_status = str(metadata.get("numeric_fact_status") or "not_applicable")
    legend_status = str(metadata.get("legend_mapping_status") or "not_applicable")
    return {
        "locator_version": LOCATOR_VERSION,
        "evidence_id": str(evidence_id) if evidence_id else None,
        "content_type": content_type,
        "document_id": metadata.get("document_id"),
        "version_id": metadata.get("version_id"),
        "chunk_id": metadata.get("chunk_id"),
        "parent_id": metadata.get("parent_id"),
        "artifact_id": artifact_id,
        "visual_entity_id": visual_entity_id,
        "panel_id": str(panel_id) if panel_id is not None else None,
        "page_start": metadata.get("page_start"),
        "page_end": metadata.get("page_end"),
        "page_unknown": bool(metadata.get("page_unknown", False)),
        "bbox": bbox,
        "bbox_coordinate_space": str(bbox_coordinate_space) if bbox_coordinate_space else None,
        "asset_uri": asset_uri,
        "retrieval_channels": _channels(
            metadata.get("retrieval_channels") or metadata.get("retrieval_channel")
        ),
        "visual_similarity": metadata.get("visual_similarity"),
        "fact_gate": {
            "readability_status": str(metadata.get("readability_status") or "unknown"),
            "legend_mapping_status": legend_status,
            "numeric_fact_status": numeric_status,
            "numeric_answer_allowed": numeric_status == "verified" and legend_status != "blocked",
        },
    }
