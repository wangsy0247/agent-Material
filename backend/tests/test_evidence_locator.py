from agent_material.rag_engine.evidence_locator import build_evidence_locator


def test_panel_match_takes_precedence_and_exposes_fact_gate():
    locator = build_evidence_locator(
        {
            "content_type": "figure",
            "document_id": "doc-1",
            "version_id": "v1",
            "parent_id": "parent-1",
            "chunk_id": "chunk-1",
            "artifact_id": "figure-3",
            "page_start": 6,
            "page_end": 6,
            "bbox": [10, 20, 900, 700],
            "bbox_coordinate_space": "normalised_1000",
            "visual_asset_uri": "rag-asset://doc-1/v1/figure.png",
            "matched_panel_id": "b",
            "matched_visual_entity_id": "figure-3:panel:b",
            "matched_visual_asset_uri": "rag-asset://doc-1/v1/panel-b.png",
            "matched_evidence_bbox": [420, 10, 800, 380],
            "matched_evidence_bbox_coordinate_space": "image_pixels",
            "retrieval_channels": ["vector", "visual", "visual"],
            "visual_similarity": 0.87,
            "readability_status": "readable",
            "legend_mapping_status": "verified",
            "numeric_fact_status": "verified",
        }
    )

    assert locator["locator_version"] == "visual-evidence-v1"
    assert locator["evidence_id"] == "figure-3:panel:b"
    assert locator["panel_id"] == "b"
    assert locator["asset_uri"].endswith("panel-b.png")
    assert locator["bbox"] == [420, 10, 800, 380]
    assert locator["bbox_coordinate_space"] == "image_pixels"
    assert locator["retrieval_channels"] == ["vector", "visual"]
    assert locator["fact_gate"]["numeric_answer_allowed"] is True


def test_locator_never_exposes_physical_path_or_invalid_bbox():
    locator = build_evidence_locator(
        {
            "content_type": "figure",
            "artifact_id": "figure-1",
            "asset_uri": "/mnt/d/private/figure.png",
            "bbox": [0, 1, "bad", 3],
            "numeric_fact_status": "verified",
            "legend_mapping_status": "blocked",
        }
    )

    assert locator["asset_uri"] is None
    assert locator["bbox"] is None
    assert locator["fact_gate"]["numeric_answer_allowed"] is False


def test_text_evidence_retains_document_and_page_location():
    locator = build_evidence_locator(
        {
            "content_type": "text",
            "document_id": "doc-2",
            "version_id": "v2",
            "chunk_id": "chunk-2",
            "page_start": 3,
            "page_end": 4,
            "retrieval_channel": "bm25",
        }
    )

    assert locator["evidence_id"] == "chunk-2"
    assert locator["asset_uri"] is None
    assert locator["page_start"] == 3
    assert locator["page_end"] == 4
    assert locator["retrieval_channels"] == ["bm25"]
