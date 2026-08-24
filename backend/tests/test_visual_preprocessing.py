from pathlib import Path

import pytest
from langchain_core.documents import Document

from agent_material.rag_engine.indexing import visual_preprocessing as preprocessing


def _enable(monkeypatch, tmp_path, **overrides):
    config = {
        "enabled": True,
        "min_width_px": 600,
        "min_height_px": 400,
        "render_dpi": 216,
        "max_render_dpi": 300,
        "padding_points": 0,
        "panel_crops_enabled": True,
        "whitespace_ratio": 0.008,
        "min_gap_ratio": 0.018,
        "min_panel_ratio": 0.2,
        **overrides,
    }
    monkeypatch.setitem(
        preprocessing.vector_store_config,
        "structured_index",
        {
            "artifact_root": str(tmp_path / "assets"),
            "visual_enrichment": {
                "enabled": True,
                "image_preprocessing": config,
            },
        },
    )


def _low_resolution_asset(tmp_path: Path) -> tuple[Path, str]:
    Image = pytest.importorskip("PIL.Image")
    root = tmp_path / "assets" / "doc" / "v1"
    root.mkdir(parents=True)
    target = root / "original.png"
    Image.new("RGB", (200, 150), "white").save(target)
    return target, "rag-asset://doc/v1/original.png"


def _pdf_with_figure(tmp_path: Path) -> Path:
    fitz = pytest.importorskip("fitz")
    target = tmp_path / "source.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=300, height=400)
    page.draw_rect(fitz.Rect(50, 100, 250, 300), color=(0, 0, 0), fill=(1, 1, 1))
    page.draw_line((60, 280), (240, 120), color=(1, 0, 0), width=2)
    pdf.save(target)
    pdf.close()
    return target


def test_low_resolution_mineru_image_is_rerendered_from_pdf(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    _, uri = _low_resolution_asset(tmp_path)
    source = _pdf_with_figure(tmp_path)
    document = Document(
        page_content="[图表] FIG. 1",
        metadata={
            "content_type": "figure",
            "source": str(source),
            "page_start": 1,
            "bbox": [50, 100, 250, 300],
            "bbox_coordinate_space": "pdf_points",
            "asset_uri": uri,
            "asset_hash": "original-hash",
            "document_id": "doc",
            "version_id": "v1",
        },
    )

    result = preprocessing.preprocess_figure_document(document)

    assert result.metadata["visual_preprocess_status"] == "pdf_rendered"
    assert result.metadata["visual_asset_uri"] != uri
    assert result.metadata["visual_width_px"] == 600
    assert result.metadata["visual_height_px"] == 600
    assert result.metadata["visual_render_bbox_mode"] == "pdf_points"
    assert preprocessing.ArtifactStore().resolve(
        result.metadata["visual_asset_uri"]
    ).is_file()


def test_panel_detection_and_query_selection_are_deterministic(monkeypatch, tmp_path):
    Image = pytest.importorskip("PIL.Image")
    ImageDraw = pytest.importorskip("PIL.ImageDraw")
    _enable(
        monkeypatch,
        tmp_path,
        min_width_px=500,
        min_height_px=300,
        auto_panel_detection=True,
    )
    root = tmp_path / "assets" / "doc" / "v1"
    root.mkdir(parents=True)
    target = root / "composite.png"
    image = Image.new("RGB", (1000, 400), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 30, 440, 370), fill="black")
    draw.rectangle((560, 30, 960, 370), fill="black")
    image.save(target)
    document = Document(
        page_content="[图表] (a) (b)",
        metadata={
            "content_type": "figure",
            "asset_uri": "rag-asset://doc/v1/composite.png",
            "asset_hash": "hash",
            "document_id": "doc",
            "version_id": "v1",
        },
    )

    result = preprocessing.preprocess_figure_document(document)

    panels = result.metadata["panel_assets"]
    assert [panel["label"] for panel in panels] == ["a", "b"]
    assert preprocessing.select_visual_asset_uri(result.metadata, "请分析图1(b)") == (
        panels[1]["asset_uri"]
    )
    assert preprocessing.select_visual_asset_uri(result.metadata, "请分析整张图") == (
        result.metadata["visual_asset_uri"]
    )


def test_preprocessing_is_disabled_with_visual_feature(monkeypatch, tmp_path):
    monkeypatch.setitem(
        preprocessing.vector_store_config,
        "structured_index",
        {"visual_enrichment": {"enabled": False}},
    )
    document = Document(page_content="x", metadata={"content_type": "figure"})
    assert preprocessing.preprocess_figure_document(document) is document
