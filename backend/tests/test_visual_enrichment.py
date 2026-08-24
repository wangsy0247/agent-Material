import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from agent_material.rag_engine.indexing import visual_enrichment as visual


class _FakeClient:
    model = "fake-vl"

    def __init__(self, *, error: Exception | None = None):
        self.calls = 0
        self.error = error

    def describe(self, image_path, context):
        self.calls += 1
        if self.error:
            raise self.error
        assert image_path.is_file()
        assert context["caption"] == "FIG. 3"
        return _description()


def _visual_doc(tmp_path) -> Document:
    target = tmp_path / "doc" / "version" / "figure.png"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"fake-image")
    return Document(
        page_content="[图表]\nFIG. 3",
        metadata={
            "content_type": "figure",
            "caption": "FIG. 3",
            "asset_uri": "rag-asset://doc/version/figure.png",
            "asset_hash": "asset-hash-1",
            "artifact_ordinal": 0,
        },
    )


def _enable(monkeypatch, tmp_path, *, failure_policy="fallback_text"):
    monkeypatch.setitem(
        visual.vector_store_config,
        "structured_index",
        {
            "artifact_root": str(tmp_path),
            "visual_enrichment": {
                "enabled": "true",
                "model": "fake-vl",
                "prompt_version": "visual-chart-v1",
                "max_images_per_document": 4,
                "failure_policy": failure_policy,
            },
        },
    )


def _description() -> visual.VisualDescription:
    return visual.VisualDescription(
        chart_type="line_chart",
        title="Thermal conductivity",
        x_axis={"name": "Temperature", "unit": "K"},
        y_axis={"name": "Thermal conductivity", "unit": "W/mK"},
        legend=["[100]", "[010]", "[001]"],
        trends=["三条曲线随温度升高而下降"],
        key_values=["300 K 时 [010] 方向最高"],
        visual_summary="比较三个晶向的热导率。",
        confidence=0.92,
    )


def test_visual_enrichment_is_opt_in(monkeypatch):
    monkeypatch.setitem(
        visual.vector_store_config,
        "structured_index",
        {"visual_enrichment": {"enabled": "false"}},
    )
    assert visual.visual_enrichment_enabled() is False


def test_parse_fenced_json_and_reject_unknown_fields():
    payload = visual.visual_description_json(_description())
    parsed = visual.parse_visual_description(f"```json\n{payload}\n```")
    assert parsed.chart_type == "line_chart"
    assert parsed.confidence == pytest.approx(0.92)

    invalid = parsed.model_dump()
    invalid["untrusted_extra"] = "value"
    with pytest.raises(ValidationError):
        visual.parse_visual_description(invalid)


def test_apply_visual_description_is_idempotent():
    source = Document(
        page_content="[图表]\nFIG. 3.",
        metadata={"artifact_id": "artifact-1", "content_type": "figure"},
    )
    first = visual.apply_visual_description(
        source, _description(), model="qwen3-vl-plus"
    )
    second = visual.apply_visual_description(
        first, _description(), model="qwen3-vl-plus"
    )

    assert first.page_content == second.page_content
    assert first.metadata["pre_visual_searchable_text"] == source.page_content
    assert first.metadata["visual_status"] == "completed"
    assert first.metadata["visual_confidence"] == pytest.approx(0.92)
    assert "300 K" not in first.page_content
    assert first.metadata["numeric_fact_status"] == "blocked"
    assert first.metadata["legend_mapping_status"] == "blocked"
    assert first.metadata["visual_description"]["key_values"] == [
        "300 K 时 [010] 方向最高"
    ]


def test_offline_enrichment_uses_validated_cache(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    client = _FakeClient()
    document = _visual_doc(tmp_path)

    first = visual.enrich_artifact_documents([document], client=client)[0]
    second = visual.enrich_artifact_documents([document], client=client)[0]

    assert client.calls == 1
    assert first.metadata["visual_status"] == "completed"
    assert second.metadata["visual_status"] == "completed"
    assert second.metadata["visual_model"] == "fake-vl"
    assert "视觉解析" in second.page_content


def test_offline_enrichment_falls_back_without_leaking_error(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    document = _visual_doc(tmp_path)
    result = visual.enrich_artifact_documents(
        [document], client=_FakeClient(error=RuntimeError("private/path"))
    )[0]

    assert result.page_content == document.page_content
    assert result.metadata["visual_status"] == "failed"
    assert result.metadata["visual_error_code"] == "RuntimeError"
    assert "private/path" not in str(result.metadata)


def test_offline_enrichment_can_fail_closed(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path, failure_policy="fail_version")
    with pytest.raises(visual.VisualEnrichmentError, match="关闭当前索引版本"):
        visual.enrich_artifact_documents(
            [_visual_doc(tmp_path)],
            client=_FakeClient(error=RuntimeError("service unavailable")),
        )
