from agent_material.rag_engine.indexing.visual_enrichment import VisualDescription
from agent_material.rag_engine.indexing.visual_fact_gate import (
    evaluate_visual_fact_gate,
    requires_precise_numeric_answer,
    should_block_numeric_answer,
)


def _description():
    return VisualDescription(
        chart_type="line_chart",
        x_axis={"name": "Temperature", "unit": "K"},
        y_axis={"name": "Thermal conductivity", "unit": "W/mK"},
        legend=["[100]", "[010]", "[001]"],
        trends=["[010] 在全部温度范围内最高"],
        key_values=["300 K 时 [010] 为 21 W/mK"],
        visual_summary="300 K 时三个晶向存在明显差异。",
        confidence=0.99,
    )


def _high_resolution_metadata():
    return {
        "visual_preprocess_status": "pdf_rendered",
        "visual_width_px": 1600,
        "visual_height_px": 1000,
    }


def test_model_confidence_does_not_bypass_fact_gate():
    gate = evaluate_visual_fact_gate(_description(), _high_resolution_metadata())

    assert gate["readability_status"] == "readable"
    assert gate["legend_mapping_status"] == "blocked"
    assert gate["numeric_fact_status"] == "blocked"
    assert gate["verified_key_values"] == []
    assert gate["safe_trends"] == []
    assert gate["safe_visual_summary"] == ""


def test_trusted_manual_verification_releases_exact_values():
    metadata = {
        **_high_resolution_metadata(),
        "legend_mapping_verified": True,
        "legend_verification_source": "human",
        "numeric_values_verified": True,
        "numeric_verification_source": "gold_eval",
        "verified_numeric_facts": ["300 K 时 [010] 为 21 W/mK"],
    }
    gate = evaluate_visual_fact_gate(_description(), metadata)

    assert gate["legend_mapping_status"] == "verified"
    assert gate["numeric_fact_status"] == "verified"
    assert gate["verified_key_values"] == ["300 K 时 [010] 为 21 W/mK"]


def test_model_self_verification_is_not_trusted():
    metadata = {
        **_high_resolution_metadata(),
        "legend_mapping_verified": True,
        "legend_verification_source": "model",
        "numeric_values_verified": True,
        "numeric_verification_source": "model",
        "verified_numeric_facts": ["300 K 时 [010] 为 21 W/mK"],
    }
    gate = evaluate_visual_fact_gate(_description(), metadata)
    assert gate["legend_mapping_status"] == "blocked"
    assert gate["numeric_fact_status"] == "blocked"


def test_numeric_question_is_blocked_without_verified_source():
    assert requires_precise_numeric_answer("300 K 时三个方向分别是多少？") is True
    assert should_block_numeric_answer(
        "300 K 时三个方向分别是多少？",
        [{"numeric_fact_status": "blocked", "legend_mapping_status": "blocked"}],
    ) is True
    assert should_block_numeric_answer(
        "图中整体趋势如何？",
        [{"numeric_fact_status": "blocked", "legend_mapping_status": "blocked"}],
    ) is False
