from agent_material.rag_engine.retrieval_routing import plan_retrieval_route


def test_plain_concept_query_stays_text_only():
    plan = plan_retrieval_route(
        "什么是密度泛函理论？", visual_available=True, question_type="factual"
    )
    assert plan.route_type == "text_only"
    assert plan.enable_visual is False
    assert plan.enable_text_vector is True
    assert plan.enable_bm25 is True


def test_panel_reference_enables_visual_and_extracts_panel_id():
    plan = plan_retrieval_route(
        "请解释图3(b)中的曲线趋势", visual_available=True
    )
    assert plan.route_type == "panel_visual"
    assert plan.enable_visual is True
    assert plan.panel_id == "b"


def test_numeric_chart_query_requires_fact_gate():
    plan = plan_retrieval_route(
        "图3(b)在300 K时的热导率是多少？", visual_available=True
    )
    assert plan.route_type == "numeric_visual"
    assert plan.numeric_fact_required is True
    assert plan.panel_id == "b"


def test_multi_figure_comparison_recommends_decomposition_without_forcing_it():
    plan = plan_retrieval_route(
        "比较图2和图3中的变化趋势", visual_available=True, question_type="comparison"
    )
    assert plan.route_type == "multi_figure_visual"
    assert plan.decomposition_recommended is True
    assert plan.enable_visual is True


def test_visual_request_records_disabled_fallback():
    plan = plan_retrieval_route("图4展示了什么？", visual_available=False)
    assert plan.visual_requested is True
    assert plan.enable_visual is False
    assert "visual_feature_disabled" in plan.reason_codes
