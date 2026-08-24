from pathlib import Path

from langchain_core.messages import AIMessage

from agent_material.rag_engine import multimodal_answer as multimodal
from agent_material.rag_engine.rag_service import RagService
from agent_material.rag_engine.request_context import RequestContext


class _FakeModel:
    def __init__(self, answer="视觉答案 [图表证据1]"):
        self.answer = answer
        self.messages = None

    def invoke(self, messages, config=None):
        self.messages = messages
        return AIMessage(content=self.answer)


class _Logger:
    def warning(self, *args, **kwargs):
        pass


def _config(tmp_path):
    return {
        "enabled": "true",
        "model": "fake-vl",
        "max_images": 2,
        "max_image_bytes": 1024,
    }


def _source(tmp_path) -> dict:
    image = tmp_path / "doc" / "v1" / "figure.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"png-bytes")
    return {
        "index": 1,
        "content_type": "figure",
        "artifact_id": "artifact-1",
        "caption": "FIG. 1",
        "asset_uri": "rag-asset://doc/v1/figure.png",
        "visual_status": "completed",
    }


def _context(source: dict) -> RequestContext:
    ctx = RequestContext(original_query="图1表达了什么？")
    ctx.route_decision = "rag"
    ctx.freeze_context("[图表证据1] FIG. 1", [], [source])
    return ctx


def test_multimodal_payload_contains_data_url_not_local_path(monkeypatch, tmp_path):
    monkeypatch.setitem(
        multimodal.vector_store_config,
        "structured_index",
        {
            "artifact_root": str(tmp_path),
            "visual_enrichment": {"online_answer": _config(tmp_path)},
        },
    )
    model = _FakeModel()
    generator = multimodal.MultimodalAnswerGenerator(_config(tmp_path), model=model)

    answer = generator.generate(
        query="图1表达了什么？",
        context="[图表证据1] FIG. 1",
        visual_sources=[_source(tmp_path)],
    )

    assert answer.startswith("视觉答案")
    content = model.messages[1].content
    image_url = next(
        block["image_url"]["url"]
        for block in content
        if block["type"] == "image_url"
    )
    assert image_url.startswith("data:image/png;base64,")
    assert str(Path(tmp_path)) not in str(model.messages)


def test_rag_service_uses_multimodal_then_falls_back(monkeypatch, tmp_path):
    config = _config(tmp_path)
    monkeypatch.setitem(
        multimodal.vector_store_config,
        "structured_index",
        {
            "artifact_root": str(tmp_path),
            "visual_enrichment": {"online_answer": config},
        },
    )
    ctx = _context(_source(tmp_path))
    service = RagService.__new__(RagService)
    service._logger = _Logger()

    class Generator:
        model_name = "fake-vl"

        def generate(self, **kwargs):
            return "多模态回答"

    service._multimodal_generator = Generator()
    assert list(service.stream_answer(ctx)) == ["多模态回答"]

    class BrokenGenerator:
        model_name = "fake-vl"

        def generate(self, **kwargs):
            raise TimeoutError("private details")

    class TextChain:
        def stream(self, payload, config=None):
            yield "文本降级回答"

    fallback_ctx = _context(_source(tmp_path))
    service._multimodal_generator = BrokenGenerator()
    service.model = object()
    service._get_chain = lambda: TextChain()
    assert list(service.stream_answer(fallback_ctx)) == ["文本降级回答"]
    assert fallback_ctx.degradation_events[0]["component"] == "multimodal_answer"
    assert "private details" not in str(fallback_ctx.degradation_events)


def test_numeric_fact_gate_returns_controlled_answer_without_model_call(
    monkeypatch, tmp_path
):
    config = _config(tmp_path)
    monkeypatch.setitem(
        multimodal.vector_store_config,
        "structured_index",
        {
            "artifact_root": str(tmp_path),
            "visual_enrichment": {"online_answer": config},
        },
    )
    source = _source(tmp_path)
    source.update(
        {"numeric_fact_status": "blocked", "legend_mapping_status": "blocked"}
    )
    ctx = RequestContext(original_query="300 K 时三个方向分别是多少？")
    ctx.route_decision = "rag"
    ctx.freeze_context("[图表证据1] FIG. 1", [], [source])
    service = RagService.__new__(RagService)
    service._logger = _Logger()

    class MustNotRun:
        model_name = "fake-vl"

        def generate(self, **kwargs):
            raise AssertionError("numeric gate should stop the model call")

    service._multimodal_generator = MustNotRun()
    answer = "".join(service.stream_answer(ctx))

    assert "尚未通过图例映射与坐标数值核验" in answer
    assert ctx.answer_mode == "numeric_guardrail"
    assert ctx.degradation_events[0]["component"] == "numeric_fact_gate"
