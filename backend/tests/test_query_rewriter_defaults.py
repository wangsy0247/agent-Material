"""查询改写默认关闭与专用模型配置测试。"""

from __future__ import annotations

import inspect

from agent_material.rag_engine import query_rewriter as query_rewriter_module
from agent_material.rag_engine.query_rewriter import QueryRewriter
from agent_material.rag_engine.rag_service import RagService
from agent_material.utils.config_handler import vector_store_config


def test_query_rewriting_defaults_to_disabled() -> None:
    config = vector_store_config["query_rewriting"]

    assert config["enabled"] is False
    assert config["default_strategy"] == ""
    assert config["enable_hyde"] is False
    assert inspect.signature(RagService.prepare).parameters["enable_rewrite"].default is False
    assert inspect.signature(RagService.search).parameters["enable_rewrite"].default is False


def test_rewrite_requires_config_and_request_switches() -> None:
    service = RagService.__new__(RagService)

    service._rewrite_config = {"enabled": False}
    assert service._rewrite_enabled(False) is False
    assert service._rewrite_enabled(True) is False

    service._rewrite_config = {"enabled": True}
    assert service._rewrite_enabled(False) is False
    assert service._rewrite_enabled(True) is True


def test_query_rewriter_model_defaults_to_non_thinking(monkeypatch) -> None:
    captured: dict = {}
    fake_model = object()

    def fake_chat_openai(**kwargs):
        captured.update(kwargs)
        return fake_model

    monkeypatch.setattr(query_rewriter_module, "ChatOpenAI", fake_chat_openai)

    rewriter = QueryRewriter()

    assert rewriter._model is fake_model
    assert captured["streaming"] is False
    assert captured["max_tokens"] == 1024
    assert captured["extra_body"] == {"enable_thinking": False}
