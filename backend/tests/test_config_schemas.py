"""配置加载与请求/响应模型校验的纯单元测试。

覆盖：
- agent_material.utils.config_handler：真实 config/*.yaml 的加载与关键字段
- agent_material.utils.path_tools：get_project_path / get_abs_path
- app.gateway.schemas：BaseResponse / AgentChatRequest / AgentApprovalRequest / SSEEvent
- app.gateway.routers.agent 顶部的 AgentResumeRequest

全部为纯单元测试，不启动 TestClient，无任何外部副作用。
"""

import os
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

# backend/ 与 backend/packages 加入路径（pyproject 仅配置了 packages）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_material.utils.config_handler import (  # noqa: E402
    get_agent_config,
    get_memory_config,
    get_prompt_config,
    get_queue_config,
    get_rag_config,
    get_sc_config,
    get_vector_store_config,
)
from agent_material.utils.path_tools import get_abs_path, get_project_path  # noqa: E402
from app.gateway.schemas import (  # noqa: E402
    AgentApprovalRequest,
    AgentChatRequest,
    BaseResponse,
    SSEEvent,
)
from app.gateway.routers.agent import AgentResumeRequest  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ==================== 路径工具 ====================


class TestPathTools:
    def test_project_path_is_repo_root(self):
        root = get_project_path()
        assert os.path.isdir(os.path.join(root, "backend"))
        assert os.path.isdir(os.path.join(root, "config"))
        assert Path(root) == PROJECT_ROOT

    def test_abs_path_joins_project_root(self):
        abs_path = get_abs_path("config/rag_config.yaml")
        assert os.path.isabs(abs_path)
        assert abs_path == os.path.normpath(
            os.path.join(get_project_path(), "config/rag_config.yaml")
        )
        assert os.path.isfile(abs_path)

    def test_abs_path_normalizes_separators(self):
        assert get_abs_path("config//rag_config.yaml") == get_abs_path(
            "config/rag_config.yaml"
        )


# ==================== 配置加载 ====================


class TestConfigHandler:
    def test_vector_store_connection_string_expands_environment(self, monkeypatch):
        expected = "postgresql+psycopg://user:pass@db.example:5432/rag"
        monkeypatch.setenv("PGVECTOR_URL", expected)
        assert get_vector_store_config()["connection_string"] == expected

    def test_rag_config_loads_with_embedding_model(self):
        config = get_rag_config()
        assert isinstance(config, dict)
        assert "embedding_model_name" in config
        assert isinstance(config["embedding_model_name"], str)

    def test_agent_config_loads_with_key_fields(self):
        config = get_agent_config()
        assert isinstance(config, dict)
        assert "chat_model_name" in config
        assert isinstance(config["chat_model_name"], str)
        assert "max_tokens" in config
        assert isinstance(config["max_tokens"], int)
        assert "checkpointer" in config
        assert "type" in config["checkpointer"]

    def test_vector_store_config_loads_with_document_directory(self):
        config = get_vector_store_config()
        assert isinstance(config, dict)
        assert "document_directory" in config
        assert config["document_directory"] == "data"
        assert "embedding_model_name" in config
        assert "chunk_size" in config
        assert isinstance(config["chunk_size"], int)

    def test_prompt_config_loads_with_prompt_paths(self):
        config = get_prompt_config()
        assert isinstance(config, dict)
        assert "rag_prompt_path" in config
        assert "sys_prompt_path" in config
        # 配置中引用的提示词文件应真实存在
        for key in ("rag_prompt_path", "sys_prompt_path"):
            prompt_file = Path(get_abs_path(config[key]))
            assert prompt_file.is_file(), f"{key} 指向的文件不存在: {prompt_file}"

    def test_memory_config_loads_with_sections(self):
        config = get_memory_config()
        assert isinstance(config, dict)
        assert "manager" in config
        assert "session_store" in config
        assert "long_term" in config
        assert "db_path" in config["session_store"]

    def test_queue_config_loads_with_queue_section(self):
        config = get_queue_config()
        assert isinstance(config, dict)
        assert "queue" in config
        assert "poll_interval" in config["queue"]
        assert "lease_seconds" in config["queue"]

    def test_sc_config_empty_file_returns_empty_dict(self):
        # config/sc_config.yaml 仅有注释无配置项，get_sc_config 兜底为空 dict
        assert get_sc_config() == {}

    def test_module_level_config_dicts(self):
        import agent_material.utils.config_handler as ch

        assert ch.rag_config == get_rag_config()
        assert ch.agent_config == get_agent_config()
        assert ch.vector_store_config == get_vector_store_config()


# ==================== 通用响应模型 ====================


class TestBaseResponse:
    def test_defaults(self):
        resp = BaseResponse()
        assert resp.code == 200
        assert resp.message == "success"
        assert resp.data is None

    def test_custom_values(self):
        resp = BaseResponse(code=404, message="not found", data={"a": 1})
        assert resp.code == 404
        assert resp.message == "not found"
        assert resp.data == {"a": 1}

    def test_serialization(self):
        dumped = BaseResponse(data=[1, 2]).model_dump()
        assert dumped == {
            "code": 200,
            "message": "success",
            "data": [1, 2],
            "total": None,
            "degradation_events": None,
        }

    def test_total_for_list_responses(self):
        """列表类响应可携带总数（修复前 total 被静默丢弃）"""
        resp = BaseResponse(data=[1, 2], total=2)
        assert resp.model_dump()["total"] == 2


# ==================== Agent 请求模型 ====================


class TestAgentChatRequest:
    def test_minimal_valid(self):
        req = AgentChatRequest(question="什么是深度学习势能？")
        assert req.question == "什么是深度学习势能？"
        assert req.session_id == "default"
        assert req.stream is True

    def test_question_required(self):
        with pytest.raises(ValidationError):
            AgentChatRequest()

    def test_empty_question_rejected(self):
        with pytest.raises(ValidationError):
            AgentChatRequest(question="")

    def test_question_max_length(self):
        req = AgentChatRequest(question="a" * 10000)
        assert len(req.question) == 10000
        with pytest.raises(ValidationError):
            AgentChatRequest(question="a" * 10001)

    def test_explicit_fields(self):
        req = AgentChatRequest(question="q", session_id="s-1", stream=False)
        assert req.session_id == "s-1"
        assert req.stream is False


class TestAgentResumeRequest:
    def test_valid(self):
        req = AgentResumeRequest(session_id="s-1")
        assert req.session_id == "s-1"

    def test_session_id_required(self):
        with pytest.raises(ValidationError):
            AgentResumeRequest()


class TestAgentApprovalRequest:
    def test_approve_minimal(self):
        req = AgentApprovalRequest(session_id="s-1", action="approve")
        assert req.session_id == "s-1"
        assert req.action == "approve"
        assert req.comment is None

    def test_reject_with_comment(self):
        req = AgentApprovalRequest(
            session_id="s-1", action="reject", comment="不允许下载"
        )
        assert req.action == "reject"
        assert req.comment == "不允许下载"

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            AgentApprovalRequest()
        with pytest.raises(ValidationError):
            AgentApprovalRequest(session_id="s-1")
        with pytest.raises(ValidationError):
            AgentApprovalRequest(action="approve")

    def test_invalid_action_rejected(self):
        with pytest.raises(ValidationError):
            AgentApprovalRequest(session_id="s-1", action="maybe")


# ==================== SSE 事件模型 ====================


class TestSSEEvent:
    def test_token_event_serialization_keys(self):
        event = SSEEvent(type="token", content="你好", index=1, timestamp=1.5)
        dumped = event.model_dump()
        assert set(dumped) == {"type", "content", "index", "timestamp"}
        assert dumped["type"] == "token"
        assert dumped["content"] == "你好"
        assert dumped["index"] == 1
        assert dumped["timestamp"] == 1.5

    def test_defaults(self):
        event = SSEEvent(type="done")
        assert event.content is None
        assert event.index is None
        assert event.timestamp is None

    def test_type_required(self):
        with pytest.raises(ValidationError):
            SSEEvent()

    def test_invalid_type_rejected(self):
        with pytest.raises(ValidationError):
            SSEEvent(type="not_a_real_event")

    def test_all_declared_event_types_accepted(self):
        for event_type in (
            "start",
            "status",
            "thinking",
            "tool_call",
            "tool_result",
            "token",
            "answer",
            "sources",
            "approval_required",
            "error",
            "done",
            "cancelled",
        ):
            assert SSEEvent(type=event_type).type == event_type

    def test_dict_and_list_content(self):
        event = SSEEvent(type="tool_call", content={"name": "mp_download"})
        assert event.content == {"name": "mp_download"}
        event = SSEEvent(type="sources", content=[{"source": "a.pdf"}])
        assert event.content == [{"source": "a.pdf"}]

    def test_serialized_event_is_json_compatible(self):
        event = SSEEvent(type="token", content="x", index=2, timestamp=3.0)
        payload = event.model_dump_json()
        assert '"type":"token"' in payload
        assert '"index":2' in payload
