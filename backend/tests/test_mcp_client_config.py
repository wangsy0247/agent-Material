"""Tests for MCP client configuration building."""

import pytest

from agent_material.config.extensions_config import McpOAuthConfig, McpServerConfig
from agent_material.mcp.client import build_server_params, build_servers_config


class TestBuildServerParams:
    def test_stdio_server(self):
        config = McpServerConfig(
            enabled=True,
            type="stdio",
            command="python",
            args=["-m", "test_server"],
            env={"KEY": "value"},
        )
        params = build_server_params("test", config)
        assert params["transport"] == "stdio"
        assert params["command"] == "python"
        assert params["args"] == ["-m", "test_server"]
        assert params["env"] == {"KEY": "value"}

    def test_http_server(self):
        config = McpServerConfig(
            enabled=True,
            type="http",
            url="http://localhost:8000/mcp",
            headers={"X-Custom": "header"},
        )
        params = build_server_params("test", config)
        assert params["transport"] == "http"
        assert params["url"] == "http://localhost:8000/mcp"
        assert params["headers"] == {"X-Custom": "header"}

    def test_sse_server(self):
        config = McpServerConfig(
            enabled=True,
            type="sse",
            url="http://localhost:8000/sse",
        )
        params = build_server_params("test", config)
        assert params["transport"] == "sse"
        assert params["url"] == "http://localhost:8000/sse"

    def test_stdio_missing_command(self):
        config = McpServerConfig(enabled=True, type="stdio")
        with pytest.raises(ValueError, match="requires 'command' field"):
            build_server_params("test", config)

    def test_http_missing_url(self):
        config = McpServerConfig(enabled=True, type="http")
        with pytest.raises(ValueError, match="requires 'url' field"):
            build_server_params("test", config)

    def test_unsupported_transport(self):
        config = McpServerConfig(enabled=True, type="websocket")
        with pytest.raises(ValueError, match="unsupported transport type"):
            build_server_params("test", config)
