"""Portable MCP stdio command configuration tests."""

import sys

from agent_material.config.extensions_config import McpServerConfig
from agent_material.mcp.client import build_server_params


def test_python_placeholder_uses_current_interpreter():
    config = McpServerConfig(command="{python}", args=["-m", "example_server"])

    params = build_server_params("example", config)

    assert params["command"] == sys.executable
    assert params["args"] == ["-m", "example_server"]


def test_explicit_stdio_command_is_preserved():
    config = McpServerConfig(command="uv", args=["run", "example_server"])

    params = build_server_params("example", config)

    assert params["command"] == "uv"
