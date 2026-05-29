"""Tests for MCP custom tool interceptors."""

import pytest

from agent_material.mcp.tools import get_mcp_tools
from agent_material.reflection import resolve_variable


def dummy_builder():
    async def interceptor(request, handler):
        return await handler(request)
    return interceptor


class TestCustomInterceptors:
    def test_resolve_variable(self):
        # Test resolving a built-in
        result = resolve_variable("builtins:len")
        assert result is len

    def test_resolve_variable_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid path format"):
            resolve_variable("no_colon_here")

    def test_resolve_variable_missing_module(self):
        with pytest.raises(ImportError):
            resolve_variable("nonexistent_module:var")
