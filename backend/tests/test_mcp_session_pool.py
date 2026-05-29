"""Tests for MCP session pool."""

import asyncio
import pytest

from agent_material.mcp.session_pool import get_session_pool, MCPSessionPool, reset_session_pool


class TestMCPSessionPool:
    def setup_method(self):
        reset_session_pool()

    def test_singleton(self):
        pool1 = get_session_pool()
        pool2 = get_session_pool()
        assert pool1 is pool2

    def test_pool_initially_empty(self):
        pool = MCPSessionPool()
        assert len(pool._entries) == 0
        assert len(pool._context_managers) == 0

    def test_reset_pool(self):
        pool1 = get_session_pool()
        reset_session_pool()
        pool2 = get_session_pool()
        assert pool1 is not pool2
