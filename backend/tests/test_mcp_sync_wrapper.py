"""Tests for sync tool wrapper."""

import asyncio
import pytest

from agent_material.tools.sync import make_sync_tool_wrapper


class TestSyncToolWrapper:
    def test_sync_wrapper_for_simple_coro(self):
        async def my_coro(x: int) -> int:
            return x * 2

        wrapper = make_sync_tool_wrapper(my_coro, "test_tool")
        result = wrapper(5)
        assert result == 10

    def test_sync_wrapper_with_exception(self):
        async def failing_coro():
            raise ValueError("test error")

        wrapper = make_sync_tool_wrapper(failing_coro, "test_tool")
        with pytest.raises(ValueError, match="test error"):
            wrapper()

    def test_sync_wrapper_from_running_loop(self):
        async def my_coro(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 3

        wrapper = make_sync_tool_wrapper(my_coro, "test_tool")

        async def run():
            # When called from a running loop, wrapper should use ThreadPoolExecutor
            return wrapper(7)

        result = asyncio.run(run())
        assert result == 21
