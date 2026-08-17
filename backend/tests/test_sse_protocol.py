"""SSE 流式协议补齐的单元测试（Phase 1）。

覆盖：帧格式（index 字段）、心跳帧、with_heartbeat 包装、
stream_buffer 的 cancelled 状态、_SseRunEmitter 的 token 序号、
_map_chunk_to_sse 的 truncated 透传。
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest

# backend/ 与 backend/packages 加入路径（pyproject 仅配置了 packages）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.gateway.sse import (  # noqa: E402
    HEARTBEAT,
    format_heartbeat,
    format_sse_event,
    with_heartbeat,
)
from app.gateway.stream_buffer import RunRecord, StreamBuffer  # noqa: E402
from app.gateway.routers.agent import _SseRunEmitter, _map_chunk_to_sse  # noqa: E402


# ------------------------------------------------------------------
# format_sse_event
# ------------------------------------------------------------------

class TestFormatSseEvent:
    def test_with_event_id_and_index(self):
        frame = format_sse_event("token", "你好", event_id=3, index=1)
        assert frame.startswith("id: 3\n")
        assert "event: token\n" in frame
        assert frame.endswith("\n\n")
        data_line = next(line for line in frame.splitlines() if line.startswith("data: "))
        data = json.loads(data_line[len("data: "):])
        assert data["type"] == "token"
        assert data["content"] == "你好"
        assert data["index"] == 1
        assert data["timestamp"] is not None

    def test_without_optional_fields_keeps_compat(self):
        frame = format_sse_event("done")
        assert "id:" not in frame
        data = json.loads(frame.split("data: ", 1)[1])
        assert data["index"] is None

    def test_cancelled_event_type_allowed(self):
        frame = format_sse_event("cancelled", {"message": "run cancelled"}, event_id=9)
        data = json.loads(frame.split("data: ", 1)[1])
        assert data["type"] == "cancelled"


# ------------------------------------------------------------------
# 心跳帧
# ------------------------------------------------------------------

class TestHeartbeat:
    def test_format(self):
        assert format_heartbeat() == ": ping\n\n"

    def test_with_heartbeat_passes_fast_events(self):
        async def gen():
            for i in range(3):
                yield i

        async def collect():
            return [x async for x in with_heartbeat(gen(), interval=0.1)]

        assert asyncio.run(collect()) == [0, 1, 2]

    def test_with_heartbeat_yields_sentinel_on_gap(self):
        async def gen():
            yield "first"
            await asyncio.sleep(0.1)
            yield "second"

        async def collect():
            return [x async for x in with_heartbeat(gen(), interval=0.02)]

        items = asyncio.run(collect())
        assert items[0] == "first"
        assert items[-1] == "second"
        assert HEARTBEAT in items[1:-1]

    def test_with_heartbeat_ends_on_exhaustion(self):
        async def gen():
            if False:
                yield None

        async def collect():
            return [x async for x in with_heartbeat(gen(), interval=0.01)]

        assert asyncio.run(collect()) == []


# ------------------------------------------------------------------
# stream_buffer cancelled
# ------------------------------------------------------------------

class TestStreamBufferCancel:
    def test_cancel_sets_flags_and_replays(self):
        async def run():
            buf = StreamBuffer()
            record = buf.begin_run("u1", "s1")
            frame = format_sse_event("token", "hi", event_id=record.next_id, index=1)
            await buf.append(record, frame)
            cancelled = format_sse_event("cancelled", {"message": "run cancelled"},
                                         event_id=record.next_id)
            await buf.append(record, cancelled)
            await buf.cancel(record)

            assert record.cancelled is True
            assert record.done is True

            got_record, frames = buf.replay("u1", "s1", 0)
            assert got_record is record
            assert len(frames) == 2
            assert '"cancelled"' in frames[-1][1]

        asyncio.run(run())

    def test_cancel_wakes_followers(self):
        async def run():
            buf = StreamBuffer()
            record = buf.begin_run("u1", "s1")

            async def follower():
                async with record.cond:
                    await asyncio.wait_for(record.cond.wait(), timeout=1.0)
                return record.cancelled

            task = asyncio.create_task(follower())
            await asyncio.sleep(0.05)
            await buf.cancel(record)
            assert await task is True

        asyncio.run(run())


# ------------------------------------------------------------------
# _SseRunEmitter token 序号
# ------------------------------------------------------------------

class TestSseRunEmitter:
    def test_token_index_increments(self):
        async def run():
            buf_record = RunRecord(user_id="u", session_id="s")
            emitter = _SseRunEmitter(buf_record)
            await emitter.emit("start", {"session_id": "s"})
            await emitter.emit("token", "你")
            await emitter.emit("thinking", "想")
            await emitter.emit("token", "好")

            assert emitter.token_index == 2
            frames = [f for _, f in buf_record.events]
            token_frames = [f for f in frames if "event: token" in f]
            indexes = [
                json.loads(f.split("data: ", 1)[1])["index"] for f in token_frames
            ]
            assert indexes == [1, 2]
            # 非 token 事件不带序号
            thinking = next(f for f in frames if "event: thinking" in f)
            assert json.loads(thinking.split("data: ", 1)[1])["index"] is None

        asyncio.run(run())


# ------------------------------------------------------------------
# _map_chunk_to_sse
# ------------------------------------------------------------------

class TestMapChunkToSse:
    def test_tool_result_truncated_passthrough(self):
        _, content = _map_chunk_to_sse({
            "type": "tool_result", "name": "rag_retrieve",
            "content": "x" * 500, "truncated": True,
        })
        assert content == {"name": "rag_retrieve", "content": "x" * 500, "truncated": True}

    def test_tool_result_truncated_defaults_false(self):
        _, content = _map_chunk_to_sse({
            "type": "tool_result", "name": "t", "content": "ok",
        })
        assert content["truncated"] is False

    def test_token_and_thinking(self):
        assert _map_chunk_to_sse({"type": "token", "content": "a"}) == ("token", "a")
        assert _map_chunk_to_sse({"type": "thinking", "content": "b"}) == ("thinking", "b")

    def test_approval_required_keeps_shape(self):
        chunk = {"type": "approval_required", "session_id": "s",
                 "tool_calls": [], "message": "m"}
        assert _map_chunk_to_sse(chunk) == ("approval_required", chunk)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
