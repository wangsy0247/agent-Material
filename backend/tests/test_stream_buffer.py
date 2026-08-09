"""StreamBuffer（SSE 断点续传缓冲）的单元测试。

覆盖：事件 id 单调递增、容量上限丢弃最旧帧、replay 的 last_id 过滤、
未命中返回 None、begin_run 同 key 重置、TTL 惰性过期、finish 语义。
"""

import asyncio
import sys
import time
from pathlib import Path

import pytest

# backend/ 与 backend/packages 加入路径（pyproject 仅配置了 packages）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.gateway.sse import format_sse_event  # noqa: E402
from app.gateway.stream_buffer import StreamBuffer  # noqa: E402


def _frame(record, text):
    return format_sse_event("token", text, event_id=record.next_id)


class TestAppend:
    def test_event_ids_monotonic(self):
        async def run():
            buf = StreamBuffer()
            record = buf.begin_run("u1", "s1")
            ids = [await buf.append(record, _frame(record, t)) for t in "abc"]
            assert ids == [1, 2, 3]
            assert record.next_id == 4

        asyncio.run(run())

    def test_capacity_evicts_oldest(self):
        async def run():
            buf = StreamBuffer(max_events=3)
            record = buf.begin_run("u1", "s1")
            for t in "abcde":
                await buf.append(record, _frame(record, t))
            kept = [eid for eid, _ in record.events]
            assert kept == [3, 4, 5]  # 最旧的 1、2 被丢弃
            assert len(record.events) == 3

        asyncio.run(run())


class TestReplay:
    def test_filters_by_last_id(self):
        async def run():
            buf = StreamBuffer()
            record = buf.begin_run("u1", "s1")
            for t in "abc":
                await buf.append(record, _frame(record, t))
            _, frames = buf.replay("u1", "s1", last_id=2)
            assert [eid for eid, _ in frames] == [3]

        asyncio.run(run())

    def test_missing_run_returns_none(self):
        buf = StreamBuffer()
        record, frames = buf.replay("nobody", "nothing", 0)
        assert record is None
        assert frames == []

    def test_replay_after_finish_keeps_frames(self):
        """finish 只标记终态，不影响已缓冲帧的回放"""
        async def run():
            buf = StreamBuffer()
            record = buf.begin_run("u1", "s1")
            for t in "ab":
                await buf.append(record, _frame(record, t))
            await buf.finish(record)
            got, frames = buf.replay("u1", "s1", 0)
            assert got is record
            assert len(frames) == 2
            assert record.done is True

        asyncio.run(run())


class TestBeginRunReset:
    def test_same_key_resets_buffer(self):
        async def run():
            buf = StreamBuffer()
            old = buf.begin_run("u1", "s1")
            await buf.append(old, _frame(old, "x"))
            await buf.append(old, _frame(old, "y"))

            new = buf.begin_run("u1", "s1")
            assert new is not old
            assert new.events == []
            assert new.next_id == 1  # id 重新从 1 开始
            _, frames = buf.replay("u1", "s1", 0)
            assert frames == []  # 旧 run 的帧不可见

        asyncio.run(run())


class TestTtlExpiry:
    def test_expired_run_purged(self):
        buf = StreamBuffer(ttl=60.0)
        record = buf.begin_run("u1", "s1")
        record.updated_at = time.time() - 120.0  # 手动改旧
        got, frames = buf.replay("u1", "s1", 0)
        assert got is None
        assert frames == []

    def test_fresh_run_not_purged(self):
        buf = StreamBuffer(ttl=60.0)
        buf.begin_run("u1", "s1")
        got, _ = buf.replay("u1", "s1", 0)
        assert got is not None


class TestFinish:
    def test_finish_wakes_followers(self):
        async def run():
            buf = StreamBuffer()
            record = buf.begin_run("u1", "s1")

            async def follower():
                async with record.cond:
                    await asyncio.wait_for(record.cond.wait(), timeout=1.0)
                return record.done

            task = asyncio.create_task(follower())
            await asyncio.sleep(0.05)
            await buf.finish(record)
            assert await task is True

        asyncio.run(run())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
