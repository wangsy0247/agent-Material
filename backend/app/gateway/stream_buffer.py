"""
进程内 run 事件缓冲：为 SSE 断点续传（Last-Event-ID）提供帧回放能力。

设计要点：
- key = (user_id, session_id)，value 为 RunRecord
- 每个 run 内事件 id 从 1 单调递增，帧以完整 SSE 字符串形式存储
- 容量上限 2000 帧/run，超出丢弃最旧帧；TTL 30 分钟惰性清理

注意：本实现为单进程内存缓冲，**仅在单 worker（dev 模式）下有效**。
prod 多 worker 部署时续传可能 miss buffer（resume 返回 404 或仅有旧事件），
需要替换为 Redis 等共享存储（本次不做）。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

MAX_EVENTS_PER_RUN = 2000
RUN_TTL_SECONDS = 30 * 60  # 30 分钟


@dataclass
class RunRecord:
    """单个 run 的事件缓冲记录。"""

    user_id: str
    session_id: str
    events: list[tuple[int, str]] = field(default_factory=list)  # [(event_id, frame_str)]
    done: bool = False
    cancelled: bool = False  # 用户中止/断连取消（终态，resume 回放后不再 follow）
    cond: asyncio.Condition = field(default_factory=asyncio.Condition)
    updated_at: float = field(default_factory=time.time)
    _next_id: int = 1

    @property
    def next_id(self) -> int:
        """下一次 append 将分配的事件 id（供调用方先格式化帧再 append）。"""
        return self._next_id


class StreamBuffer:
    """run 事件缓冲（单进程内有效）。"""

    def __init__(self, max_events: int = MAX_EVENTS_PER_RUN, ttl: float = RUN_TTL_SECONDS):
        self._runs: dict[tuple[str, str], RunRecord] = {}
        self._max_events = max_events
        self._ttl = ttl

    def _purge_expired(self) -> None:
        """惰性清理过期 run。"""
        now = time.time()
        expired = [k for k, r in self._runs.items() if now - r.updated_at > self._ttl]
        for k in expired:
            del self._runs[k]

    def begin_run(self, user_id: str, session_id: str) -> RunRecord:
        """开始一个新的 run：重置该 (user_id, session_id) 的缓冲，事件 id 从 1 递增。"""
        self._purge_expired()
        record = RunRecord(user_id=user_id, session_id=session_id)
        self._runs[(user_id, session_id)] = record
        return record

    async def append(self, record: RunRecord, frame: str) -> int:
        """追加一帧（完整 SSE 帧字符串），返回分配的事件 id。

        容量上限 self._max_events 帧/run，超出时丢弃最旧帧。
        """
        async with record.cond:
            event_id = record._next_id
            record._next_id += 1
            record.events.append((event_id, frame))
            if len(record.events) > self._max_events:
                del record.events[: len(record.events) - self._max_events]
            record.updated_at = time.time()
            record.cond.notify_all()
            return event_id

    async def finish(self, record: RunRecord) -> None:
        """标记 run 结束（done），唤醒所有 follow 等待方。"""
        async with record.cond:
            record.done = True
            record.updated_at = time.time()
            record.cond.notify_all()

    async def cancel(self, record: RunRecord) -> None:
        """标记 run 被取消（用户中止/断连）。

        置 cancelled + done 并唤醒 follow 等待方；调用方应在此之前
        向 buffer 追加 cancelled 帧，使 resume 回放方能收到终结信号。
        """
        async with record.cond:
            record.cancelled = True
            record.done = True
            record.updated_at = time.time()
            record.cond.notify_all()

    def replay(
        self, user_id: str, session_id: str, last_id: int = 0
    ) -> tuple[RunRecord | None, list[tuple[int, str]]]:
        """返回 (record, 需回放的帧列表)；buffer 中无此 run 时 record 为 None。"""
        self._purge_expired()
        record = self._runs.get((user_id, session_id))
        if record is None:
            return None, []
        return record, [(eid, frame) for eid, frame in record.events if eid > last_id]


# 模块级单例
stream_buffer = StreamBuffer()
