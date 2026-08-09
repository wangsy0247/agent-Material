"""SSE 帧格式化工具（agent / rag 路由共用）。

帧格式为 id/event/data 三段式（借鉴 LangGraph Platform 协议）：
- id：run 内单调递增，供前端 Last-Event-ID 断点续传（可选）
- event：与 data 内 type 一致，便于原生 EventSource / 日志排查
- data：SSEEvent JSON，前端解析以此为准（向后兼容旧前端）

另提供注释心跳帧（`: ping`），用于 LLM 思考/工具执行等长时间无事件
时防止中间代理/浏览器判死断连。心跳帧无 id/event/data，不进 run 缓冲，
符合 SSE 规范的客户端（含本项目 stream_client.js）会自动忽略。
"""

import asyncio
import time
from collections.abc import AsyncIterator
from typing import TypeVar

from .schemas import SSEEvent

_T = TypeVar("_T")

#: 心跳间隔（秒）：超过该时长无任何事件时补发注释帧
HEARTBEAT_INTERVAL = 15.0


class _Heartbeat:
    """with_heartbeat 产出的心跳标记哨兵（与真实事件区分，避免类型冲突）。"""


HEARTBEAT = _Heartbeat()


def format_sse_event(
    event_type: str,
    content: str | dict | list | None = None,
    event_id: int | None = None,
    index: int | None = None,
) -> str:
    """将数据格式化为 SSE 事件字符串（id/event/data 三段式）。

    event_id 为 None 时省略 id 行（保持非流式/其它调用兼容）。
    index 仅用于 token 事件序号（乱序/丢帧检测），为 None 时省略。
    """
    event = SSEEvent(type=event_type, content=content, index=index, timestamp=time.time())
    lines = ""
    if event_id is not None:
        lines += f"id: {event_id}\n"
    lines += f"event: {event_type}\n"
    lines += f"data: {event.model_dump_json()}\n\n"
    return lines


def format_heartbeat() -> str:
    """SSE 注释心跳帧（`: ping`），客户端按规范忽略。"""
    return ": ping\n\n"


async def with_heartbeat(
    agen: AsyncIterator[_T],
    interval: float = HEARTBEAT_INTERVAL,
) -> AsyncIterator[_T | _Heartbeat]:
    """包装异步迭代器：事件间隔超过 interval 时先 yield HEARTBEAT 哨兵。

    调用方按 `item is HEARTBEAT` 区分：心跳时向客户端发 format_heartbeat()
    注释帧（不进 run 缓冲），其余为真实事件正常处理。
    迭代器耗尽（StopAsyncIteration）后本生成器正常结束。

    实现注意：挂起的 __anext__ 在整个超时期间保持不取消（用 asyncio.wait
    竞速而非 wait_for），否则心跳超时会误杀被包装流中正在进行的工作
    （如 LLM 调用）。
    """
    it = agen.__aiter__()
    pending = asyncio.ensure_future(it.__anext__())
    try:
        while True:
            done, _ = await asyncio.wait({pending}, timeout=interval)
            if not done:
                # 超时：anext 仍在进行，仅发心跳哨兵，下轮继续等同一个 future
                yield HEARTBEAT
                continue
            try:
                item = pending.result()
            except StopAsyncIteration:
                return
            yield item
            pending = asyncio.ensure_future(it.__anext__())
    finally:
        if not pending.done():
            pending.cancel()
