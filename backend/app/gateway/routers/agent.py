"""
Agent 路由：支持 SSE 流式对话 + HITL + 三层记忆系统集成（按用户隔离）

记忆集成设计：
1. 自动创建/加载会话（根据 session_id + user_id）
2. 用户消息自动保存到短期记忆 + SQLite
3. Agent 调用时注入长期记忆（USER.md / MEMORY.md）到系统提示
4. 助手回复自动保存到记忆系统
5. Token 超限时自动触发上下文压缩

HITL 设计：
1. 敏感操作触发 approval_required SSE 事件
2. 前端确认后调用 /approve 恢复执行

SSE 协议补充（对齐 docs/FASTAPI 事件分类法）：
- token 事件带 index 序号（run 内从 1 递增，前端乱序/丢帧检测）
- 首个 token 前补发 status {"phase": "generating"} 阶段事件
- 长时间无事件时发注释心跳帧（`: ping`，不进 run 缓冲）
- 客户端断连/中止时向 run 缓冲追加 cancelled 帧（终态），
  使 /chat/resume 的 follow 方能收到终结信号而不空等
"""

import asyncio
import time
from typing import TYPE_CHECKING
from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from ..schemas import (
    AgentChatRequest,
    AgentChatResponse,
    AgentApprovalRequest,
)
from ..sse import HEARTBEAT, format_heartbeat, format_sse_event, with_heartbeat
from ..stream_buffer import RunRecord, stream_buffer
from ..auth.dependencies import get_agent_with_memory, get_memory_manager, get_current_user
from ..auth.schemas import UserInfo
from agent_material.observability.structured_logger import get_logger, set_request_context
from agent_material.observability.metrics import (
    http_request_total,
    http_request_duration,
    http_request_in_flight,
    agent_invocation_total,
    agent_duration,
    StreamMetrics,
)

logger = get_logger("router.agent")

if TYPE_CHECKING:
    from agent_material.memory.manager import MemoryManager
    from agent_material.agents.React_agent import ReactAgent

router = APIRouter(prefix="/agent", tags=["Agent 对话"])


class AgentResumeRequest(BaseModel):
    """断点续传请求（配合 Last-Event-ID 请求头）"""
    session_id: str = Field(..., description="会话ID（与中断时的 run 一致）")


class _SseRunEmitter:
    """run 内统一的 SSE 帧发射：格式化 + run 缓冲追加 + token 序号。

    token 事件的 data.index 从 1 单调递增，供前端做乱序/丢帧检测。
    /chat 与 /approve 两条流共用，避免三处复制 emit 闭包。
    """

    def __init__(self, record: RunRecord):
        self._record = record
        self.token_index = 0

    async def emit(self, event_type: str, content=None) -> str:
        index = None
        if event_type == "token":
            self.token_index += 1
            index = self.token_index
        frame = format_sse_event(
            event_type, content, event_id=self._record.next_id, index=index
        )
        await stream_buffer.append(self._record, frame)
        return frame


def _map_chunk_to_sse(chunk: dict) -> tuple[str, object]:
    """将 ReactAgent 产出的结构化事件 dict 映射为 (event_type, content)。

    event_type 直接由 chunk["type"] 映射：
    token/thinking/tool_call/tool_result/approval_required
    """
    event_type = chunk.get("type", "token")
    if event_type in ("token", "thinking"):
        return event_type, chunk.get("content", "")
    if event_type == "tool_call":
        return event_type, {"name": chunk.get("name"), "args": chunk.get("args")}
    if event_type == "tool_result":
        return event_type, {
            "name": chunk.get("name"),
            "content": chunk.get("content", ""),
            "truncated": chunk.get("truncated", False),
        }
    # approval_required：保持原 dict 形状（type/session_id/tool_calls/message）
    return event_type, chunk


def _make_session_title(question: str, max_len: int = 20) -> str:
    """首问截断作为会话标题：压缩空白，超长截断加省略号。"""
    text = " ".join(question.split())
    if not text:
        return "新会话"
    if len(text) > max_len:
        return text[:max_len] + "…"
    return text


def _is_default_title(title: str | None) -> bool:
    """判断是否为系统默认/占位标题（首问时可被替换）。"""
    if not title:
        return True
    return title.startswith("会话 ") or title in ("Agent 对话", "新会话")


def _ensure_session(mm: "MemoryManager", session_id: str | None, user_id: str, title_hint: str | None = None) -> str:
    """确保有活跃会话，自动创建或加载（按 user_id 隔离）。

    title_hint：首问截断标题。新建会话时直接用作标题；已有会话仍是默认
    标题且尚无消息（message_count == 0，即本次为首问）时替换为首问标题。
    """
    if session_id and session_id != "default":
        # 已经在内存池中，更新活跃状态后直接返回
        if (user_id, session_id) in mm._session_pool:
            updates: dict = {"is_active": True, "end_reason": None}
            if title_hint:
                info = mm.session_store.get_session(session_id, user_id=user_id)
                if info and info.message_count == 0 and _is_default_title(info.title):
                    updates["title"] = title_hint
            mm.session_store.update_session(session_id, user_id=user_id, **updates)
            return session_id
        # 尝试从 SQLite 加载已有会话
        if mm.load_session(session_id, user_id=user_id):
            if title_hint:
                info = mm.session_store.get_session(session_id, user_id=user_id)
                if info and info.message_count == 0 and _is_default_title(info.title):
                    mm.session_store.update_session(session_id, user_id=user_id, title=title_hint)
            return session_id
        else:
            # 会话不存在，创建新会话
            return mm.create_session(user_id=user_id, title=title_hint or f"会话 {session_id[:8]}")

    # 没有指定 session_id（或传了 "default"），创建新会话
    return mm.create_session(user_id=user_id, title=title_hint or "Agent 对话")


@router.post("/chat", summary="Agent 对话（SSE 流式 + 记忆集成）")
async def agent_chat(
    request: AgentChatRequest,
    current_user: UserInfo = Depends(get_current_user),
    agent: "ReactAgent" = Depends(get_agent_with_memory),
    mm: "MemoryManager" = Depends(get_memory_manager),
):
    """
    与材料科学 Agent 进行对话，支持 SSE 流式输出 + 三层记忆系统 + HITL

    **记忆集成说明：**
    - 自动创建/管理会话（根据 session_id，按用户隔离）
    - 长期记忆（USER.md / MEMORY.md）自动注入系统提示
    - 对话历史自动保存到 SQLite
    - Token 超限时自动压缩上下文

    **HITL 说明：**
    - 敏感操作（如 mp_download）会触发 `approval_required` SSE 事件
    - 前端需调用 `/api/v1/agent/approve` 恢复执行

    **流式事件类型说明：**
    - `start`: 开始处理请求，包含 session_id
    - `status`: 记忆注入状态 / 阶段切换（{"phase": "generating"}）
    - `thinking`: Agent 思考过程
    - `tool_call`: 调用外部工具
    - `tool_result`: 工具返回结果（含 truncated 截断标记）
    - `token`: 大模型实时生成的 token（data.index 从 1 递增）
    - `approval_required`: HITL 敏感操作需人工确认
    - `done`: 流结束标记，包含 session_id
    - `error`: 发生错误
    - `cancelled`: 客户端断连/中止导致的取消（终态）
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    user_id = current_user.user_id
    set_request_context(request_id="", session_id=request.session_id or "", user_id=user_id)

    # 确保会话（首问截断作为新会话/默认标题会话的标题）
    active_session_id = _ensure_session(
        mm, request.session_id, user_id, title_hint=_make_session_title(request.question)
    )
    endpoint = "/api/v1/agent/chat"
    http_request_in_flight.labels(method="POST").inc()
    t_req = time.time()

    # 非流式模式：直接返回完整结果
    if not request.stream:
        try:
            result = await agent.excutor(
                request.question,
                session_id=active_session_id,
                user_id=user_id,
            )
            agent_duration.labels(mode="invoke").observe(time.time() - t_req)

            if isinstance(result, dict):
                agent_invocation_total.labels(mode="invoke", status="hitl_interrupt").inc()
                http_request_total.labels(method="POST", endpoint=endpoint, status_code=202).inc()
                return {"code": 202, "message": "等待人工确认", "data": result}

            agent_invocation_total.labels(mode="invoke", status="success").inc()
            http_request_total.labels(method="POST", endpoint=endpoint, status_code=200).inc()
            return AgentChatResponse(answer=result, session_id=active_session_id)
        except Exception as e:
            agent_invocation_total.labels(mode="invoke", status="error").inc()
            http_request_total.labels(method="POST", endpoint=endpoint, status_code=500).inc()
            logger.error("agent_chat_failed", extra={"error": str(e)}, exc_info=True)
            raise HTTPException(status_code=500, detail="Agent 执行失败")
        finally:
            http_request_duration.labels(method="POST", endpoint=endpoint).observe(time.time() - t_req)
            http_request_in_flight.labels(method="POST").dec()

    # 流式模式：SSE（接入 stream_buffer 支持 Last-Event-ID 断点续传）
    sm = StreamMetrics(mode="agent_chat")
    record = stream_buffer.begin_run(user_id, active_session_id)

    async def event_stream():
        # status_code 记录真实结果：success=200 / error=500 / hitl=202
        status_code = 200
        emitter = _SseRunEmitter(record)

        logger.info("agent_sse_start", extra={
            "user_id": user_id, "session_id": active_session_id,
            "question": request.question[:100],
        })
        yield await emitter.emit("start", {"session_id": active_session_id})

        memory_status = []
        lt = mm._get_long_term(user_id)
        if lt:
            if lt.user_file.exists():
                memory_status.append("用户画像已加载")
            if lt.project_file.exists():
                memory_status.append("项目知识已加载")
        if memory_status:
            yield await emitter.emit("status", {"memory": memory_status})

        session_stats = mm.get_session_stats(active_session_id, user_id=user_id)
        yield await emitter.emit("status", {"context": session_stats})

        full_answer = ""
        generating_marked = False
        try:
            # with_heartbeat 包在 StreamMetrics 外层：心跳哨兵不计入 token 指标
            stream = with_heartbeat(sm.awrap(agent.excutor_stream(
                request.question,
                session_id=active_session_id,
                user_id=user_id,
            )))
            async for chunk in stream:
                if chunk is HEARTBEAT:
                    # 注释心跳帧：直接发客户端，不进 run 缓冲（无 id、无回放价值）
                    yield format_heartbeat()
                    continue
                event_type, content = _map_chunk_to_sse(chunk)
                if event_type == "approval_required":
                    agent_invocation_total.labels(mode="stream", status="hitl_interrupt").inc()
                    status_code = 202
                    yield await emitter.emit("approval_required", content)
                    await stream_buffer.finish(record)
                    return
                if event_type == "token":
                    if not generating_marked:
                        generating_marked = True
                        yield await emitter.emit("status", {"phase": "generating"})
                    full_answer += content
                yield await emitter.emit(event_type, content)

            done_stats = mm.get_session_stats(active_session_id, user_id=user_id)
            yield await emitter.emit("done", {
                "session_id": active_session_id,
                "total_tokens": done_stats["total_tokens"],
            })
            agent_invocation_total.labels(mode="stream", status="success").inc()
            logger.info("agent_sse_done", extra={
                "user_id": user_id, "session_id": active_session_id,
                "answer_len": len(full_answer),
                "first_token_latency": sm.first_token_latency,
                "total_chunks": sm._chunk_count,
            })
        except asyncio.CancelledError:
            # 客户端断连/中止：向 run 缓冲追加 cancelled 帧（终态），
            # 使 /chat/resume 的 follow 方收到终结信号而不空等
            logger.info("agent_sse_cancelled", extra={
                "user_id": user_id, "session_id": active_session_id,
            })
            await emitter.emit("cancelled", {"message": "run cancelled"})
            await stream_buffer.cancel(record)
            raise
        except Exception as e:
            agent_invocation_total.labels(mode="stream", status="error").inc()
            status_code = 500
            logger.error("agent_sse_failed", extra={
                "user_id": user_id, "session_id": active_session_id,
                "error": str(e), "error_type": type(e).__name__,
            }, exc_info=True)
            # error 为终态：只发 error 帧，不再补发 done（done/error 互斥）
            yield await emitter.emit("error", {"message": str(e), "type": type(e).__name__})
        finally:
            await stream_buffer.finish(record)
            # agent_duration 由 React_agent.excutor_stream 内 observe（此处不重复）
            http_request_total.labels(method="POST", endpoint=endpoint, status_code=status_code).inc()
            http_request_duration.labels(method="POST", endpoint=endpoint).observe(time.time() - t_req)
            http_request_in_flight.labels(method="POST").dec()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat/resume", summary="SSE 断点续传（Last-Event-ID 回放 + follow）")
async def agent_chat_resume(
    request: AgentResumeRequest,
    current_user: UserInfo = Depends(get_current_user),
    last_event_id: int = Header(default=0),
):
    """从 run 事件缓冲回放 id 大于 Last-Event-ID 的帧；run 未结束时挂起 follow
    新事件，直到 done/error/cancelled 帧发出后关流。buffer 中无此 run 时返回 404。

    注意：stream_buffer 为进程内实现，仅单 worker（dev 模式）有效。
    """
    user_id = current_user.user_id
    record, frames = stream_buffer.replay(user_id, request.session_id, last_event_id)
    if record is None:
        raise HTTPException(status_code=404, detail="没有可续传的 run（buffer 中不存在或已过期）")

    async def event_stream():
        seen = last_event_id
        for fid, frame in frames:
            yield frame
            seen = fid
        if record.done:
            return
        # run 未结束：挂起 follow 新事件，done/error/cancelled 帧（finish/cancel
        # 置 done 前已 append）发出后关流
        async with record.cond:
            while True:
                pending = [(fid, f) for fid, f in record.events if fid > seen]
                for fid, frame in pending:
                    yield frame
                    seen = fid
                if record.done:
                    return
                await record.cond.wait()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/approve", summary="恢复被 HITL 挂起的 Agent 执行")
async def agent_approve(
    request: AgentApprovalRequest,
    current_user: UserInfo = Depends(get_current_user),
    agent: "ReactAgent" = Depends(get_agent_with_memory),
):
    """
    对 HITL 中断的敏感操作进行人工确认，恢复 Agent 执行。

    **请求示例：**
    ```json
    {
        "session_id": "uuid-xxxx",
        "action": "approve",
        "comment": "确认下载"
    }
    ```

    **响应：**
    - 流式模式：返回 SSE 流，与 `/chat` 格式一致
    - 非流式模式：返回完整回答
    """
    approval_result = {
        "action": request.action,
        "comment": request.comment or "",
    }

    user_id = current_user.user_id
    session_id = request.session_id

    # 接入 stream_buffer：恢复执行视为一个新 run（id 重新从 1 递增）
    record = stream_buffer.begin_run(user_id, session_id)

    # 流式恢复
    async def event_stream():
        emitter = _SseRunEmitter(record)

        logger.info(
            f"[Agent Approve] HITL 恢复 | user={user_id} | session={session_id} | "
            f"action={request.action}"
        )

        yield await emitter.emit("start", {"session_id": session_id, "resumed": True})

        full_answer = ""
        generating_marked = False

        try:
            async for chunk in with_heartbeat(
                agent.resume_stream(session_id, approval_result, user_id=user_id)
            ):
                if chunk is HEARTBEAT:
                    yield format_heartbeat()
                    continue
                event_type, content = _map_chunk_to_sse(chunk)
                if event_type == "approval_required":
                    # 再次遇到 HITL（理论上不太可能）
                    yield await emitter.emit("approval_required", content)
                    return
                if event_type == "token":
                    if not generating_marked:
                        generating_marked = True
                        yield await emitter.emit("status", {"phase": "generating"})
                    full_answer += content
                yield await emitter.emit(event_type, content)

            yield await emitter.emit("done", {"session_id": session_id})
            logger.info(
                f"[Agent Approve] 恢复完成 | user={user_id} | session={session_id} | "
                f"answer_len={len(full_answer)}"
            )

        except asyncio.CancelledError:
            logger.info(
                f"[Agent Approve] 恢复被取消 | user={user_id} | session={session_id}"
            )
            await emitter.emit("cancelled", {"message": "run cancelled"})
            await stream_buffer.cancel(record)
            raise
        except Exception as e:
            logger.error(
                f"[Agent Approve] 恢复失败 | user={user_id} | session={session_id} | error={e}"
            )
            # error 为终态：只发 error 帧，不再补发 done（done/error 互斥）
            yield await emitter.emit("error", {"message": str(e), "type": type(e).__name__})
        finally:
            await stream_buffer.finish(record)

    # 如果请求头中有 Accept: text/event-stream，返回 SSE
    # 否则返回非流式结果
    # 这里简化处理：默认返回 SSE（因为前端通常使用 SSE）
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
