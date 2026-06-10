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
"""

import json
import time
from typing import TYPE_CHECKING
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, ToolMessage
from ..schemas import (
    AgentChatRequest,
    AgentChatResponse,
    AgentApprovalRequest,
    SSEEvent,
)
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


def format_sse_event(event_type: str, content: str | dict | list | None = None) -> str:
    """将数据格式化为 SSE 事件字符串"""
    event = SSEEvent(type=event_type, content=content, timestamp=time.time())
    return f"data: {event.model_dump_json()}\n\n"


def _ensure_session(mm: "MemoryManager", session_id: str | None, user_id: str) -> str:
    """确保有活跃会话，自动创建或加载（按 user_id 隔离）"""
    if session_id and session_id != "default":
        # 已经在内存池中，更新活跃状态后直接返回
        if (user_id, session_id) in mm._session_pool:
            mm.session_store.update_session(session_id, user_id=user_id, is_active=True, end_reason=None)
            return session_id
        # 尝试从 SQLite 加载已有会话
        if mm.load_session(session_id, user_id=user_id):
            return session_id
        else:
            # 会话不存在，创建新会话
            return mm.create_session(user_id=user_id, title=f"会话 {session_id[:8]}")

    # 没有指定 session_id（或传了 "default"），创建新会话
    return mm.create_session(user_id=user_id, title="Agent 对话")


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
    - `status`: 记忆注入状态（如"已加载用户画像"）
    - `thinking`: Agent 思考过程
    - `tool_call`: 调用外部工具
    - `tool_result`: 工具返回结果
    - `token`: 大模型实时生成的 token
    - `approval_required`: HITL 敏感操作需人工确认
    - `done`: 流结束标记，包含 session_id
    - `error`: 发生错误
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    user_id = current_user.user_id
    set_request_context(request_id="", session_id=request.session_id or "", user_id=user_id)

    # 确保会话
    active_session_id = _ensure_session(mm, request.session_id, user_id)
    mode = "invoke" if not request.stream else "stream"
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
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            http_request_duration.labels(method="POST", endpoint=endpoint).observe(time.time() - t_req)
            http_request_in_flight.labels(method="POST").dec()

    # 流式模式：SSE
    sm = StreamMetrics(mode="agent_chat")

    async def event_stream():
        logger.info("agent_sse_start", extra={
            "user_id": user_id, "session_id": active_session_id,
            "question": request.question[:100],
        })
        yield format_sse_event("start", {"session_id": active_session_id})

        memory_status = []
        lt = mm._get_long_term(user_id)
        if lt:
            if lt.user_file.exists():
                memory_status.append("用户画像已加载")
            if lt.project_file.exists():
                memory_status.append("项目知识已加载")
        if memory_status:
            yield format_sse_event("status", {"memory": memory_status})

        session_stats = mm.get_session_stats(active_session_id, user_id=user_id)
        yield format_sse_event("status", {"context": session_stats})

        full_answer = ""
        try:
            async for chunk in sm.awrap(agent.excutor_stream(
                request.question,
                session_id=active_session_id,
                user_id=user_id,
            )):
                if isinstance(chunk, dict):
                    agent_invocation_total.labels(mode="stream", status="hitl_interrupt").inc()
                    yield format_sse_event("approval_required", chunk)
                    return
                elif chunk:
                    full_answer += chunk
                    yield format_sse_event("token", chunk)

            done_stats = mm.get_session_stats(active_session_id, user_id=user_id)
            yield format_sse_event("done", {
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
        except Exception as e:
            agent_invocation_total.labels(mode="stream", status="error").inc()
            logger.error("agent_sse_failed", extra={
                "user_id": user_id, "session_id": active_session_id,
                "error": str(e), "error_type": type(e).__name__,
            }, exc_info=True)
            yield format_sse_event("error", {"message": str(e), "type": type(e).__name__})
            yield format_sse_event("done")
        finally:
            agent_duration.labels(mode="stream").observe(time.time() - t_req)
            http_request_total.labels(method="POST", endpoint=endpoint, status_code=200).inc()
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

    # 流式恢复
    async def event_stream():
        logger.info(
            f"[Agent Approve] HITL 恢复 | user={user_id} | session={session_id} | "
            f"action={request.action}"
        )

        yield format_sse_event("start", {"session_id": session_id, "resumed": True})

        full_answer = ""

        try:
            async for chunk in agent.resume_stream(session_id, approval_result, user_id=user_id):
                if isinstance(chunk, dict):
                    # 再次遇到 HITL（理论上不太可能）
                    yield format_sse_event("approval_required", chunk)
                    return
                elif chunk:
                    full_answer += chunk
                    yield format_sse_event("token", chunk)

            yield format_sse_event("done", {"session_id": session_id})
            logger.info(
                f"[Agent Approve] 恢复完成 | user={user_id} | session={session_id} | "
                f"answer_len={len(full_answer)}"
            )

        except Exception as e:
            logger.error(
                f"[Agent Approve] 恢复失败 | user={user_id} | session={session_id} | error={e}"
            )
            yield format_sse_event("error", {"message": str(e), "type": type(e).__name__})
            yield format_sse_event("done")

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
