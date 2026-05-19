"""
Agent 路由：支持 SSE 流式对话 + HITL + 三层记忆系统集成

记忆集成设计：
1. 自动创建/加载会话（根据 session_id）
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
from api.schemas import (
    AgentChatRequest,
    AgentChatResponse,
    AgentApprovalRequest,
    SSEEvent,
)
from api.dependencies import get_agent_with_memory, get_memory_manager
from utils.logger import logger

if TYPE_CHECKING:
    from memory.manager import MemoryManager
    from agent.React_agent import ReactAgent

router = APIRouter(prefix="/agent", tags=["Agent 对话"])


def format_sse_event(event_type: str, content: str | dict | list | None = None) -> str:
    """将数据格式化为 SSE 事件字符串"""
    event = SSEEvent(type=event_type, content=content, timestamp=time.time())
    return f"data: {event.model_dump_json()}\n\n"


def _ensure_session(mm: "MemoryManager", session_id: str | None) -> str:
    """确保有活跃会话，自动创建或加载（方案1：按 session_id 隔离）"""
    if session_id and session_id != "default":
        # 已经在内存池中，更新活跃状态后直接返回
        if session_id in mm._session_pool:
            mm.session_store.update_session(session_id, is_active=True, end_reason=None)
            return session_id
        # 尝试从 SQLite 加载已有会话
        if mm.load_session(session_id):
            return session_id
        else:
            # 会话不存在，创建新会话
            return mm.create_session(title=f"会话 {session_id[:8]}")

    # 没有指定 session_id（或传了 "default"），创建新会话
    return mm.create_session(title="Agent 对话")


@router.post("/chat", summary="Agent 对话（SSE 流式 + 记忆集成）")
async def agent_chat(
    request: AgentChatRequest,
    agent: "ReactAgent" = Depends(get_agent_with_memory),
    mm: "MemoryManager" = Depends(get_memory_manager),
):
    """
    与材料科学 Agent 进行对话，支持 SSE 流式输出 + 三层记忆系统 + HITL

    **记忆集成说明：**
    - 自动创建/管理会话（根据 session_id）
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

    # 确保会话
    active_session_id = _ensure_session(mm, request.session_id)

    # 非流式模式：直接返回完整结果
    if not request.stream:
        try:
            # 注入记忆状态信息
            memory_info = []
            if mm.long_term:
                if mm.long_term.user_file.exists():
                    memory_info.append("已加载用户画像")
                if mm.long_term.project_file.exists():
                    memory_info.append("已加载项目知识")

            result = await agent.excutor(
                request.question,
                session_id=active_session_id,
            )

            # HITL 中断返回事件字典
            if isinstance(result, dict):
                return {
                    "code": 202,
                    "message": "等待人工确认",
                    "data": result,
                }

            return AgentChatResponse(
                answer=result,
                session_id=active_session_id,
            )
        except Exception as e:
            logger.error(f"[Agent] 非流式调用失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # 流式模式：SSE
    async def event_stream():
        """生成 SSE 事件流"""
        logger.info(
            f"[Agent SSE] 新请求 | session={active_session_id} | "
            f"question={request.question[:50]}..."
        )

        # 发送启动事件
        yield format_sse_event("start", {"session_id": active_session_id})

        # 发送记忆注入状态
        memory_status = []
        if mm.long_term:
            if mm.long_term.user_file.exists():
                memory_status.append("用户画像已加载")
            if mm.long_term.project_file.exists():
                memory_status.append("项目知识已加载")
        if memory_status:
            yield format_sse_event("status", {"memory": memory_status})

        # 发送上下文统计（从指定会话的短期记忆中获取）
        session_stats = mm.get_session_stats(active_session_id)
        yield format_sse_event("status", {"context": session_stats})

        full_answer = ""

        try:
            # 使用 Agent 的流式执行（已集成记忆）
            async for chunk in agent.excutor_stream(
                request.question,
                session_id=active_session_id,
            ):
                if isinstance(chunk, dict):
                    # HITL 中断事件
                    yield format_sse_event("approval_required", chunk)
                    # HITL 中断后直接结束，不发送 done（等待 approve 恢复）
                    return
                elif chunk:
                    full_answer += chunk
                    yield format_sse_event("token", chunk)

            # 正常结束
            done_stats = mm.get_session_stats(active_session_id)
            yield format_sse_event("done", {
                "session_id": active_session_id,
                "total_tokens": done_stats["total_tokens"],
            })
            logger.info(
                f"[Agent SSE] 请求完成 | session={active_session_id} | "
                f"answer_len={len(full_answer)}"
            )

        except Exception as e:
            logger.error(f"[Agent SSE] 请求失败 | session={active_session_id} | error={e}")
            yield format_sse_event("error", {"message": str(e), "type": type(e).__name__})
            yield format_sse_event("done")

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

    # 流式恢复
    async def event_stream():
        session_id = request.session_id
        logger.info(
            f"[Agent Approve] HITL 恢复 | session={session_id} | "
            f"action={request.action}"
        )

        yield format_sse_event("start", {"session_id": session_id, "resumed": True})

        full_answer = ""

        try:
            async for chunk in agent.resume_stream(session_id, approval_result):
                if isinstance(chunk, dict):
                    # 再次遇到 HITL（理论上不太可能）
                    yield format_sse_event("approval_required", chunk)
                    return
                elif chunk:
                    full_answer += chunk
                    yield format_sse_event("token", chunk)

            yield format_sse_event("done", {"session_id": session_id})
            logger.info(
                f"[Agent Approve] 恢复完成 | session={session_id} | "
                f"answer_len={len(full_answer)}"
            )

        except Exception as e:
            logger.error(
                f"[Agent Approve] 恢复失败 | session={session_id} | error={e}"
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
