"""
Agent 路由：支持 SSE 流式对话 + 三层记忆系统集成

记忆集成设计：
1. 自动创建/加载会话（根据 session_id）
2. 用户消息自动保存到短期记忆 + SQLite
3. Agent 调用时注入长期记忆（USER.md / MEMORY.md）到系统提示
4. 助手回复自动保存到记忆系统
5. Token 超限时自动触发上下文压缩
"""

import json
import time
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, ToolMessage
from api.schemas import AgentChatRequest, AgentChatResponse, SSEEvent
from api.dependencies import get_agent_with_memory, get_memory_manager
from memory.manager import MemoryManager
from agent.React_agent import ReactAgent
from utils.logger import logger

router = APIRouter(prefix="/agent", tags=["Agent 对话"])


def format_sse_event(event_type: str, content: str | dict | list | None = None) -> str:
    """将数据格式化为 SSE 事件字符串"""
    event = SSEEvent(type=event_type, content=content, timestamp=time.time())
    return f"data: {event.model_dump_json()}\n\n"


def _ensure_session(mm: MemoryManager, session_id: str | None) -> str:
    """确保有活跃会话，自动创建或加载"""
    if session_id and session_id != "default":
        # 尝试加载已有会话
        if mm.load_session(session_id):
            return session_id
        else:
            # 会话不存在，创建新会话
            return mm.create_session(title=f"会话 {session_id[:8]}")
    
    # 没有指定 session_id，检查是否有当前会话
    if mm._current_session_id:
        return mm._current_session_id
    
    # 创建新会话
    return mm.create_session(title="Agent 对话")


@router.post("/chat", summary="Agent 对话（SSE 流式 + 记忆集成）")
async def agent_chat(
    request: AgentChatRequest,
    agent: ReactAgent = Depends(get_agent_with_memory),
    mm: MemoryManager = Depends(get_memory_manager),
):
    """
    与材料科学 Agent 进行对话，支持 SSE 流式输出 + 三层记忆系统
    
    **记忆集成说明：**
    - 自动创建/管理会话（根据 session_id）
    - 长期记忆（USER.md / MEMORY.md）自动注入系统提示
    - 对话历史自动保存到 SQLite
    - Token 超限时自动压缩上下文
    
    **流式事件类型说明：**
    - `start`: 开始处理请求，包含 session_id
    - `status`: 记忆注入状态（如"已加载用户画像"）
    - `thinking`: Agent 思考过程
    - `tool_call`: 调用外部工具
    - `tool_result`: 工具返回结果
    - `token`: 大模型实时生成的 token
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
            
            full_answer = agent.excutor(
                request.question,
                session_id=active_session_id,
            )
            
            return AgentChatResponse(
                answer=full_answer,
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
        
        # 发送上下文统计
        yield format_sse_event("status", {
            "context": {
                "messages_in_history": len(mm.short_term.messages),
                "total_tokens": mm.short_term.total_tokens,
                "compression_count": mm.short_term._context.compression_count,
            }
        })
        
        full_answer = ""
        
        try:
            # 使用 Agent 的流式执行（已集成记忆）
            for chunk in agent.excutor_stream(
                request.question,
                session_id=active_session_id,
            ):
                if chunk:
                    full_answer += chunk
                    yield format_sse_event("token", chunk)
            
            # 正常结束
            yield format_sse_event("done", {
                "session_id": active_session_id,
                "total_tokens": mm.short_term.total_tokens,
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
