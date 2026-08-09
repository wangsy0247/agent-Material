"""
记忆系统路由：会话管理、记忆召回、记忆操作（按用户隔离）
"""

from typing import TYPE_CHECKING
from fastapi import APIRouter, Depends, HTTPException
from ..schemas import (
    SessionCreateRequest,
    SessionInfoResponse,
    SessionListResponse,
    MemoryRecallRequest,
    MemoryEntryResponse,
    MemoryRecallResponse,
    MemoryStatsResponse,
    BaseResponse,
)
from ..auth.dependencies import get_memory_manager, get_current_user
from ..auth.schemas import UserInfo

if TYPE_CHECKING:
    from agent_material.memory.manager import MemoryManager

router = APIRouter(prefix="/memory", tags=["记忆系统"])


# ==================== 会话管理 ====================

@router.post("/sessions", response_model=BaseResponse, summary="创建新会话")
async def create_session(
    request: SessionCreateRequest,
    current_user: UserInfo = Depends(get_current_user),
    mm: "MemoryManager" = Depends(get_memory_manager),
):
    """创建一个新的对话会话"""
    session_id = mm.create_session(user_id=current_user.user_id, title=request.title, source=request.source)
    return BaseResponse(code=200, message="会话创建成功", data={"session_id": session_id})


@router.get("/sessions", response_model=SessionListResponse, summary="列出会话")
async def list_sessions(
    limit: int = 20,
    current_user: UserInfo = Depends(get_current_user),
    mm: "MemoryManager" = Depends(get_memory_manager),
):
    """获取当前用户的会话列表（按更新时间倒序）"""
    sessions = mm.list_sessions(user_id=current_user.user_id, limit=limit)
    data = []
    for s in sessions:
        data.append(SessionInfoResponse(
            session_id=s.session_id,
            title=s.title,
            created_at=s.created_at,
            updated_at=s.updated_at,
            message_count=s.message_count,
            total_tokens=s.total_tokens,
            source=s.source,
            is_active=s.is_active,
            end_reason=s.end_reason,
        ))
    return SessionListResponse(code=200, message=f"共 {len(data)} 个会话", data=data, total=len(data))


@router.post("/sessions/{session_id}/load", response_model=BaseResponse, summary="加载会话")
async def load_session(
    session_id: str,
    current_user: UserInfo = Depends(get_current_user),
    mm: "MemoryManager" = Depends(get_memory_manager),
):
    """加载指定会话，恢复对话历史"""
    success = mm.load_session(session_id, user_id=current_user.user_id)
    if not success:
        raise HTTPException(status_code=404, detail="会话不存在")
    return BaseResponse(code=200, message="会话加载成功", data={"session_id": session_id})


@router.post("/sessions/{session_id}/end", response_model=BaseResponse, summary="结束会话")
async def end_session(
    session_id: str,
    current_user: UserInfo = Depends(get_current_user),
    mm: "MemoryManager" = Depends(get_memory_manager),
):
    """结束指定会话并保存，触发自动记忆提取（带用户归属校验）"""
    info = mm.session_store.get_session(session_id, user_id=current_user.user_id)
    if not info:
        raise HTTPException(status_code=404, detail="会话不存在")
    await mm.end_session(session_id=session_id, user_id=current_user.user_id, reason="user_close")
    return BaseResponse(code=200, message="会话已结束")


@router.delete("/sessions/{session_id}", response_model=BaseResponse, summary="删除会话")
async def delete_session(
    session_id: str,
    current_user: UserInfo = Depends(get_current_user),
    mm: "MemoryManager" = Depends(get_memory_manager),
):
    """删除会话及其所有消息"""
    success = mm.delete_session(session_id, user_id=current_user.user_id)
    if not success:
        raise HTTPException(status_code=404, detail="会话不存在")
    return BaseResponse(code=200, message="会话已删除")


@router.get("/sessions/{session_id}/messages", response_model=BaseResponse, summary="获取会话消息")
async def get_session_messages(
    session_id: str,
    current_user: UserInfo = Depends(get_current_user),
    mm: "MemoryManager" = Depends(get_memory_manager),
):
    """获取指定会话的完整消息历史（带用户归属校验）"""
    # 先校验会话是否属于当前用户
    info = mm.session_store.get_session(session_id, user_id=current_user.user_id)
    if not info:
        raise HTTPException(status_code=404, detail="会话不存在")

    messages = mm.session_store.get_messages(session_id, user_id=current_user.user_id)
    return BaseResponse(
        code=200,
        message=f"共 {len(messages)} 条消息",
        data=[
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp,
                "name": m.name,
            }
            for m in messages
        ],
    )


# ==================== 记忆操作 ====================

@router.post("/recall", response_model=MemoryRecallResponse, summary="召回记忆")
async def recall_memory(
    request: MemoryRecallRequest,
    current_user: UserInfo = Depends(get_current_user),
    mm: "MemoryManager" = Depends(get_memory_manager),
):
    """根据查询召回当前用户的相关长期记忆"""
    entries = mm.recall_memory(
        user_id=current_user.user_id,
        query=request.query,
        category=request.category,
        top_k=request.top_k,
    )
    
    data = []
    for e in entries:
        data.append(MemoryEntryResponse(
            category=e.category,
            content=e.content,
            confidence=round(e.confidence, 4),
            created_at=e.created_at,
            tags=e.tags,
        ))
    
    return MemoryRecallResponse(
        code=200,
        message=f"召回 {len(data)} 条记忆",
        data=data,
    )


@router.get("/stats", response_model=MemoryStatsResponse, summary="记忆统计")
async def memory_stats(
    current_user: UserInfo = Depends(get_current_user),
    mm: "MemoryManager" = Depends(get_memory_manager),
):
    """获取当前用户的记忆系统统计信息"""
    stats = mm.get_stats(user_id=current_user.user_id)
    return MemoryStatsResponse(code=200, message="获取成功", data=stats)
