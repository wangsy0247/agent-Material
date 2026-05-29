"""
RAG 路由（已迁移）

RAG 能力已解耦至 rag_mcp_server/，通过 MCP 协议提供：
    - stdio 模式：供 Agent 本地调用（extensions_config.json 配置）
    - SSE 模式：供前端直接流式访问（python -m rag_mcp_server --transport sse --port 8003）

本模块保留路由占位，返回迁移提示信息。
"""

from fastapi import APIRouter, Depends, HTTPException
from ..auth.dependencies import get_current_user
from ..auth.schemas import UserInfo

router = APIRouter(prefix="/rag", tags=["RAG 知识库"])


MIGRATION_NOTICE = (
    "RAG 服务已解耦至独立的 MCP Server (rag_mcp_server)。\n"
    "使用方式：\n"
    "  1. Agent 侧：通过 MCP 调用 rag_search / rag_query / rag_load_documents 等工具\n"
    "  2. 前端侧：直接连接 SSE 端点 python -m rag_mcp_server --transport sse --port 8003\n"
    "  3. 本地 CLI：python -m rag_mcp_server --transport stdio"
)


@router.post("/retrieve", summary="检索相关文档（已迁移）")
async def rag_retrieve(
    current_user: UserInfo = Depends(get_current_user),
):
    """RAG 检索已迁移至 rag_mcp_server。"""
    raise HTTPException(
        status_code=410,
        detail=MIGRATION_NOTICE,
    )


@router.post("/query", summary="RAG 问答（已迁移）")
async def rag_query(
    current_user: UserInfo = Depends(get_current_user),
):
    """RAG 问答已迁移至 rag_mcp_server。"""
    raise HTTPException(
        status_code=410,
        detail=MIGRATION_NOTICE,
    )
