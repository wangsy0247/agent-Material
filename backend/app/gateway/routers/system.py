"""
系统路由：健康检查、统计信息等
"""

import time
from fastapi import APIRouter
from ..schemas import HealthResponse

router = APIRouter(prefix="/system", tags=["系统管理"])

# 服务启动时间
START_TIME = time.time()


@router.get("/health", response_model=HealthResponse, summary="健康检查")
async def health_check():
    """检查服务是否正常运行（公开访问，无需认证）"""
    uptime = time.time() - START_TIME
    return HealthResponse(
        code=200,
        message="服务运行正常",
        data={
            "status": "healthy",
            "uptime_seconds": round(uptime, 2),
            "version": "0.1.0",
        },
    )


# 注意：/system/rag/stats 已随 RAG 解耦一起迁移至 rag_mcp_server。
#       前端可通过 rag_get_stats MCP Tool 获取统计信息。
