"""
系统路由：健康检查、统计信息等
"""

import time
from fastapi import APIRouter, Depends
from api.schemas import HealthResponse, RAGStatsResponse
from api.dependencies import get_rag_service

router = APIRouter(prefix="/system", tags=["系统管理"])

# 服务启动时间
START_TIME = time.time()


@router.get("/health", response_model=HealthResponse, summary="健康检查")
async def health_check():
    """检查服务是否正常运行"""
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


@router.get("/rag/stats", response_model=RAGStatsResponse, summary="RAG 统计信息")
async def rag_stats(rag_service=Depends(get_rag_service)):
    """获取 RAG 向量库的统计信息"""
    try:
        doc_count = rag_service.vector_store.get_document_count()
        return RAGStatsResponse(
            code=200,
            message="获取成功",
            data={
                "document_count": doc_count,
                "collection_name": rag_service.vector_store.vector_store._collection.name
                if hasattr(rag_service.vector_store.vector_store, "_collection")
                else "unknown",
            },
        )
    except Exception as e:
        return RAGStatsResponse(
            code=500,
            message=f"获取统计信息失败: {str(e)}",
            data={},
        )
