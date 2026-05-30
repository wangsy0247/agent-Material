"""
RAG 路由（REST API 直连版）

为前端 RAG 管理页面提供直接的 REST API 调用能力，
绕过 MCP 协议，直接调用 rag_mcp_server 内部模块。
"""

from __future__ import annotations

import json
import time
import traceback
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..auth.dependencies import get_current_user
from ..auth.schemas import UserInfo
from ..schemas import BaseResponse, SSEEvent

if TYPE_CHECKING:
    pass

router = APIRouter(prefix="/rag", tags=["RAG 知识库"])

# ------------------------------------------------------------------
# 延迟初始化（避免模块导入时触发重量级初始化）
# ------------------------------------------------------------------
_VectorStore = None
_RagService = None


def _get_vector_store():
    global _VectorStore
    if _VectorStore is None:
        from rag_mcp_server.rag.vector_store import VectorStore

        _VectorStore = VectorStore()
    return _VectorStore


def _get_rag_service():
    global _RagService
    if _RagService is None:
        from rag_mcp_server.rag.rag_service import RagService

        _RagService = RagService()
    return _RagService


# ------------------------------------------------------------------
# 请求模型
# ------------------------------------------------------------------


class RAGSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000, description="检索查询")


class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000, description="用户问题")


class RAGAddDocumentsRequest(BaseModel):
    documents: list[str] = Field(..., min_length=1, description="文本内容列表")
    sources: list[str] | None = Field(default=None, description="来源标识列表")


class RAGDeleteRequest(BaseModel):
    source: str = Field(..., min_length=1, description="来源路径或标识")


# ------------------------------------------------------------------
# SSE 辅助
# ------------------------------------------------------------------


def _sse(event_type: str, content: str | dict | list | None = None) -> str:
    event = SSEEvent(type=event_type, content=content, timestamp=time.time())
    return f"data: {event.model_dump_json()}\n\n"


# ------------------------------------------------------------------
# 路由
# ------------------------------------------------------------------


@router.get("/stats", summary="获取 RAG 统计信息", response_model=BaseResponse)
async def rag_stats(current_user: UserInfo = Depends(get_current_user)):
    """获取向量库统计信息：文档总数、配置参数等。"""
    try:
        store = _get_vector_store()
        from rag_mcp_server.config import vector_store_config

        def _mask_conn_str(conn_str: str) -> str:
            try:
                if "@" in conn_str:
                    prefix, rest = conn_str.split("@", 1)
                    if ":" in prefix:
                        scheme_user, _ = prefix.rsplit(":", 1)
                        return f"{scheme_user}:****@{rest}"
                return conn_str
            except Exception:
                return "***masked***"

        stats = {
            "document_count": store.get_document_count(),
            "collection_name": vector_store_config.get("character_name", "agent"),
            "connection_string_masked": _mask_conn_str(
                vector_store_config.get("connection_string", "")
            ),
            "splitter_type": vector_store_config.get("splitter_type", "recursive"),
            "chunk_size": vector_store_config.get("chunk_size"),
            "chunk_overlap": vector_store_config.get("chunk_overlap"),
            "vector_weight": vector_store_config.get("vector_weight", 0.7),
            "bm25_weight": vector_store_config.get("bm25_weight", 0.3),
            "enable_reranker": vector_store_config.get("enable_reranker", True),
            "top_n": vector_store_config.get("top_n", 3),
            "score_threshold": vector_store_config.get("score_threshold", 0.3),
            "data_directory": vector_store_config.get("data_directory", "data/"),
        }
        return BaseResponse(code=200, message="success", data=stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {e}")


@router.post("/search", summary="检索相关文档", response_model=BaseResponse)
async def rag_search(
    request: RAGSearchRequest,
    current_user: UserInfo = Depends(get_current_user),
):
    """根据查询内容从向量数据库检索最相关的文档片段。"""
    try:
        service = _get_rag_service()
        docs = service.retriever_docx(request.query)
        if not docs:
            return BaseResponse(code=200, message="未检索到相关文档", data=[])

        results = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "未知来源")
            score = doc.metadata.get("score", 0.0)
            results.append(
                {
                    "index": i,
                    "source": source,
                    "score": score,
                    "snippet": doc.page_content,
                    "metadata": doc.metadata,
                }
            )
        return BaseResponse(code=200, message="success", data=results, total=len(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败: {e}")


@router.post("/query", summary="RAG 问答（SSE 流式）")
async def rag_query(
    request: RAGQueryRequest,
    current_user: UserInfo = Depends(get_current_user),
):
    """
    RAG 问答：检索相关文档并流式生成回答。

    SSE 事件类型：
    - `start`: 开始处理
    - `sources`: 检索到的引用来源
    - `token`: 大模型实时生成的 token
    - `done`: 完成
    - `error`: 错误
    """

    async def event_generator():
        try:
            service = _get_rag_service()

            # 1. 先执行检索
            docs = service.retriever_docx(request.query)

            # 2. 发送 start 事件
            yield _sse("start", {"query": request.query})

            # 3. 发送 sources 事件
            sources = []
            for i, doc in enumerate(docs, start=1):
                source = doc.metadata.get("source", "未知来源")
                score = doc.metadata.get("score", 0.0)
                sources.append(
                    {
                        "index": i,
                        "source": source,
                        "score": score,
                        "snippet": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    }
                )
            yield _sse("sources", sources)

            if not docs:
                yield _sse("token", "根据现有资料无法回答该问题。")
                yield _sse("done")
                return

            # 4. 流式生成
            for chunk in service.get_summary_stream(request.query):
                yield _sse("token", chunk)

            yield _sse("done")
        except Exception as e:
            traceback_str = traceback.format_exc()
            yield _sse("error", {"message": str(e), "detail": traceback_str})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


@router.post("/load-documents", summary="扫描并加载文档", response_model=BaseResponse)
async def rag_load_documents(current_user: UserInfo = Depends(get_current_user)):
    """扫描配置的 data/ 目录，加载新增或修改的文档到向量库。"""
    try:
        store = _get_vector_store()
        before = store.get_document_count()
        store.load_document()
        after = store.get_document_count()
        loaded = after - before
        return BaseResponse(
            code=200,
            message="文档加载完成",
            data={
                "before": before,
                "after": after,
                "loaded": loaded,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"加载文档失败: {e}")


@router.post("/add-documents", summary="手动添加文档", response_model=BaseResponse)
async def rag_add_documents(
    request: RAGAddDocumentsRequest,
    current_user: UserInfo = Depends(get_current_user),
):
    """手动向向量库添加文本内容。"""
    try:
        from langchain_core.documents import Document

        sources = request.sources or [f"manual_{i}" for i in range(len(request.documents))]
        if len(sources) != len(request.documents):
            raise HTTPException(status_code=400, detail="sources 长度必须与 documents 一致")

        docs = [
            Document(page_content=text, metadata={"source": src})
            for text, src in zip(request.documents, sources)
        ]
        store = _get_vector_store()
        store.add_documents(docs)
        return BaseResponse(
            code=200,
            message=f"成功添加 {len(docs)} 条文档到向量库",
            data={"added": len(docs)},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加文档失败: {e}")


@router.post("/delete-by-source", summary="按来源删除文档", response_model=BaseResponse)
async def rag_delete_by_source(
    request: RAGDeleteRequest,
    current_user: UserInfo = Depends(get_current_user),
):
    """根据来源路径删除向量库中的文档片段。"""
    try:
        store = _get_vector_store()
        store.delete_documents_by_source(request.source)
        return BaseResponse(
            code=200,
            message=f"已删除来源为 '{request.source}' 的文档",
            data={"source": request.source},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除文档失败: {e}")
