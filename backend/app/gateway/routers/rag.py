"""
RAG 路由（REST API 直连版）

为前端 RAG 管理页面提供直接的 REST API 调用能力，
绕过 MCP 协议，直接调用 rag_engine 内部模块。
"""

from __future__ import annotations

import asyncio
import traceback
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..auth.dependencies import get_current_user
from ..auth.schemas import UserInfo
from ..schemas import BaseResponse
from ..sse import HEARTBEAT_INTERVAL, format_heartbeat, format_sse_event as _sse

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
        import os
        import faulthandler
        faulthandler.enable()

        # 诊断：检测 libffi 冲突（已知 WSL+Anaconda 会导致 psycopg segfault）
        ld_preload = os.environ.get("LD_PRELOAD", "")
        if "libffi" not in ld_preload:
            import logging
            logging.warning(
                "[RAG] LD_PRELOAD 未设置 libffi 系统库。"
                "如果后续出现 Segmentation fault，请使用 scripts/start_server.sh 启动，"
                "或手动执行: export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7"
            )

        from agent_material.rag_engine.vector_store import VectorStore

        _VectorStore = VectorStore()
    return _VectorStore


def _get_rag_service():
    global _RagService
    if _RagService is None:
        from agent_material.rag_engine.rag_service import RagService

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


class RAGBatchDeleteRequest(BaseModel):
    sources: list[str] = Field(..., min_length=1, description="要删除的来源路径列表")


class RAGAddChunksRequest(BaseModel):
    chunks: list[str] = Field(..., min_length=1, description="已切分好的文本片段列表")
    sources: list[str] | None = Field(default=None, description="来源标识列表，与 chunks 一一对应；不传则自动分配")
    metadatas: list[dict] | None = Field(default=None, description="额外元数据列表，与 chunks 一一对应")


# ------------------------------------------------------------------
# 路由
# ------------------------------------------------------------------


@router.get("/stats", summary="获取 RAG 统计信息", response_model=BaseResponse)
async def rag_stats(current_user: UserInfo = Depends(get_current_user)):
    """获取向量库统计信息：文档总数、配置参数等。"""
    try:
        store = _get_vector_store()
        from agent_material.utils.config_handler import vector_store_config

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
    - `status`: 阶段切换（{"phase": "retrieving" | "generating"}）
    - `sources`: 检索到的引用来源
    - `token`: 大模型实时生成的 token（data.index 从 1 递增）
    - `done`: 完成
    - `error`: 错误
    - `cancelled`: 客户端断连/中止导致的取消（终态）
    """

    async def event_generator():
        # 帧 id 计数器：run 内单调递增，供前端记录 lastEventId
        event_id = 0
        # token 序号：run 内从 1 递增，供前端乱序/丢帧检测
        token_index = 0

        def next_sse(event_type: str, content=None, index=None) -> str:
            nonlocal event_id
            event_id += 1
            return _sse(event_type, content, event_id, index=index)

        def token_sse(text: str) -> str:
            nonlocal token_index
            token_index += 1
            return next_sse("token", text, index=token_index)

        try:
            # RagService 首次初始化是重量级同步操作，放线程避免阻塞 event loop
            service = await asyncio.to_thread(_get_rag_service)

            # 1. 阶段事件：开始检索
            yield next_sse("status", {"phase": "retrieving"})

            # 2. 先执行检索（同步调用，放线程避免阻塞 event loop）
            docs = await asyncio.to_thread(service.retriever_docx, request.query)

            # 3. 发送 start 事件
            yield next_sse("start", {"query": request.query})

            # 4. 发送 sources 事件
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
            yield next_sse("sources", sources)

            if not docs:
                yield token_sse("根据现有资料无法回答该问题。")
                yield next_sse("done")
                return

            # 5. 阶段事件：开始生成
            yield next_sse("status", {"phase": "generating"})

            # 6. 流式生成：get_summary_stream 是同步生成器函数（惰性），
            # 整个迭代都在线程中执行，经 asyncio.Queue 桥接回 async 侧逐 chunk 消费，
            # 避免普通 for 迭代独占 event loop。
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()
            _SENTINEL = None  # 结束哨兵

            def _consume_stream():
                try:
                    for chunk in service.get_summary_stream(request.query):
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
                    loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)
                except Exception as exc:  # 异常经队列传回 async 侧 re-raise
                    loop.call_soon_threadsafe(queue.put_nowait, exc)

            consume_task = asyncio.create_task(asyncio.to_thread(_consume_stream))
            try:
                while True:
                    try:
                        # 超时发注释心跳帧：长时间无 token 时防代理/浏览器判死断连
                        item = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_INTERVAL)
                    except TimeoutError:
                        yield format_heartbeat()
                        continue
                    if item is _SENTINEL:
                        break
                    if isinstance(item, Exception):
                        raise item
                    yield token_sse(item)
            finally:
                # 确保线程任务收尾（正常路径已结束；异常路径兜底等待）
                await asyncio.shield(asyncio.gather(consume_task, return_exceptions=True))

            yield next_sse("done")
        except asyncio.CancelledError:
            # 客户端断连/中止：尽力向当前连接发 cancelled 帧（连接已断开时
            # 发送会失败，静默忽略），随后原样抛出结束流
            try:
                yield next_sse("cancelled", {"message": "run cancelled"})
            except BaseException:
                pass
            raise
        except Exception as e:
            traceback_str = traceback.format_exc()
            # error 为终态：发完 error 帧直接结束，不再依赖连接关闭收尾
            yield next_sse("error", {"message": str(e), "detail": traceback_str})
            return

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 关闭 nginx 代理缓冲，保证实时推送
        },
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


@router.post("/add-chunks", summary="批量添加已切分的 chunks（不再二次切分）", response_model=BaseResponse)
async def rag_add_chunks(
    request: RAGAddChunksRequest,
    current_user: UserInfo = Depends(get_current_user),
):
    """将已切分好的文本片段直接转为向量并批量写入向量库，不做额外切分。"""
    try:
        from langchain_core.documents import Document

        if request.metadatas and len(request.metadatas) != len(request.chunks):
            raise HTTPException(status_code=400, detail="metadatas 长度必须与 chunks 一致")

        sources = request.sources or [f"chunk_batch_{i}" for i in range(len(request.chunks))]
        if len(sources) != len(request.chunks):
            raise HTTPException(status_code=400, detail="sources 长度必须与 chunks 一致")

        docs: list[Document] = []
        for i, text in enumerate(request.chunks):
            meta = dict(request.metadatas[i]) if request.metadatas and i < len(request.metadatas) else {}
            meta["source"] = sources[i]
            docs.append(Document(page_content=text, metadata=meta))

        store = _get_vector_store()
        # 直接走 PGVector，不再经过 VectorStore.add_documents 的二次切分
        store.vector_store.add_documents(docs)
        store.all_documents.extend(docs)
        store._retrievers_initialized = False
        return BaseResponse(
            code=200,
            message=f"成功入库 {len(docs)} 条 chunks",
            data={"added": len(docs)},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"入库失败: {e}")


@router.get("/sources", summary="获取所有文献来源", response_model=BaseResponse)
async def rag_sources(current_user: UserInfo = Depends(get_current_user)):
    """获取向量库中所有唯一的文献来源列表（含片段数）。"""
    try:
        store = _get_vector_store()
        sources = store.get_all_sources()
        return BaseResponse(
            code=200,
            message="success",
            data=sources,
            total=len(sources),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文献列表失败: {e}")


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


@router.post("/delete-sources", summary="批量删除文献", response_model=BaseResponse)
async def rag_delete_sources(
    request: RAGBatchDeleteRequest,
    current_user: UserInfo = Depends(get_current_user),
):
    """批量删除向量库中的文献。"""
    try:
        store = _get_vector_store()
        failed = []
        for source in request.sources:
            try:
                store.delete_documents_by_source(source)
            except Exception as e:
                failed.append({"source": source, "error": str(e)})
        success_count = len(request.sources) - len(failed)
        msg = f"成功删除 {success_count} 篇文献"
        if failed:
            msg += f"，{len(failed)} 篇失败"
        return BaseResponse(
            code=200,
            message=msg,
            data={"success": success_count, "failed": failed},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量删除失败: {e}")
