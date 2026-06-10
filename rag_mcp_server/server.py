"""
RAG MCP Server（完全独立）

所有业务逻辑（vector_store, rag_service, text_spliter）和基础设施
（config, models, logger, file_loader, prompt_loader）均已内聚于本包。
不再通过 sys.path 注入依赖主项目的 agent_material 包。

提供的 Tools:
    rag_search            — 向量检索，返回相关文档片段（支持查询改写）
    rag_query             — RAG 问答（自动路由 + 可选改写 + LLM 生成）
    rag_load_documents    — 扫描 data/ 目录，增量入队处理
    rag_add_documents     — 手动添加文本到向量库
    rag_delete_by_source  — 按来源路径删除文档
    rag_get_stats         — 获取向量库统计信息
    rag_get_task_status   — 查询队列任务状态
    rag_get_pending_tasks — 查询队列中待处理任务数
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
import uuid

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── 可观测性 ──
from rag_mcp_server.observability.structured_logger import (
    get_logger,
    set_request_context,
)
from rag_mcp_server.observability.metrics import (
    request_total,
    request_duration,
    request_in_flight,
    retrieval_empty_total,
)
from rag_mcp_server.observability.mcp_trace import extract_trace_context as _extract_trace

# 延迟导入 RAG 模块（避免在模块顶层触发重量级初始化）
_VectorStore = None
_RagService = None
_TaskQueue = None


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


def _get_task_queue():
    global _TaskQueue
    if _TaskQueue is None:
        from rag_mcp_server.db.task_queue import TaskQueue
        from rag_mcp_server.config import queue_config
        db_path = queue_config.get("queue", {}).get("db_path", "rag_queue.db")
        _TaskQueue = TaskQueue(db_path=db_path)
    return _TaskQueue


logger = get_logger("server")

# ------------------------------------------------------------------
# 序列化辅助
# ------------------------------------------------------------------


def _serialize_docs(docs) -> list[dict]:
    """将 LangChain Document 列表序列化为可 JSON 的 dict 列表。"""
    return [
        {
            "page_content": d.page_content,
            "metadata": d.metadata,
        }
        for d in docs
    ]


# ------------------------------------------------------------------
# FastMCP Server
# ------------------------------------------------------------------


def create_server(host: str = "127.0.0.1", port: int = 8003) -> "FastMCP":
    from mcp.server import FastMCP
    from mcp.types import ToolAnnotations

    # ── 初始化可观测性 ──
    try:
        from rag_mcp_server.observability import init_all
        init_all(service_name="rag_mcp_server")
    except Exception:
        logger.warning("可观测性初始化失败，使用默认日志", exc_info=True)

    mcp = FastMCP(
        "rag-server",
        instructions=(
            "Abacus和lammps软件使用的 RAG（检索增强生成）服务。基于 PGVector + BM25 + CrossEncoder "
            "构建的混合检索系统，支持向量检索、文档加载、增量更新和智能问答。"
            "支持多粒度索引（子chunk检索 + 父chunk上下文）和后台任务队列。"
        ),
        host=host,
        port=port,
    )

    # ================================================================
    # Tool: rag_search — 向量检索
    # ================================================================
    @mcp.tool(
        name="rag_search",
        description=(
            "根据查询内容从向量数据库中检索最相关的文档片段。"
            "使用混合检索（向量相似度 + BM25 关键词 + CrossEncoder 重排序）。"
            "多粒度模式下返回父chunk以提供更丰富的上下文。"
            "可选启用查询改写（enable_rewrite=True），支持组合策略（如 step_back+expand）。"
            "单策略: expand/hyde/step_back/decompose，用 + 组合。"
            "返回文档片段的文本内容、来源文件和元数据。"
        ),
        annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False, idempotentHint=True),
    )
    async def rag_search(
        query: str,
        enable_rewrite: bool = False,
        rewrite_strategy: str = "",
        tracing_meta: dict | None = None,  # MCP Trace Context 跨进程传播
    ) -> str:
        """检索与查询相关的文档片段。"""
        req_id = uuid.uuid4().hex[:12]
        set_request_context(request_id=req_id)

        # MCP 跨进程 trace context 恢复
        parent_ctx = _extract_trace(tracing_meta)
        try:
            from opentelemetry import trace
            tracer = trace.get_tracer("rag_mcp_server")
        except ImportError:
            tracer = None

        method = "rag_search"
        request_in_flight.labels(method=method).inc()
        start = time.time()

        try:
            logger.info(
                "rag_search_start",
                extra={
                    "query": query[:100],
                    "enable_rewrite": enable_rewrite,
                    "strategy": rewrite_strategy or "default",
                },
            )
            if tracer:
                with tracer.start_as_current_span(
                    "rag.search",
                    context=parent_ctx,
                ) as span:
                    span.set_attribute("query", query[:200])
                    span.set_attribute("enable_rewrite", enable_rewrite)
                    if rewrite_strategy:
                        span.set_attribute("strategy", rewrite_strategy)
                    service = _get_rag_service()
                    if enable_rewrite:
                        docs = await asyncio.to_thread(
                            service._retrieve_with_strategy, query, rewrite_strategy
                        )
                    else:
                        docs = await asyncio.to_thread(service.retriever_docx, query)
                    span.set_attribute("result_count", len(docs) if docs else 0)
            else:
                service = _get_rag_service()
                if enable_rewrite:
                    docs = await asyncio.to_thread(
                        service._retrieve_with_strategy, query, rewrite_strategy
                    )
                else:
                    docs = await asyncio.to_thread(service.retriever_docx, query)

            if not docs:
                retrieval_empty_total.labels(method=method).inc()
                request_total.labels(method=method, status="empty").inc()
                return json.dumps(
                    {"results": [], "message": "未检索到相关文档"}, ensure_ascii=False
                )

            request_total.labels(method=method, status="success").inc()
            return json.dumps(
                {"results": _serialize_docs(docs), "count": len(docs)},
                ensure_ascii=False,
                indent=2,
            )
        except Exception as e:
            request_total.labels(method=method, status="error").inc()
            logger.error(
                "rag_search_failed",
                extra={"error": str(e)},
                exc_info=True,
            )
            return json.dumps({"error": str(e)}, ensure_ascii=False)
        finally:
            elapsed = time.time() - start
            request_duration.labels(method=method).observe(elapsed)
            request_in_flight.labels(method=method).dec()
            logger.info(
                "rag_search_end",
                extra={"duration_ms": elapsed * 1000},
            )

    # ================================================================
    # Tool: rag_query — RAG 问答
    # ================================================================
    @mcp.tool(
        name="rag_query",
        description=(
            "基于向量检索结果，调用大语言模型生成回答。"
            "内置查询路由：自动检测闲聊（直答）、无关问题（拒答）、问题类型标注。"
            "可选启用查询改写（enable_rewrite=True），支持组合策略（如 step_back+expand）。"
            "单策略: expand/hyde/step_back/decompose，用 + 组合。"
            "适合需要综合多篇文献进行总结、分析、对比的复杂问题。"
        ),
        annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False, idempotentHint=False),
    )
    async def rag_query(
        query: str,
        enable_rewrite: bool = False,
        rewrite_strategy: str = "",
        tracing_meta: dict | None = None,  # MCP Trace Context 跨进程传播
    ) -> str:
        """RAG 问答：检索相关文档并生成回答（自动路由 + 可选改写）。"""
        req_id = uuid.uuid4().hex[:12]
        set_request_context(request_id=req_id)

        parent_ctx = _extract_trace(tracing_meta)
        try:
            from opentelemetry import trace
            tracer = trace.get_tracer("rag_mcp_server")
        except ImportError:
            tracer = None

        method = "rag_query"
        request_in_flight.labels(method=method).inc()
        start = time.time()

        try:
            logger.info(
                "rag_query_start",
                extra={
                    "query": query[:100],
                    "enable_rewrite": enable_rewrite,
                    "strategy": rewrite_strategy or "default",
                },
            )
            if tracer:
                with tracer.start_as_current_span(
                    "rag.query",
                    context=parent_ctx,
                ) as span:
                    span.set_attribute("query", query[:200])
                    span.set_attribute("enable_rewrite", enable_rewrite)
                    if rewrite_strategy:
                        span.set_attribute("strategy", rewrite_strategy)
                    service = _get_rag_service()
                    answer = await asyncio.to_thread(
                        service.get_summary,
                        query,
                        enable_rewrite=enable_rewrite,
                        rewrite_strategy=rewrite_strategy,
                    )
                    span.set_attribute("answer_length", len(answer) if answer else 0)
            else:
                service = _get_rag_service()
                answer = await asyncio.to_thread(
                    service.get_summary,
                    query,
                    enable_rewrite=enable_rewrite,
                    rewrite_strategy=rewrite_strategy,
                )

            request_total.labels(method=method, status="success").inc()
            return answer
        except Exception as e:
            request_total.labels(method=method, status="error").inc()
            logger.error(
                "rag_query_failed",
                extra={"error": str(e)},
                exc_info=True,
            )
            return f"RAG 问答失败: {e}"
        finally:
            elapsed = time.time() - start
            request_duration.labels(method=method).observe(elapsed)
            request_in_flight.labels(method=method).dec()
            logger.info(
                "rag_query_end",
                extra={"duration_ms": elapsed * 1000},
            )

    # ================================================================
    # Tool: rag_load_documents — 增量扫描入队（非阻塞）
    # ================================================================
    @mcp.tool(
        name="rag_load_documents",
        description=(
            "扫描配置的 data/ 目录，自动识别新增/修改/删除的文档，"
            "将处理任务加入后台队列异步执行。基于 SQLite 文件追踪实现增量更新，"
            "只有变化的文件会被重新处理。任务在后台 Worker 中执行，此工具非阻塞返回。"
        ),
        annotations=ToolAnnotations(destructiveHint=True, openWorldHint=False, idempotentHint=False),
    )
    async def rag_load_documents() -> str:
        """扫描并加载 data/ 目录下的文档到向量库（非阻塞入队）。"""
        logger.info("[rag_load_documents] 开始扫描文档目录")
        try:
            store = _get_vector_store()
            before = store.get_document_count()
            await asyncio.to_thread(store.load_document)
            after = store.get_document_count()

            queue = _get_task_queue()
            summary = queue.get_summary()

            msg = (
                f"文档扫描完成 | 扫描前: {before} 条 | 当前: {after} 条\n"
                f"队列状态: 待处理 {summary['pending']} | 处理中 {summary['processing']} | "
                f"已完成 {summary['completed']} | 失败 {summary['failed']}"
            )
            logger.info(f"[rag_load_documents] {msg}")
            return msg
        except Exception as e:
            logger.error(f"[rag_load_documents] 失败: {e}\n{traceback.format_exc()}")
            return f"文档加载失败: {e}"

    # ================================================================
    # Tool: rag_add_documents — 手动添加文档
    # ================================================================
    @mcp.tool(
        name="rag_add_documents",
        description=(
            "手动向向量库添加文本内容。适合一次性注入少量知识片段。"
            "文本会自动经过切分器处理后再写入向量库。"
        ),
        annotations=ToolAnnotations(destructiveHint=True, openWorldHint=False, idempotentHint=False),
    )
    async def rag_add_documents(
        documents: list[str],
        sources: list[str] | None = None,
    ) -> str:
        """手动添加文本到向量库。

        Args:
            documents: 文本内容列表，每条会作为一个文档处理
            sources: （可选）来源标识列表，与 documents 一一对应，如 ["manual_note_1", "manual_note_2"]
        """
        logger.info(f"[rag_add_documents] count={len(documents)}")
        try:
            from langchain_core.documents import Document

            if sources is None:
                sources = [f"manual_{i}" for i in range(len(documents))]
            if len(sources) != len(documents):
                return "参数错误: sources 长度必须与 documents 一致"

            docs = [
                Document(page_content=text, metadata={"source": src})
                for text, src in zip(documents, sources)
            ]
            store = _get_vector_store()
            await asyncio.to_thread(store.add_documents, docs)
            return f"成功添加 {len(docs)} 条文档到向量库"
        except Exception as e:
            logger.error(f"[rag_add_documents] 失败: {e}\n{traceback.format_exc()}")
            return f"添加文档失败: {e}"

    # ================================================================
    # Tool: rag_delete_by_source — 按来源删除
    # ================================================================
    @mcp.tool(
        name="rag_delete_by_source",
        description=(
            "根据来源路径删除向量库中的文档片段。"
            "支持精确匹配 metadata 中的 source 字段，用于增量更新或清理错误数据。"
        ),
        annotations=ToolAnnotations(destructiveHint=True, openWorldHint=False, idempotentHint=False),
    )
    async def rag_delete_by_source(source: str) -> str:
        """按来源路径删除文档。

        Args:
            source: 来源路径或标识，如 "data/某论文.pdf" 或 "manual_note_1"
        """
        logger.info(f"[rag_delete_by_source] source={source}")
        try:
            store = _get_vector_store()
            await asyncio.to_thread(store.delete_documents_by_source, source)
            return f"已删除来源为 '{source}' 的文档"
        except Exception as e:
            logger.error(f"[rag_delete_by_source] 失败: {e}\n{traceback.format_exc()}")
            return f"删除文档失败: {e}"

    # ================================================================
    # Tool: rag_get_stats — 统计信息
    # ================================================================
    @mcp.tool(
        name="rag_get_stats",
        description="获取 RAG 向量库的当前统计信息，包括文档总数、配置参数等。",
        annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False, idempotentHint=True),
    )
    async def rag_get_stats() -> str:
        """获取向量库统计信息。"""
        logger.info("[rag_get_stats]")
        try:
            store = _get_vector_store()
            from rag_mcp_server.config import vector_store_config

            stats = {
                "document_count": store.get_document_count(),
                "collection_name": vector_store_config.get("character_name", "agent"),
                "connection_string_masked": _mask_conn_str(
                    vector_store_config.get("connection_string", "")
                ),
                "chunk_size": vector_store_config.get("chunk_size"),
                "chunk_overlap": vector_store_config.get("chunk_overlap"),
                "vector_weight": vector_store_config.get("vector_weight", 0.7),
                "bm25_weight": vector_store_config.get("bm25_weight", 0.3),
                "enable_reranker": vector_store_config.get("enable_reranker", True),
                "top_n": vector_store_config.get("top_n", 3),
                "score_threshold": vector_store_config.get("score_threshold", 0.3),
                "multi_granularity": vector_store_config.get("multi_granularity", {}).get("enabled", False),
            }
            return json.dumps(stats, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[rag_get_stats] 失败: {e}\n{traceback.format_exc()}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    # ================================================================
    # Tool: rag_get_task_status — 查询任务状态
    # ================================================================
    @mcp.tool(
        name="rag_get_task_status",
        description="根据任务 ID 查询后台处理任务的状态。",
        annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False, idempotentHint=True),
    )
    async def rag_get_task_status(task_id: int) -> str:
        """查询任务状态。

        Args:
            task_id: 任务 ID
        """
        logger.info(f"[rag_get_task_status] task_id={task_id}")
        try:
            queue = _get_task_queue()
            status = queue.get_status(task_id)
            if status is None:
                return json.dumps({"error": f"任务 {task_id} 不存在"}, ensure_ascii=False)
            return json.dumps(status, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[rag_get_task_status] 失败: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    # ================================================================
    # Tool: rag_get_pending_tasks — 查询待处理任务
    # ================================================================
    @mcp.tool(
        name="rag_get_pending_tasks",
        description="获取任务队列的当前状态汇总（各状态任务数量）。",
        annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False, idempotentHint=True),
    )
    async def rag_get_pending_tasks() -> str:
        """获取队列状态汇总。"""
        logger.info("[rag_get_pending_tasks]")
        try:
            queue = _get_task_queue()
            summary = queue.get_summary()
            return json.dumps(summary, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[rag_get_pending_tasks] 失败: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    return mcp


def _mask_conn_str(conn_str: str) -> str:
    """隐藏连接字符串中的密码。"""
    try:
        if "@" in conn_str:
            prefix, rest = conn_str.split("@", 1)
            if ":" in prefix:
                scheme_user, _ = prefix.rsplit(":", 1)
                return f"{scheme_user}:****@{rest}"
        return conn_str
    except Exception:
        return "***masked***"


# ------------------------------------------------------------------
# CLI 入口
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="RAG MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="传输模式: stdio(子进程) 或 sse(HTTP 流式)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="SSE 模式监听地址")
    parser.add_argument("--port", type=int, default=8003, help="SSE 模式监听端口")
    args = parser.parse_args()

    server = create_server(host=args.host, port=args.port)

    tool_list = asyncio.run(server.list_tools())
    tool_names = [t.name for t in tool_list]
    print(f"[RAG MCP Server] 启动中...", file=sys.stderr)
    print(f"  名称: {server.name}", file=sys.stderr)
    print(f"  工具: {tool_names}", file=sys.stderr)
    print(f"  传输: {args.transport}", file=sys.stderr)

    if args.transport == "stdio":
        asyncio.run(server.run_stdio_async())
    else:
        asyncio.run(server.run_sse_async())


if __name__ == "__main__":
    main()
