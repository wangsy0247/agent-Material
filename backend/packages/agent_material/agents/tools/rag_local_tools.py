"""
RAG 本地工具 — 进程内直接调用 agent_material.rag_engine，不走 MCP 协议。

将 rag_mcp_server 的 8 个 MCP 工具转为 LangChain @tool 本地函数，
消除 JSON-RPC 序列化 + 子进程通信开销。RAG 核心代码已迁入 agent_material.rag_engine。

工具列表:
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

import json

from langchain_core.tools import tool

from agent_material.utils.logger import get_logger

logger = get_logger("rag_local_tools")

# ── 懒加载全局变量（与 rag_engine 保持一致） ──
_RagService = None
_VectorStore = None
_TaskQueue = None


def _get_rag_service():
    global _RagService
    if _RagService is None:
        from agent_material.rag_engine.rag_service import RagService

        _RagService = RagService()
    return _RagService


def _get_vector_store():
    global _VectorStore
    if _VectorStore is None:
        from agent_material.rag_engine.vector_store import VectorStore

        _VectorStore = VectorStore()
    return _VectorStore


def _get_task_queue():
    global _TaskQueue
    if _TaskQueue is None:
        from agent_material.rag_engine.db.task_queue import TaskQueue
        from agent_material.utils.config_handler import queue_config

        db_path = queue_config.get("queue", {}).get("db_path", "rag_queue.db")
        _TaskQueue = TaskQueue(db_path=db_path)
    return _TaskQueue


def _serialize_docs(docs) -> list[dict]:
    """将 LangChain Document 列表序列化为可 JSON 的 dict 列表。"""
    return [
        {"page_content": d.page_content, "metadata": d.metadata}
        for d in docs
    ]


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


# ================================================================
# Tool: rag_search — 向量检索
# ================================================================
@tool
def rag_search(
    query: str,
    enable_rewrite: bool = False,
    rewrite_strategy: str = "",
) -> str:
    """根据查询内容从向量数据库中检索最相关的文档片段。

    使用混合检索（向量相似度 + BM25 关键词 + CrossEncoder 重排序）。
    多粒度模式下返回父chunk以提供更丰富的上下文。
    可选启用查询改写（enable_rewrite=True），支持组合策略（如 step_back+expand）。
    单策略: expand/hyde/step_back/decompose，用 + 组合。
    返回文档片段的文本内容、来源文件和元数据。

    Args:
        query: 查询文本
        enable_rewrite: 是否启用查询改写
        rewrite_strategy: 改写策略，空字符串使用默认策略
    """
    try:
        service = _get_rag_service()
        if enable_rewrite:
            docs = service._retrieve_with_strategy(query, rewrite_strategy)
        else:
            docs = service.retriever_docx(query)

        if not docs:
            return json.dumps(
                {"results": [], "message": "未检索到相关文档"}, ensure_ascii=False
            )

        return json.dumps(
            {"results": _serialize_docs(docs), "count": len(docs)},
            ensure_ascii=False,
            indent=2,
        )
    except Exception as e:
        logger.error(f"[rag_search] 失败: {e}", exc_info=True)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ================================================================
# Tool: rag_query — RAG 问答
# ================================================================
@tool
def rag_query(
    query: str,
    enable_rewrite: bool = False,
    rewrite_strategy: str = "",
) -> str:
    """基于向量检索结果，调用大语言模型生成回答。

    内置查询路由：自动检测闲聊（直答）、无关问题（拒答）、问题类型标注。
    可选启用查询改写（enable_rewrite=True），支持组合策略（如 step_back+expand）。
    单策略: expand/hyde/step_back/decompose，用 + 组合。
    适合需要综合多篇文献进行总结、分析、对比的复杂问题。

    Args:
        query: 用户问题
        enable_rewrite: 是否启用查询改写
        rewrite_strategy: 改写策略，空字符串使用默认策略
    """
    try:
        service = _get_rag_service()
        answer = service.get_summary(
            query,
            enable_rewrite=enable_rewrite,
            rewrite_strategy=rewrite_strategy,
        )
        return answer
    except Exception as e:
        logger.error(f"[rag_query] 失败: {e}", exc_info=True)
        return f"RAG 问答失败: {e}"


# ================================================================
# Tool: rag_load_documents — 增量扫描入队（非阻塞）
# ================================================================
@tool
def rag_load_documents() -> str:
    """扫描配置的 data/ 目录，自动识别新增/修改/删除的文档。

    将处理任务加入后台队列异步执行。基于 SQLite 文件追踪实现增量更新，
    只有变化的文件会被重新处理。任务在后台 Worker 中执行，此工具非阻塞返回。
    """
    try:
        store = _get_vector_store()
        before = store.get_document_count()
        store.load_document()
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
        logger.error(f"[rag_load_documents] 失败: {e}", exc_info=True)
        return f"文档加载失败: {e}"


# ================================================================
# Tool: rag_add_documents — 手动添加文档
# ================================================================
@tool
def rag_add_documents(
    documents: list[str],
    sources: list[str] | None = None,
) -> str:
    """手动向向量库添加文本内容。

    适合一次性注入少量知识片段。
    文本会自动经过切分器处理后再写入向量库。

    Args:
        documents: 文本内容列表，每条会作为一个文档处理
        sources: 来源标识列表，与 documents 一一对应
    """
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
        store.add_documents(docs)
        return f"成功添加 {len(docs)} 条文档到向量库"
    except Exception as e:
        logger.error(f"[rag_add_documents] 失败: {e}", exc_info=True)
        return f"添加文档失败: {e}"


# ================================================================
# Tool: rag_delete_by_source — 按来源删除
# ================================================================
@tool
def rag_delete_by_source(source: str) -> str:
    """根据来源路径删除向量库中的文档片段。

    支持精确匹配 metadata 中的 source 字段，用于增量更新或清理错误数据。

    Args:
        source: 来源路径或标识，如 "data/某论文.pdf" 或 "manual_note_1"
    """
    try:
        store = _get_vector_store()
        store.delete_documents_by_source(source)
        return f"已删除来源为 '{source}' 的文档"
    except Exception as e:
        logger.error(f"[rag_delete_by_source] 失败: {e}", exc_info=True)
        return f"删除文档失败: {e}"


# ================================================================
# Tool: rag_get_stats — 统计信息
# ================================================================
@tool
def rag_get_stats() -> str:
    """获取 RAG 向量库的当前统计信息，包括文档总数、配置参数等。"""
    try:
        store = _get_vector_store()
        from agent_material.utils.config_handler import vector_store_config

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
            "multi_granularity": vector_store_config.get("multi_granularity", {}).get(
                "enabled", False
            ),
        }
        return json.dumps(stats, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"[rag_get_stats] 失败: {e}", exc_info=True)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ================================================================
# Tool: rag_get_task_status — 查询任务状态
# ================================================================
@tool
def rag_get_task_status(task_id: int) -> str:
    """根据任务 ID 查询后台处理任务的状态。

    Args:
        task_id: 任务 ID
    """
    try:
        queue = _get_task_queue()
        status = queue.get_status(task_id)
        if status is None:
            return json.dumps({"error": f"任务 {task_id} 不存在"}, ensure_ascii=False)
        return json.dumps(status, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"[rag_get_task_status] 失败: {e}", exc_info=True)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ================================================================
# Tool: rag_get_pending_tasks — 查询待处理任务
# ================================================================
@tool
def rag_get_pending_tasks() -> str:
    """获取任务队列的当前状态汇总（各状态任务数量）。"""
    try:
        queue = _get_task_queue()
        summary = queue.get_summary()
        return json.dumps(summary, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"[rag_get_pending_tasks] 失败: {e}", exc_info=True)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def get_local_rag_tools() -> list:
    """返回所有本地 RAG 工具列表。"""
    return [
        rag_search,
        rag_query,
        rag_load_documents,
        rag_add_documents,
        rag_delete_by_source,
        rag_get_stats,
        rag_get_task_status,
        rag_get_pending_tasks,
    ]
