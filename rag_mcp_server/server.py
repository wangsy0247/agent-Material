"""
RAG MCP Server（完全独立）

所有业务逻辑（vector_store, rag_service, text_spliter）和基础设施
（config, models, logger, file_loader, prompt_loader）均已内聚于本包。
不再通过 sys.path 注入依赖主项目的 agent_material 包。

提供的 Tools:
    rag_search          — 向量检索，返回相关文档片段
    rag_query           — RAG 问答（检索 + LLM 生成）
    rag_load_documents  — 扫描 data/ 目录并加载新文档
    rag_add_documents   — 手动添加文本到向量库
    rag_delete_by_source — 按来源路径删除文档
    rag_get_stats       — 获取向量库统计信息
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 延迟导入 RAG 模块（避免在模块顶层触发重量级初始化）
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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    mcp = FastMCP(
        "rag-server",
        instructions=(
            "材料科学领域 RAG（检索增强生成）服务。基于 PGVector + BM25 + CrossEncoder "
            "构建的混合检索系统，支持向量检索、文档加载、增量更新和智能问答。"
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
            "返回文档片段的文本内容、来源文件和元数据。"
        ),
        annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False, idempotentHint=True),
    )
    def rag_search(query: str) -> str:
        """检索与查询相关的文档片段。

        Args:
            query: 查询语句，如 "锂离子电池正极材料有哪些"
        """
        logger.info(f"[rag_search] query={query}")
        try:
            service = _get_rag_service()
            docs = service.retriever_docx(query)
            if not docs:
                return json.dumps({"results": [], "message": "未检索到相关文档"}, ensure_ascii=False)
            return json.dumps(
                {"results": _serialize_docs(docs), "count": len(docs)},
                ensure_ascii=False,
                indent=2,
            )
        except Exception as e:
            logger.error(f"[rag_search] 失败: {e}\n{traceback.format_exc()}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    # ================================================================
    # Tool: rag_query — RAG 问答
    # ================================================================
    @mcp.tool(
        name="rag_query",
        description=(
            "基于向量检索结果，调用大语言模型生成回答。"
            "适合需要综合多篇文献进行总结、分析、对比的复杂问题。"
        ),
        annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False, idempotentHint=False),
    )
    def rag_query(query: str) -> str:
        """RAG 问答：检索相关文档并生成回答。

        Args:
            query: 用户问题，如 "固态电解质的优缺点"
        """
        logger.info(f"[rag_query] query={query}")
        try:
            service = _get_rag_service()
            answer = service.get_summary(query)
            return answer
        except Exception as e:
            logger.error(f"[rag_query] 失败: {e}\n{traceback.format_exc()}")
            return f"RAG 问答失败: {e}"

    # ================================================================
    # Tool: rag_load_documents — 批量加载文档
    # ================================================================
    @mcp.tool(
        name="rag_load_documents",
        description=(
            "扫描配置的 data/ 目录，自动识别新增或修改的文档（PDF/TXT/MD/DOCX/PPTX），"
            "进行文本切分并写入 PGVector 向量库。基于 MD5 去重，已加载过的文件自动跳过。"
        ),
        annotations=ToolAnnotations(destructiveHint=True, openWorldHint=False, idempotentHint=False),
    )
    def rag_load_documents() -> str:
        """扫描并加载 data/ 目录下的文档到向量库。"""
        logger.info("[rag_load_documents] 开始加载文档")
        try:
            store = _get_vector_store()
            before = store.get_document_count()
            store.load_document()
            after = store.get_document_count()
            loaded = after - before
            msg = f"文档加载完成 | 加载前: {before} 条 | 加载后: {after} 条 | 新增: {loaded} 条"
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
    def rag_add_documents(
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
            store.add_documents(docs)
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
    def rag_delete_by_source(source: str) -> str:
        """按来源路径删除文档。

        Args:
            source: 来源路径或标识，如 "data/某论文.pdf" 或 "manual_note_1"
        """
        logger.info(f"[rag_delete_by_source] source={source}")
        try:
            store = _get_vector_store()
            store.delete_documents_by_source(source)
            return f"已触发删除来源为 '{source}' 的文档"
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
    def rag_get_stats() -> str:
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
                "splitter_type": vector_store_config.get("splitter_type", "recursive"),
                "chunk_size": vector_store_config.get("chunk_size"),
                "chunk_overlap": vector_store_config.get("chunk_overlap"),
                "vector_weight": vector_store_config.get("vector_weight", 0.7),
                "bm25_weight": vector_store_config.get("bm25_weight", 0.3),
                "enable_reranker": vector_store_config.get("enable_reranker", True),
                "top_n": vector_store_config.get("top_n", 3),
                "score_threshold": vector_store_config.get("score_threshold", 0.3),
            }
            return json.dumps(stats, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[rag_get_stats] 失败: {e}\n{traceback.format_exc()}")
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

    import asyncio

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
