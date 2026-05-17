"""
RAG MCP Server — 将 RAG 检索增强生成功能暴露为 MCP Tool

传输模式：
    - stdio: 适合本地子进程调用（如 Claude Desktop）
    - sse:  适合远程服务接入

用法：
    # stdio 模式
    python -m mcp_servers.tools.rag_mcp_server

    # sse 模式
    python -m mcp_servers.tools.rag_mcp_server --transport sse --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 将项目根目录加入 Python 路径（兼容多种启动方式）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp_servers.server import BaseMCPServer
from utils.logger import logger

# 懒加载 RAG Service（避免模块导入时初始化重量级组件）
_rag_service = None


def _get_rag_service():
    global _rag_service
    if _rag_service is None:
        from rag.rag_service import RagService

        _rag_service = RagService()
    return _rag_service


def create_server() -> BaseMCPServer:
    """创建并配置 RAG MCP Server 实例。"""
    server = BaseMCPServer(
        name="material-rag-server",
        instructions=(
            "材料科学 RAG 知识库服务。提供从向量数据库检索学术论文和文献，"
            "并基于检索结果生成专业回答的能力。"
        ),
    )

    @server.tool(
        name="rag_search",
        description=(
            "从材料科学向量知识库中检索相关文献，并生成综合回答。"
            "适用于查询材料性质、合成方法、应用场景等专业问题。"
        ),
    )
    def rag_search(query: str) -> str:
        """检索材料科学文献并生成回答。

        Args:
            query: 用户查询，如"石墨烯的导电性如何？"

        Returns:
            基于检索结果生成的综合回答
        """
        logger.info(f"[RAG MCP] 接收查询: {query[:80]}...")
        try:
            result = _get_rag_service().get_summary(query)
            logger.info(f"[RAG MCP] 查询完成 | 结果长度: {len(result)}")
            return result
        except Exception as e:
            logger.error(f"[RAG MCP] 查询失败: {e}")
            return f"检索失败: {e}"

    @server.tool(
        name="rag_retrieve",
        description=(
            "从材料科学向量知识库中仅执行检索，返回相关文档片段列表（不生成回答）。"
            "适用于需要查看原始文献出处的场景。"
        ),
    )
    def rag_retrieve(query: str) -> str:
        """检索相关文献片段，返回带来源的原始片段。

        Args:
            query: 用户查询

        Returns:
            JSON 格式的文档片段列表（包含来源文件名和内容）
        """
        import json

        logger.info(f"[RAG MCP] 检索文档: {query[:80]}...")
        try:
            docs = _get_rag_service().retriever_docx(query)
            results = []
            for i, doc in enumerate(docs, start=1):
                source = doc.metadata.get("source", "未知来源")
                results.append(
                    {
                        "index": i,
                        "source": source,
                        "content": doc.page_content[:500],  # 截断避免过长
                    }
                )
            return json.dumps(results, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[RAG MCP] 检索失败: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    return server


# ------------------------------------------------------------------
# CLI 入口
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="材料科学 RAG MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="传输模式: stdio(子进程) 或 sse(HTTP 流式)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="SSE 模式监听地址")
    parser.add_argument("--port", type=int, default=8001, help="SSE 模式监听端口")
    args = parser.parse_args()

    server = create_server()

    if args.transport == "stdio":
        server.run_stdio()
    else:
        server.run_sse(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
