"""
Agent 工具集合 — 全部通过 MCP 获取

RAG 能力已解耦至 rag_mcp_server/，Agent 通过 MCP 调用：
    - rag_search
    - rag_query
    - rag_load_documents
    - rag_add_documents
    - rag_delete_by_source
    - rag_get_stats

配置方式：extensions_config.json → mcpServers → rag-server
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from agent_material.utils.config_handler import agent_config
from agent_material.utils.logger import logger


# ================================================================
# 工具获取（仅 MCP 模式）
# ================================================================


def get_all_tools() -> list[BaseTool]:
    """通过 MCP 获取全部工具。

    远程调用。若 MCP 连接失败，Agent 将只剩下非 MCP 工具可用。
    """
    try:
        from agent_material.mcp.cache import get_cached_mcp_tools
        mcp_tools = get_cached_mcp_tools()
        tool_names = [t.name for t in mcp_tools]
        logger.info(
            f"[AgentTools] MCP 模式 | 工具数={len(mcp_tools)} | 工具名={tool_names}"
        )
        return mcp_tools
    except Exception as e:
        logger.warning(f"[AgentTools] MCP 工具获取失败: {e}")
        return []
