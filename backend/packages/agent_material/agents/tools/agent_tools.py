"""
Agent 工具集合 — 支持本地直连 + MCP 混合模式

RAG 工具已从 MCP 解耦为本地直连（进程内调用 agent_material.rag_engine）：
    - rag_search
    - rag_query
    - rag_load_documents
    - rag_add_documents
    - rag_delete_by_source
    - rag_get_stats
    - rag_get_task_status
    - rag_get_pending_tasks

其他工具（如 material-project）仍通过 MCP 加载：
    extensions_config.json → mcpServers → material-project
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from agent_material.utils.config_handler import agent_config
from agent_material.utils.logger import logger


# ================================================================
# 工具获取（local + MCP 混合模式）
# ================================================================


def get_all_tools() -> list[BaseTool]:
    """获取全部工具：本地 RAG 工具 + MCP 工具。

    mp_tool_mode=local 时优先加载本地 RAG 工具，
    其余 MCP 工具从 extensions_config.json 中启用的服务器加载。
    """
    mp_tool_mode = agent_config.get("mp_tool_mode", "local")
    tools: list[BaseTool] = []

    # 1. 本地 RAG 工具
    if mp_tool_mode == "local":
        try:
            from agent_material.agents.tools.rag_local_tools import (
                get_local_rag_tools,
            )

            rag_tools = get_local_rag_tools()
            tools.extend(rag_tools)
            logger.info(
                f"[AgentTools] 本地 RAG 工具 | 工具数={len(rag_tools)} | "
                f"工具名={[t.name for t in rag_tools]}"
            )
        except Exception as e:
            logger.warning(f"[AgentTools] 本地 RAG 工具加载失败: {e}")

    # 2. MCP 工具（extensions_config.json 中启用的服务器）
    try:
        from agent_material.mcp.cache import get_cached_mcp_tools

        mcp_tools = get_cached_mcp_tools()
        tools.extend(mcp_tools)
        logger.info(
            f"[AgentTools] MCP 工具 | 工具数={len(mcp_tools)} | "
            f"工具名={[t.name for t in mcp_tools]}"
        )
    except Exception as e:
        logger.warning(f"[AgentTools] MCP 工具获取失败: {e}")

    logger.info(
        f"[AgentTools] 工具总数={len(tools)} | "
        f"模式={mp_tool_mode}"
    )
    return tools
