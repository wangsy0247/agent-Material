"""
Agent 工具集合 — 完全解耦架构

本地工具（原生 LangChain Tool）：
    - rag_summary: RAG 检索 + 总结（直接函数调用，不走 MCP）

MCP 工具（远程 JSON-RPC）：
    - mp_download: 下载材料结构（通过 MCP Server）
    - mp_query: 查询材料属性（通过 MCP Server）
    - mp_list: 列出下载历史（通过 MCP Server）

解耦原则：
    1. RAG 能力保留在本地（进程内调用，低延迟）
    2. Materials Project 能力完全通过 MCP 暴露（独立部署）
    3. 不再存在功能重复的两套实现

使用：
    from agent.tools.agent_tools import get_all_tools
    tools = get_all_tools()  # 本地 RAG + MCP MP
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from utils.logger import logger

# 延迟初始化：避免模块导入时就实例化重量级对象
_rag_service = None


def _get_rag_service():
    """懒加载 RagService，避免导入时初始化 VectorStore / 重排序模型。"""
    global _rag_service
    if _rag_service is None:
        from rag.rag_service import RagService

        _rag_service = RagService()
    return _rag_service


# ================================================================
# 本地 LangChain Tools（RAG — 进程内直接调用）
# ================================================================

@tool(description="从向量数据库中检索参考资料并进行总结")
def rag_summary(query: str) -> str:
    """从材料科学向量知识库中检索相关文献，并基于检索结果生成综合回答。

    适用于查询材料性质、合成方法、应用场景等专业问题。

    Args:
        query: 用户查询，如"石墨烯的导电性如何？"

    Returns:
        基于检索结果生成的综合回答
    """
    return _get_rag_service().get_summary(query)


# ================================================================
# 工具获取接口
# ================================================================

def get_native_tools() -> list[BaseTool]:
    """获取本地原生 LangChain 工具列表。

    Returns:
        本地工具：rag_summary（RAG 检索 + 总结）
    """
    return [rag_summary]


def get_all_tools() -> list[BaseTool]:
    """获取全部工具：本地 RAG 工具 + MCP Materials Project 工具。

    本地工具：rag_summary（进程内直接调用）
    MCP 工具：mp_download, mp_query, mp_list（通过 MCP Server 远程调用）

    如果 MCP 工具已初始化（在 lifespan 中调用了 init_mcp_tools()），
    则返回完整列表；否则仅返回本地工具。

    Returns:
        BaseTool 列表，可直接传入 create_agent(tools=...)
    """
    tools = get_native_tools()

    try:
        from agent.tools.mcp_tool_manager import get_mcp_tools
        mcp_tools = get_mcp_tools()
        tools = tools + mcp_tools
        logger.info(
            f"[AgentTools] 工具加载完成 | 本地={len(get_native_tools())} "
            f"MCP={len(mcp_tools)} 总计={len(tools)}"
        )
    except RuntimeError:
        logger.warning(
            "[AgentTools] MCP 工具未初始化，仅使用本地 RAG 工具。"
            "Materials Project 相关功能（下载/查询/列表）将不可用。"
            "请在应用 lifespan startup 中调用 init_mcp_tools() 以启用 MCP 工具。"
        )
    except Exception as e:
        logger.error(f"[AgentTools] 加载 MCP 工具失败: {e}")

    return tools
