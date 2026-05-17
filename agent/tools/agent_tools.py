"""
Agent 工具集合 — 原生 LangChain Tool + MCP 工具

提供统一的工具获取接口：
    - get_native_tools(): 仅返回原生 LangChain 工具
    - get_all_tools(): 返回原生工具 + MCP 工具（混合模式）

MCP 工具初始化：
    需要在应用 lifespan startup 中调用 init_mcp_tools()，
    否则 get_all_tools() 只会返回原生工具（自动降级）。
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from agent.tools.mp_downloader import MaterialsProjectDownloader
from utils.logger import logger

# 延迟初始化：避免模块导入时就实例化重量级对象
_rag_service = None
_mp_downloader = None


def _get_rag_service():
    """懒加载 RagService，避免导入时初始化 VectorStore / 重排序模型。"""
    global _rag_service
    if _rag_service is None:
        from rag.rag_service import RagService

        _rag_service = RagService()
    return _rag_service


def _get_mp_downloader():
    """懒加载 MaterialsProjectDownloader。"""
    global _mp_downloader
    if _mp_downloader is None:
        _mp_downloader = MaterialsProjectDownloader()
    return _mp_downloader


# ================================================================
# 原生 LangChain Tools
# ================================================================

@tool(description="从向量数据库中检索参考资料并进行总结")
def rag_summary(query: str) -> str:
    return _get_rag_service().get_summary(query)


@tool(
    description="根据化学式从 Materials Project 数据库下载材料结构（POSCAR）和属性文件"
)
def mp_download_tool(formula: str) -> str:
    """
    调用 Materials Project 下载工具，检索并保存指定化学式的材料。
    若该材料已存在于本地历史记录中，会返回重复提示。
    """
    return _get_mp_downloader().download(formula)


@tool(description="查询指定化学式材料的属性信息，直接从 Materials Project 数据库获取")
def mp_query_tool(formula: str) -> dict:
    """
    直接查询 Materials Project 数据库，获取指定化学式材料的属性信息。
    """
    return _get_mp_downloader().query_material_properties(formula)


@tool(description="列出本地已从 Materials Project 下载的材料历史记录")
def mp_list_tool() -> str:
    """
    查看本地已下载的 Materials Project 材料列表。
    """
    return _get_mp_downloader().list_history()


# ================================================================
# 工具获取接口
# ================================================================

def get_native_tools() -> list[BaseTool]:
    """获取原生 LangChain 工具列表。

    Returns:
        4 个原生工具：rag_summary, mp_download_tool, mp_query_tool, mp_list_tool
    """
    return [rag_summary, mp_download_tool, mp_query_tool, mp_list_tool]


def get_all_tools() -> list[BaseTool]:
    """获取全部工具：原生工具 + MCP 工具。

    如果 MCP 工具已初始化（在 lifespan 中调用了 init_mcp_tools()），
    则返回混合列表；否则仅返回原生工具（自动降级）。

    Returns:
        BaseTool 列表，可直接传入 create_agent(tools=...)
    """
    tools = get_native_tools()

    try:
        from agent.tools.mcp_tool_manager import get_mcp_tools
        mcp_tools = get_mcp_tools()
        tools = tools + mcp_tools
        logger.info(
            f"[AgentTools] 工具加载完成 | 原生={len(get_native_tools())} "
            f"MCP={len(mcp_tools)} 总计={len(tools)}"
        )
    except RuntimeError:
        logger.warning(
            "[AgentTools] MCP 工具未初始化，仅使用原生工具。"
            "请在应用 lifespan startup 中调用 init_mcp_tools() 以启用 MCP 工具。"
        )
    except Exception as e:
        logger.error(f"[AgentTools] 加载 MCP 工具失败: {e}，回退到原生工具")

    return tools
