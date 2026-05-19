"""
Agent 工具集合 — 支持本地/MCP 双模式

本地模式 (mp_tool_mode: "local")：
    - rag_summary, mp_download, mp_query, mp_list 全部进程内直连

MCP 模式 (mp_tool_mode: "mcp")：
    - rag_summary 本地，mp_* 工具通过 MCP Server 远程调用
    - MCP Server 独立部署在 mp_material_server/

切换方式：修改 config/agent_config.yaml → mp_tool_mode
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from utils.config_handler import agent_config
from utils.logger import logger

# 延迟初始化
_rag_service = None
_mp_downloader = None


def _get_rag_service():
    global _rag_service
    if _rag_service is None:
        from rag.rag_service import RagService
        _rag_service = RagService()
    return _rag_service


def _get_mp_downloader():
    global _mp_downloader
    if _mp_downloader is None:
        from agent.tools.mp_downloader import MaterialsProjectDownloader
        _mp_downloader = MaterialsProjectDownloader()
    return _mp_downloader


# ================================================================
# 本地 LangChain Tools
# ================================================================

@tool(description="从向量数据库中检索参考资料并进行总结")
def rag_summary(query: str) -> str:
    """从材料科学向量知识库中检索相关文献，并基于检索结果生成综合回答。"""
    return _get_rag_service().get_summary(query)


@tool(description="根据化学式从 Materials Project 下载材料结构（POSCAR）和属性文件")
def mp_download(formula: str) -> str:
    """下载指定化学式的材料结构文件和属性。"""
    return _get_mp_downloader().download(formula)


@tool(description="查询指定化学式材料的属性信息（带隙、密度、空间群、晶格参数等）")
def mp_query(formula: str) -> str:
    """查询材料物理属性，从 Materials Project 数据库获取。"""
    import json
    props = _get_mp_downloader().query_material_properties(formula)
    if props is None:
        return json.dumps({"error": f"未找到化学式 '{formula}' 对应的材料"}, ensure_ascii=False)
    return json.dumps(props, ensure_ascii=False, indent=2, default=str)


@tool(description="列出本地已从 Materials Project 下载的材料历史记录")
def mp_list() -> str:
    """查看本地下载历史。"""
    return _get_mp_downloader().list_history()


def get_native_tools() -> list[BaseTool]:
    """获取本地直连工具（不含 MCP）。"""
    return [rag_summary, mp_download, mp_query, mp_list]


# ================================================================
# 双模式工具获取
# ================================================================


def get_all_tools() -> list[BaseTool]:
    """根据 config/agent_config.yaml 中的 mp_tool_mode 自动切换。

    "local" — 全部工具进程内直连（默认，低延迟）
    "mcp"   — mp_* 工具通过 MCP Server 远程调用（需先在 lifespan 中初始化 MCP 连接）
    """
    mode = agent_config.get("mp_tool_mode", "local")

    if mode == "mcp":
        try:
            from agent.tools.mcp_adapter import get_mcp_tools
            mcp_tools = get_mcp_tools()
            logger.info(
                f"[AgentTools] MCP 模式 | 远程={len(mcp_tools)} 本地=1 | "
                f"工具名={[t.name for t in mcp_tools] + ['rag_summary']}"
            )
            return [rag_summary] + mcp_tools
        except RuntimeError as e:
            logger.warning(f"[AgentTools] MCP 未初始化，回退到本地模式: {e}")

    # 本地模式（默认 + MCP 失败时回退）
    tools = [rag_summary, mp_download, mp_query, mp_list]
    logger.info(
        f"[AgentTools] 本地模式 | 工具数={len(tools)} | "
        f"工具名={[t.name for t in tools]}"
    )
    return tools
