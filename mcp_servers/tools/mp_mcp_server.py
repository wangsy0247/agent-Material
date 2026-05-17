"""
Materials Project MCP Server — 完整的 MCP 标准实现

提供能力：
    Tools（工具调用）:
        1. mp_download  — 按化学式下载材料结构（POSCAR）和属性文件
        2. mp_query     — 查询材料属性信息（带隙、密度、空间群等）
        3. mp_list      — 列出本地下载历史

    Resources（数据资源）:
        1. materials://local-downloads   — 本地下载的材料历史
        2. materials://workspace-path    — 工作空间路径
        3. materials://stats             — 下载统计信息

    Prompts（提示模板）:
        1. material-analysis        — 材料结构分析
        2. bandgap-explanation      — 带隙解释
        3. crystal-structure-guide  — 晶体结构分析指南

传输模式：
    - stdio: 适合本地子进程调用
    - sse:  适合远程服务接入

用法：
    # stdio 模式
    python -m mcp_servers.tools.mp_mcp_server

    # sse 模式
    python -m mcp_servers.tools.mp_mcp_server --transport sse --host 0.0.0.0 --port 8002
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# 将项目根目录加入 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp_servers.server import BaseMCPServer
from mcp.types import ToolAnnotations
from utils.logger import logger

# 懒加载 MP Downloader
_mp_downloader = None


def _get_mp_downloader():
    global _mp_downloader
    if _mp_downloader is None:
        from agent.tools.mp_downloader import MaterialsProjectDownloader

        _mp_downloader = MaterialsProjectDownloader()
    return _mp_downloader


def create_server() -> BaseMCPServer:
    """创建并配置 Materials Project MCP Server 实例。"""
    server = BaseMCPServer(
        name="material-project-server",
        instructions=(
            "Materials Project 材料数据库服务。提供材料结构下载、属性查询、"
            "下载历史管理等能力，覆盖晶体学信息、电子结构、热力学性质等。"
        ),
    )

    # ================================================================
    # Tools（工具调用）
    # ================================================================

    @server.tool(
        name="mp_download",
        description=(
            "根据化学式从 Materials Project 数据库下载材料结构（POSCAR）和属性文件。"
            "若该材料已存在于本地下载历史中，会返回重复提示。"
        ),
        annotations=ToolAnnotations(
            destructiveHint=True,    # 会修改文件系统（写入 POSCAR）
            openWorldHint=True,      # 会访问外部网络（MP API）
            idempotentHint=True,     # 重复调用结果一致（有去重机制）
        ),
    )
    def mp_download(formula: str) -> str:
        """下载指定化学式的材料。

        Args:
            formula: 材料化学式，如 "LiFePO4", "ZnO", "Si"

        Returns:
            下载结果描述，包含保存路径或重复提示
        """
        logger.info(f"[MP MCP] 下载材料: {formula}")
        try:
            result = _get_mp_downloader().download(formula)
            logger.info(f"[MP MCP] 下载完成: {formula}")
            return result
        except Exception as e:
            logger.error(f"[MP MCP] 下载失败: {e}")
            return f"下载失败: {e}"

    @server.tool(
        name="mp_query",
        description=(
            "查询指定化学式材料的属性信息，直接从 Materials Project 数据库获取。"
            "返回带隙、密度、空间群、晶格参数等关键物理属性。"
        ),
        annotations=ToolAnnotations(
            readOnlyHint=True,       # 只读查询，不修改任何状态
            openWorldHint=True,      # 访问外部网络
            idempotentHint=True,     # 重复调用结果一致
        ),
    )
    def mp_query(formula: str) -> str:
        """查询材料属性。

        Args:
            formula: 材料化学式，如 "LiFePO4", "ZnO"

        Returns:
            JSON 格式的材料属性字典
        """
        logger.info(f"[MP MCP] 查询属性: {formula}")
        try:
            props = _get_mp_downloader().query_material_properties(formula)
            if props is None:
                return json.dumps(
                    {"error": f"未找到化学式 '{formula}' 对应的材料"},
                    ensure_ascii=False,
                )
            return json.dumps(props, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.error(f"[MP MCP] 查询失败: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    @server.tool(
        name="mp_list",
        description="列出本地已从 Materials Project 下载的材料历史记录",
        annotations=ToolAnnotations(
            readOnlyHint=True,       # 只读查询
            idempotentHint=True,
        ),
    )
    def mp_list() -> str:
        """查看本地下载历史。

        Returns:
            已下载材料列表，每行一个 "化学式 (MP-ID)"
        """
        logger.info("[MP MCP] 查询下载历史")
        try:
            result = _get_mp_downloader().list_history()
            return result
        except Exception as e:
            logger.error(f"[MP MCP] 查询历史失败: {e}")
            return f"查询失败: {e}"

    # ================================================================
    # Resources（数据资源）
    # ================================================================

    @server.resource(
        uri="materials://local-downloads",
        name="本地已下载材料列表",
    )
    def local_downloads_resource() -> str:
        """Resource: 本地下载的材料历史记录。

        MCP Client 可以通过 URI `materials://local-downloads` 读取此资源。
        适用于 LLM 需要了解本地已有材料数据的场景。
        """
        logger.info("[MP MCP Resource] 读取 local-downloads")
        try:
            return _get_mp_downloader().list_history()
        except Exception as e:
            logger.error(f"[MP MCP Resource] 读取失败: {e}")
            return f"读取失败: {e}"

    @server.resource(
        uri="materials://workspace-path",
        name="工作空间路径",
    )
    def workspace_path_resource() -> str:
        """Resource: 材料下载工作空间的绝对路径。"""
        downloader = _get_mp_downloader()
        return downloader.workspace_dir

    @server.resource(
        uri="materials://stats",
        name="下载统计信息",
    )
    def stats_resource() -> str:
        """Resource: 本地下载统计（总数、最近下载等）。"""
        try:
            formulas = _get_mp_downloader().get_downloaded_formulas()
            history_file = _get_mp_downloader().history_file
            total = len(formulas)

            stats = {
                "total_downloaded": total,
                "history_file": history_file,
                "workspace_dir": _get_mp_downloader().workspace_dir,
                "recent_formulas": sorted(list(formulas))[:10],
            }
            return json.dumps(stats, ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    # ================================================================
    # Prompts（提示模板）
    # ================================================================

    @server.prompt(name="material-analysis")
    def material_analysis_prompt(formula: str) -> str:
        """Prompt: 材料综合分析模板。

        LLM 可以使用此 Prompt 作为系统提示，对指定材料进行专业分析。

        Args:
            formula: 材料化学式
        """
        return (
            f"你是一位材料科学专家。请对化学式为 {formula} 的材料进行综合分析，"
            f"包括但不限于以下方面：\n\n"
            f"1. 晶体结构分析（晶系、空间群、晶格参数）\n"
            f"2. 电子结构特性（带隙、导电类型）\n"
            f"3. 热力学稳定性（形成能）\n"
            f"4. 潜在应用场景\n"
            f"5. 合成方法建议\n\n"
            f"请基于 Materials Project 数据库提供的数据进行分析，"
            f"并给出具体的数据支撑。"
        )

    @server.prompt(name="bandgap-explanation")
    def bandgap_explanation_prompt(bandgap_value: float) -> str:
        """Prompt: 带隙数值解释模板。

        Args:
            bandgap_value: 带隙数值（eV）
        """
        if bandgap_value < 0.1:
            category = " metallic / semi-metallic"
        elif bandgap_value < 1.0:
            category = " narrow-gap semiconductor"
        elif bandgap_value < 3.0:
            category = " semiconductor"
        else:
            category = " insulator / wide-gap semiconductor"

        return (
            f"你正在分析一种带隙为 {bandgap_value} eV 的材料。\n\n"
            f"根据带隙数值，该材料属于：{category}\n\n"
            f"请解释：\n"
            f"1. 这个带隙值在材料科学中的意义\n"
            f"2. 该材料适合哪些电子/光电应用\n"
            f"3. 与常见材料（Si: 1.1eV, GaAs: 1.4eV, TiO2: 3.0eV）的对比\n"
            f"4. 是否适合作为太阳能电池、LED 或透明导电材料"
        )

    @server.prompt(name="crystal-structure-guide")
    def crystal_structure_guide_prompt() -> str:
        """Prompt: 晶体结构分析指南（无参数模板）。"""
        return (
            "你是一位晶体学专家。当用户提供材料结构数据时，请按以下框架进行分析：\n\n"
            "1. 结构识别\n"
            "   - 晶系、布拉维格子、空间群\n"
            "   - 与常见结构的同构关系\n\n"
            "2. 配位环境\n"
            "   - 阳离子配位数、配位多面体形状\n"
            "   - 键长、键角分析\n\n"
            "3. 结构稳定性\n"
            "   - 容忍因子（如适用）\n"
            "   - 与已知稳定结构的对比\n\n"
            "4. 结构-性能关系\n"
            "   - 结构特征如何影响离子传输、电子传导等性质\n\n"
            "请使用 Materials Project 提供的结构数据进行分析。"
        )

    return server


# ------------------------------------------------------------------
# CLI 入口
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Materials Project MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="传输模式: stdio(子进程) 或 sse(HTTP 流式)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="SSE 模式监听地址")
    parser.add_argument("--port", type=int, default=8002, help="SSE 模式监听端口")
    args = parser.parse_args()

    server = create_server()

    # 打印 Server 能力概览
    tool_names = server._tool_names()
    print(f"[MP MCP Server] 启动中...")
    print(f"  名称: {server.name}")
    print(f"  工具: {tool_names}")
    print(f"  传输: {args.transport}")

    if args.transport == "stdio":
        server.run_stdio()
    else:
        server.run_sse(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
