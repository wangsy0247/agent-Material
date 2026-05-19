"""
Materials Project MCP Server

提供 3 个 MCP Tool：
    mp_download  — 下载材料结构（POSCAR + 属性 JSON）
    mp_query     — 查询材料属性（带隙、密度、空间群等）
    mp_list      — 列出本地下载历史

支持 stdio 和 SSE 两种传输模式，与 Agent 项目完全解耦。
"""

from __future__ import annotations

import argparse
import json
import logging

from mcp.server import FastMCP
from mcp.types import ToolAnnotations

from .downloader import MaterialsProjectDownloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 懒加载
_downloader: MaterialsProjectDownloader | None = None


def _get_downloader() -> MaterialsProjectDownloader:
    global _downloader
    if _downloader is None:
        _downloader = MaterialsProjectDownloader()
    return _downloader


def create_server(host: str = "127.0.0.1", port: int = 8002) -> FastMCP:
    """创建并配置 MCP Server。"""
    mcp = FastMCP(
        "material-project-server",
        instructions=(
            "Materials Project 材料数据库服务。提供材料结构下载、属性查询、"
            "下载历史管理等能力，覆盖晶体学信息、电子结构、热力学性质等。"
        ),
        host=host,
        port=port,
    )

    # ================================================================
    # Tools
    # ================================================================

    @mcp.tool(
        name="mp_download",
        description=(
            "根据化学式从 Materials Project 数据库下载材料结构（POSCAR）和属性文件。"
            "若该材料已存在于本地下载历史中，会返回重复提示。"
        ),
        annotations=ToolAnnotations(
            destructiveHint=True,
            openWorldHint=True,
            idempotentHint=True,
        ),
    )
    def mp_download(formula: str) -> str:
        """下载指定化学式的材料。

        Args:
            formula: 材料化学式，如 "LiFePO4", "ZnO", "Si"
        """
        logger.info(f"下载材料: {formula}")
        try:
            result = _get_downloader().download(formula)
            logger.info(f"下载完成: {formula}")
            return result
        except Exception as e:
            logger.error(f"下载失败: {e}")
            return f"下载失败: {e}"

    @mcp.tool(
        name="mp_query",
        description=(
            "查询指定化学式材料的属性信息，直接从 Materials Project 数据库获取。"
            "返回带隙、密度、空间群、晶格参数等关键物理属性。"
        ),
        annotations=ToolAnnotations(
            readOnlyHint=True,
            openWorldHint=True,
            idempotentHint=True,
        ),
    )
    def mp_query(formula: str) -> str:
        """查询材料属性。

        Args:
            formula: 材料化学式，如 "LiFePO4", "ZnO"
        """
        logger.info(f"查询属性: {formula}")
        try:
            props = _get_downloader().query_material_properties(formula)
            if props is None:
                return json.dumps(
                    {"error": f"未找到化学式 '{formula}' 对应的材料"},
                    ensure_ascii=False,
                )
            return json.dumps(props, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    @mcp.tool(
        name="mp_list",
        description="列出本地已从 Materials Project 下载的材料历史记录",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
    )
    def mp_list() -> str:
        """查看本地下载历史。"""
        logger.info("查询下载历史")
        try:
            return _get_downloader().list_history()
        except Exception as e:
            logger.error(f"查询历史失败: {e}")
            return f"查询失败: {e}"

    return mcp


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

    server = create_server(host=args.host, port=args.port)

    import asyncio, sys
    tool_list = asyncio.run(server.list_tools())
    tool_names = [t.name for t in tool_list]
    print(f"[MP MCP Server] 启动中...", file=sys.stderr)
    print(f"  名称: {server.name}", file=sys.stderr)
    print(f"  工具: {tool_names}", file=sys.stderr)
    print(f"  传输: {args.transport}", file=sys.stderr)

    import asyncio
    if args.transport == "stdio":
        asyncio.run(server.run_stdio_async())
    else:
        asyncio.run(server.run_sse_async())


if __name__ == "__main__":
    main()
