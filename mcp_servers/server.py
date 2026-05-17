"""
MCP Server 基类 — 支持 stdio / sse 两种传输模式

基于 FastMCP 封装，提供统一的工具注册、生命周期管理和传输模式切换能力。

用法示例：
    # stdio 模式（子进程通信，适合本地工具）
    server = BaseMCPServer(name="my-local-server")
    
    @server.tool()
    def search_docs(query: str) -> str:
        return f"search result for {query}"
    
    if __name__ == "__main__":
        server.run_stdio()

    # sse 模式（HTTP 流式，适合远程服务）
    server = BaseMCPServer(name="my-remote-server", host="0.0.0.0", port=8001)
    
    @server.tool()
    def calc_area(length: float, width: float) -> float:
        return length * width
    
    if __name__ == "__main__":
        server.run_sse()

    # 嵌入 FastAPI
    from fastapi import FastAPI
    app = FastAPI()
    app.mount("/mcp", server.get_starlette_app(mount_path="/mcp"))
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable

from starlette.applications import Starlette

logger = logging.getLogger(__name__)


async def _default_lifespan(server: BaseMCPServer) -> AsyncGenerator[None, None]:
    """默认 lifespan：仅记录启动和关闭日志。"""
    logger.info(f"[MCP Server '{server.name}'] lifespan startup")
    yield
    logger.info(f"[MCP Server '{server.name}'] lifespan shutdown")


class BaseMCPServer:
    """MCP Server 基类。

    封装 FastMCP，提供：
    1. 统一的工具/资源/Prompt 注册接口
    2. stdio / sse 两种传输模式
    3. lifespan 生命周期管理
    4. 与 FastAPI 的集成能力

    Args:
        name: Server 名称（必须唯一，MCP Client 通过名称识别）
        instructions: 给 LLM 的系统指令，描述 Server 能力
        host: sse 模式监听地址，默认 127.0.0.1
        port: sse 模式监听端口，默认 8000
        mount_path: sse 模式挂载路径，默认 "/"
        sse_path: sse 端点路径，默认 "/sse"
        message_path: 消息 POST 端点路径，默认 "/messages/"
        lifespan: 自定义生命周期管理器（asynccontextmanager）
        debug: 是否开启调试日志
    """

    def __init__(
        self,
        name: str,
        instructions: str | None = None,
        host: str = "127.0.0.1",
        port: int = 8000,
        mount_path: str = "/",
        sse_path: str = "/sse",
        message_path: str = "/messages/",
        lifespan: Callable[[BaseMCPServer], AsyncGenerator[Any, None]] | None = None,
        debug: bool = False,
    ):
        self.name = name
        self._instructions = instructions or f"MCP Server: {name}"
        self._host = host
        self._port = port
        self._mount_path = mount_path
        self._sse_path = sse_path
        self._message_path = message_path
        self._debug = debug
        self._lifespan = lifespan or _default_lifespan

        # 延迟导入，避免模块加载时触发 Heavy Import
        from mcp.server import FastMCP

        self._mcp: FastMCP = FastMCP(
            name=name,
            instructions=self._instructions,
            host=host,
            port=port,
            mount_path=mount_path,
            sse_path=sse_path,
            message_path=message_path,
            lifespan=self._wrap_lifespan(),
            debug=debug,
        )

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _wrap_lifespan(
        self,
    ) -> Callable[[Any], AsyncGenerator[Any, None]]:
        """将 BaseMCPServer lifespan 适配为 FastMCP 所需的签名。"""
        outer_self = self

        @asynccontextmanager
        async def _adapter(app: Any) -> AsyncGenerator[Any, None]:
            async_gen = outer_self._lifespan(outer_self)
            try:
                result = await async_gen.asend(None)
                yield result
            finally:
                try:
                    await async_gen.aclose()
                except Exception as e:
                    logger.warning(f"[MCP Server '{outer_self.name}'] lifespan cleanup error: {e}")

        return _adapter

    def _tool_count(self) -> int:
        """同步获取已注册工具数量（内部使用）。"""
        return len(self._mcp._tool_manager._tools)

    def _tool_names(self) -> list[str]:
        """同步获取已注册工具名称列表（内部使用）。"""
        return list(self._mcp._tool_manager._tools.keys())

    # ------------------------------------------------------------------
    # 注册接口（代理到 FastMCP）
    # ------------------------------------------------------------------

    def tool(
        self,
        name: str | None = None,
        description: str | None = None,
        annotations: Any | None = None,
    ) -> Callable:
        """注册 MCP Tool。

        用法：
            @server.tool(name="rag_search", description="从知识库检索")
            def rag_search(query: str) -> str:
                ...

            @server.tool(
                name="delete_file",
                description="删除文件",
                annotations=ToolAnnotations(destructiveHint=True),
            )
            def delete_file(path: str) -> str:
                ...
        """
        return self._mcp.tool(name=name, description=description, annotations=annotations)

    def resource(self, uri: str, name: str | None = None) -> Callable:
        """注册 MCP Resource。"""
        return self._mcp.resource(uri=uri, name=name)

    def prompt(self, name: str) -> Callable:
        """注册 MCP Prompt。"""
        return self._mcp.prompt(name=name)

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------

    async def list_tools(self) -> list[Any]:
        """返回当前已注册的所有 Tool 元数据列表（异步）。"""
        return await self._mcp.list_tools()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """调用指定 Tool（异步）。

        Args:
            name: Tool 名称
            arguments: Tool 参数字典

        Returns:
            Tool 执行结果
        """
        return await self._mcp.call_tool(name, arguments)

    # ------------------------------------------------------------------
    # 传输模式启动
    # ------------------------------------------------------------------

    def run_stdio(self) -> None:
        """启动 stdio 传输模式（阻塞）。

        通过标准输入输出与 MCP Client 通信，适合本地子进程模式。
        这是 MCP 最经典的传输方式，Claude Desktop 等客户端原生支持。
        """
        logger.info(
            f"[MCP Server '{self.name}'] 启动 stdio 模式 | "
            f"tools={self._tool_count()}"
        )
        try:
            asyncio.run(self._mcp.run_stdio_async())
        except KeyboardInterrupt:
            logger.info(f"[MCP Server '{self.name}'] stdio 模式被用户中断")

    def run_sse(self, host: str | None = None, port: int | None = None) -> None:
        """启动 SSE 传输模式（阻塞）。

        启动独立的 HTTP 服务，通过 Server-Sent Events 与 MCP Client 通信。
        适合远程服务或多语言客户端接入。

        Args:
            host: 覆盖构造时的 host
            port: 覆盖构造时的 port
        """
        if host:
            self._mcp.settings.host = host
        if port:
            self._mcp.settings.port = port

        actual_host = self._mcp.settings.host
        actual_port = self._mcp.settings.port

        logger.info(
            f"[MCP Server '{self.name}'] 启动 sse 模式 | "
            f"host={actual_host} port={actual_port} "
            f"mount_path={self._mcp.settings.mount_path} "
            f"tools={self._tool_count()}"
        )
        try:
            asyncio.run(self._mcp.run_sse_async())
        except KeyboardInterrupt:
            logger.info(f"[MCP Server '{self.name}'] sse 模式被用户中断")

    def get_starlette_app(self, mount_path: str | None = None) -> Starlette:
        """获取 SSE Starlette Application，用于嵌入 FastAPI。

        用法：
            from fastapi import FastAPI
            app = FastAPI()
            app.mount("/mcp", server.get_starlette_app(mount_path="/mcp"))

        Args:
            mount_path: 在 FastAPI 中的挂载路径，需与 mount() 参数一致

        Returns:
            Starlette 应用实例（包含 /sse 和 /messages/ 端点）
        """
        return self._mcp.sse_app(mount_path=mount_path)

    # ------------------------------------------------------------------
    # 属性暴露
    # ------------------------------------------------------------------

    @property
    def instructions(self) -> str:
        return self._instructions

    @instructions.setter
    def instructions(self, value: str) -> None:
        self._instructions = value

    @property
    def fastmcp(self) -> Any:
        """暴露底层 FastMCP 实例（高级用法）。"""
        return self._mcp

    def __repr__(self) -> str:
        return (
            f"<BaseMCPServer name='{self.name}' "
            f"tools={self._tool_count()} "
            f"host={self._host} port={self._port}>"
        )
