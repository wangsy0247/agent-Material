"""
MCP Client — 连接外部 MCP Server，支持 stdio 和 sse 两种传输模式

用法示例：
    # stdio 模式（连接本地子进程 Server）
    async with MCPClient.stdio("python", "-m", "mcp_servers.tools.rag_mcp_server") as client:
        tools = await client.list_tools()
        result = await client.call_tool("rag_search", {"query": "石墨烯导电性"})

    # sse 模式（连接远程 HTTP Server）
    async with MCPClient.sse("http://localhost:8001/sse") as client:
        tools = await client.list_tools()
        result = await client.call_tool("rag_search", {"query": "石墨烯导电性"})
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Sequence

from mcp.client.session import ClientSession
from mcp.types import CallToolResult, TextContent

logger = logging.getLogger(__name__)


@dataclass
class ToolInfo:
    """MCP Tool 元数据。"""

    name: str
    description: str
    parameters: dict[str, Any]


class MCPClient:
    """MCP 客户端，封装 stdio / sse 连接和工具调用。

    两种使用模式：
        1. 上下文管理器模式（推荐）：自动管理连接生命周期
        2. 手动模式：需要显式调用 connect() / close()

    Args:
        session: 底层 ClientSession 实例（由 connect 方法创建）
    """

    def __init__(self, session: ClientSession | None = None):
        self._session = session
        self._tools: list[ToolInfo] | None = None

    # ------------------------------------------------------------------
    # 工厂方法：stdio / sse 连接
    # ------------------------------------------------------------------

    @classmethod
    @asynccontextmanager
    async def stdio(
        cls,
        command: str,
        *args: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> AsyncGenerator[MCPClient, None]:
        """通过 stdio 连接本地 MCP Server（子进程模式）。

        Args:
            command: 可执行文件路径，如 "python", "node", "uv"
            args: 传递给命令的参数，如 "-m", "mcp_servers.tools.rag_mcp_server"
            env: 额外的环境变量
            cwd: 子进程工作目录

        用法：
            async with MCPClient.stdio("python", "-m", "mcp_servers.tools.rag_mcp_server") as client:
                result = await client.call_tool("rag_search", {"query": "..."})
        """
        from mcp.client.stdio import stdio_client, StdioServerParameters

        server_params = StdioServerParameters(
            command=command,
            args=list(args),
            env=env,
        )

        logger.info(f"[MCP Client] 连接 stdio Server: {command} {' '.join(args)}")

        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                logger.info("[MCP Client] stdio 连接初始化完成")
                client = cls(session)
                await client._refresh_tools()
                yield client

    @classmethod
    @asynccontextmanager
    async def sse(
        cls,
        url: str,
        headers: dict[str, Any] | None = None,
        timeout: float = 5.0,
    ) -> AsyncGenerator[MCPClient, None]:
        """通过 SSE 连接远程 MCP Server（HTTP 流式模式）。

        Args:
            url: SSE 端点 URL，如 "http://localhost:8001/sse"
            headers: 自定义请求头（如认证 Token）
            timeout: 连接超时时间（秒）

        用法：
            async with MCPClient.sse("http://localhost:8001/sse") as client:
                result = await client.call_tool("rag_search", {"query": "..."})
        """
        from mcp.client.sse import sse_client

        logger.info(f"[MCP Client] 连接 SSE Server: {url}")

        async with sse_client(url, headers=headers, timeout=timeout) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                logger.info("[MCP Client] SSE 连接初始化完成")
                client = cls(session)
                await client._refresh_tools()
                yield client

    # ------------------------------------------------------------------
    # 工具管理
    # ------------------------------------------------------------------

    async def _refresh_tools(self) -> None:
        """从 Server 刷新工具列表。"""
        if self._session is None:
            raise RuntimeError("Client 未连接，请先通过 stdio() 或 sse() 建立连接")

        response = await self._session.list_tools()
        self._tools = []
        for tool in response.tools:
            self._tools.append(
                ToolInfo(
                    name=tool.name,
                    description=tool.description or "",
                    parameters=tool.inputSchema,
                )
            )
        logger.info(f"[MCP Client] 发现 {len(self._tools)} 个工具")

    @property
    def tools(self) -> list[ToolInfo]:
        """当前 Server 提供的所有 Tool 元数据。"""
        if self._tools is None:
            raise RuntimeError("工具列表未加载，请先建立连接")
        return self._tools

    def get_tool_names(self) -> list[str]:
        """返回所有 Tool 名称列表。"""
        return [t.name for t in self.tools]

    # ------------------------------------------------------------------
    # 工具调用
    # ------------------------------------------------------------------

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> str:
        """调用指定 Tool。

        Args:
            name: Tool 名称
            arguments: Tool 参数字典

        Returns:
            Tool 返回的文本内容（如果返回多个内容块，用换行拼接）
        """
        if self._session is None:
            raise RuntimeError("Client 未连接")

        logger.info(f"[MCP Client] 调用工具: {name} | args={arguments}")

        result: CallToolResult = await self._session.call_tool(name, arguments)

        # 提取文本内容
        texts = []
        for content in result.content:
            if isinstance(content, TextContent):
                texts.append(content.text)
            else:
                texts.append(str(content))

        output = "\n".join(texts)
        logger.info(f"[MCP Client] 工具返回: {len(output)} 字符")
        return output

    async def call_tool_raw(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> CallToolResult:
        """调用指定 Tool，返回原始结果（包含 isError 标志）。"""
        if self._session is None:
            raise RuntimeError("Client 未连接")
        return await self._session.call_tool(name, arguments)

    # ------------------------------------------------------------------
    # 资源与 Prompt（可选能力）
    # ------------------------------------------------------------------

    async def list_resources(self) -> list[dict[str, Any]]:
        """列出 Server 提供的所有 Resource。"""
        if self._session is None:
            raise RuntimeError("Client 未连接")
        response = await self._session.list_resources()
        return [{"uri": r.uri, "name": r.name, "mimeType": r.mimeType} for r in response.resources]

    async def read_resource(self, uri: str) -> str:
        """读取指定 Resource 内容。"""
        if self._session is None:
            raise RuntimeError("Client 未连接")
        result = await self._session.read_resource(uri)
        texts = []
        for content in result.contents:
            if hasattr(content, "text"):
                texts.append(content.text)
        return "\n".join(texts)

    async def list_prompts(self) -> list[dict[str, Any]]:
        """列出 Server 提供的所有 Prompt。"""
        if self._session is None:
            raise RuntimeError("Client 未连接")
        response = await self._session.list_prompts()
        return [{"name": p.name, "description": p.description} for p in response.prompts]

    # ------------------------------------------------------------------
    # 手动连接模式（非上下文管理器）
    # ------------------------------------------------------------------

    async def connect_stdio(self, command: str, *args: str, **kwargs) -> None:
        """手动建立 stdio 连接（需配合 close() 使用）。"""
        from mcp.client.stdio import stdio_client, StdioServerParameters

        server_params = StdioServerParameters(command=command, args=list(args), env=kwargs.get("env"))
        self._stdio_ctx = stdio_client(server_params)
        read_stream, write_stream = await self._stdio_ctx.__aenter__()
        self._session = ClientSession(read_stream, write_stream)
        await self._session.__aenter__()  # 启动后台接收循环
        await self._session.initialize()
        await self._refresh_tools()

    async def connect_sse(self, url: str, **kwargs) -> None:
        """手动建立 SSE 连接（需配合 close() 使用）。"""
        from mcp.client.sse import sse_client

        self._sse_ctx = sse_client(url, **kwargs)
        read_stream, write_stream = await self._sse_ctx.__aenter__()
        self._session = ClientSession(read_stream, write_stream)
        await self._session.__aenter__()  # 启动后台接收循环（关键修复）
        await self._session.initialize()
        await self._refresh_tools()

    async def close(self) -> None:
        """关闭连接，清理资源。

        关键设计：只关闭底层传输上下文（_sse_ctx / _stdio_ctx）。
        这会自然关闭 read_stream / write_stream，ClientSession 的后台接收循环
        检测到 stream 关闭后会自行退出，无需（也不应）手动调用 session.__aexit__。
        手动调用 session.__aexit__ 会导致 anyio cancel scope 的 task 绑定冲突。
        """
        self._tools = None
        self._session = None
        if hasattr(self, "_stdio_ctx"):
            try:
                await self._stdio_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            delattr(self, "_stdio_ctx")
        if hasattr(self, "_sse_ctx"):
            try:
                await self._sse_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            delattr(self, "_sse_ctx")
        logger.info("[MCP Client] 连接已关闭")

    # ------------------------------------------------------------------
    # 便捷同步接口（用于非 async 场景）
    # ------------------------------------------------------------------

    @classmethod
    def call_stdio_sync(
        cls,
        command: str,
        *args: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """同步调用 stdio Server 的单个工具（一次性连接，用完即关）。

        适用于脚本、测试等不需要保持长连接的场景。
        """

        async def _inner() -> str:
            async with cls.stdio(command, *args) as client:
                return await client.call_tool(tool_name, arguments)

        return asyncio.run(_inner())

    @classmethod
    def call_sse_sync(
        cls,
        url: str,
        tool_name: str,
        arguments: dict[str, Any],
        headers: dict[str, Any] | None = None,
    ) -> str:
        """同步调用 SSE Server 的单个工具（一次性连接，用完即关）。"""

        async def _inner() -> str:
            async with cls.sse(url, headers=headers) as client:
                return await client.call_tool(tool_name, arguments)

        return asyncio.run(_inner())

    def __repr__(self) -> str:
        tool_count = len(self._tools) if self._tools else 0
        return f"<MCPClient connected={self._session is not None} tools={tool_count}>"
