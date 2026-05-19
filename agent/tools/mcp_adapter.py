"""
MCP 工具适配器 — 连接多个 MCP Server (SSE / stdio)，自动发现并转换为 LangChain Tool

特性：
    - 同时连接多个 MCP Server，支持 SSE 和 stdio 两种传输
    - 同名校验，自动加前缀区分
    - 单个连接失败不影响其他 Server
    - JSON Schema → Pydantic → LangChain StructuredTool

配置示例 (config/agent_config.yaml)：
    mcp_servers:
      # SSE 模式 — 连接独立运行的 MCP 服务
      - transport: "sse"
        url: "http://127.0.0.1:8002/sse"
        name: "material-project"

      # stdio 模式 — 启动子进程通信
      - transport: "stdio"
        command: ["python", "-m", "mp_material_server"]
        name: "material-project"
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client, StdioServerParameters
from pydantic import BaseModel, Field, create_model

from utils.logger import logger

# 全局状态
_adapters: dict[str, MCPAdapter] = {}
_mcp_tools: list[BaseTool] = []


class MCPAdapter:
    """单个 MCP Server 客户端，支持 SSE 和 stdio 两种传输。

    负责：连接 → 发现工具 → JSON Schema 转 Pydantic → 生成 LangChain Tool。
    每个 Tool 的闭包持有当前 session 引用，调用时复用长连接。
    """

    def __init__(self, cfg: dict):
        self.name: str = cfg.get("name", "")
        self.transport: str = cfg.get("transport", "sse")

        # SSE 参数
        self.url: str = cfg.get("url", "")

        # stdio 参数
        self.command: list[str] = cfg.get("command", [])
        self.env: dict | None = cfg.get("env")
        self.cwd: str | None = cfg.get("cwd")

        # 运行时
        self._session: ClientSession | None = None
        self._transport_ctx = None
        # stdio 管道串行锁 — 同一 Server 的所有工具共享
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """根据 transport 类型建立连接。"""
        if self.transport == "stdio":
            await self._connect_stdio()
        else:
            await self._connect_sse()

        assert self._session is not None
        await self._session.initialize()
        logger.info(f"[MCPAdapter] 已连接 '{self.name}' (transport={self.transport})")

    async def _connect_sse(self) -> None:
        if not self.url:
            raise ValueError(f"'{self.name}': SSE 模式需要 url 参数")
        # 自动生成名称
        if not self.name:
            self.name = _name_from_url(self.url)

        self._transport_ctx = sse_client(self.url)
        read, write = await self._transport_ctx.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.__aenter__()

    async def _connect_stdio(self) -> None:
        if not self.command:
            raise ValueError(f"'{self.name}': stdio 模式需要 command 参数")
        # 自动生成名称
        if not self.name:
            self.name = f"stdio:{self.command[0]}"

        # 若未显式配置 env，则继承当前进程环境变量（确保 API Key 等能传递到子进程）
        env = self.env if self.env is not None else os.environ.copy()
        params = StdioServerParameters(
            command=self.command[0],
            args=self.command[1:] if len(self.command) > 1 else [],
            env=env,
        )
        self._transport_ctx = stdio_client(params)
        read, write = await self._transport_ctx.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.__aenter__()

    async def close(self) -> None:
        if self._session is not None:
            await self._session.__aexit__(None, None, None)
            self._session = None
        if self._transport_ctx is not None:
            await self._transport_ctx.__aexit__(None, None, None)
            self._transport_ctx = None
        logger.info(f"[MCPAdapter] 已断开 '{self.name}'")

    # ------------------------------------------------------------------
    # 工具发现与转换
    # ------------------------------------------------------------------

    async def discover_tools(
        self, existing_names: set[str] | None = None
    ) -> list[BaseTool]:
        """发现工具并转换为 LangChain Tool。

        Args:
            existing_names: 已注册的工具名集合，用于重名检测
        """
        if self._session is None:
            raise RuntimeError(f"'{self.name}' 未连接")

        existing_names = existing_names or set()
        response = await self._session.list_tools()
        tools: list[BaseTool] = []

        for tool in response.tools:
            raw_name = tool.name
            final_name = (
                f"{self.name}_{raw_name}" if raw_name in existing_names else raw_name
            )
            existing_names.add(final_name)

            lc_tool = self._to_langchain_tool(
                mcp_tool_name=raw_name,
                langchain_tool_name=final_name,
                description=tool.description or "",
                schema=tool.inputSchema,
            )
            tools.append(lc_tool)

        logger.info(
            f"[MCPAdapter] '{self.name}' 发现 {len(tools)} 个工具: "
            f"{[t.name for t in tools]}"
        )
        return tools

    def _to_langchain_tool(
        self,
        mcp_tool_name: str,
        langchain_tool_name: str,
        description: str,
        schema: dict,
    ) -> BaseTool:
        """JSON Schema → Pydantic → LangChain StructuredTool。"""
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        type_map = {"string": str, "integer": int, "number": float, "boolean": bool}
        fields: dict[str, tuple] = {}
        for field_name, field_schema in properties.items():
            py_type = type_map.get(field_schema.get("type", "string"), str)
            default = ... if field_name in required else None
            if default is None:
                py_type = py_type | None
            fields[field_name] = (
                py_type,
                Field(default=default, description=field_schema.get("description", "")),
            )

        args_schema = create_model(f"{langchain_tool_name}_Input", **fields)
        session = self._session
        adapter_lock = self._lock  # 同一 Server 的所有工具共享此锁

        async def _arun(**kwargs: Any) -> str:
            if self.transport == "stdio":
                async with adapter_lock:
                    result = await session.call_tool(mcp_tool_name, kwargs)
            else:
                result = await session.call_tool(mcp_tool_name, kwargs)
            texts = []
            for content in result.content:
                if hasattr(content, "text"):
                    texts.append(content.text)
            return "\n".join(texts)

        def _run(**kwargs: Any) -> str:
            return asyncio.run(_arun(**kwargs))

        return StructuredTool.from_function(
            func=_run,
            coroutine=_arun,
            name=langchain_tool_name,
            description=description,
            args_schema=args_schema,
        )


# ================================================================
# 模块级便捷函数
# ================================================================


async def init_mcp_tools(servers: list[dict]) -> list[BaseTool]:
    """连接多个 MCP Server（SSE / stdio），汇聚所有工具。

    Args:
        servers: [
            {"transport": "sse",  "url": "http://...", "name": "..."},
            {"transport": "stdio", "command": ["python", "-m", "..."], "name": "..."},
        ]

    Returns:
        所有 Server 的 LangChain Tool 汇总列表

    单个 Server 连接失败不会中断整体初始化。
    """
    global _adapters, _mcp_tools
    # 先关闭旧连接
    for adapter in _adapters.values():
        try:
            await adapter.close()
        except Exception:
            pass
    _adapters.clear()
    _mcp_tools.clear()

    existing_names: set[str] = set()
    total = 0

    for cfg in servers:
        transport = cfg.get("transport", "sse")
        adapter = MCPAdapter(cfg)
        try:
            await adapter.connect()
            tools = await adapter.discover_tools(existing_names)
            _adapters[adapter.name] = adapter
            _mcp_tools.extend(tools)
            total += len(tools)
        except Exception as e:
            logger.error(f"[MCPAdapter] 连接 '{adapter.name}' 失败: {e}")
            await adapter.close()

    logger.info(f"[MCPAdapter] 汇总: {len(_adapters)} 个 Server, {total} 个工具")
    return _mcp_tools


async def close_mcp() -> None:
    """关闭所有 MCP Server 连接。"""
    global _adapters, _mcp_tools
    for adapter in _adapters.values():
        await adapter.close()
    _adapters.clear()
    _mcp_tools.clear()


def get_mcp_tools() -> list[BaseTool]:
    """获取已缓存的所有 MCP 工具（同步接口）。"""
    if not _mcp_tools:
        raise RuntimeError("MCP 工具未初始化，请在 lifespan 中调用 init_mcp_tools()")
    return _mcp_tools


def _name_from_url(url: str) -> str:
    """从 URL 提取默认名称，如 http://127.0.0.1:8002/sse → mcp_8002"""
    import re

    match = re.search(r"([\d.]+):(\d+)", url)
    if match:
        return f"mcp_{match.group(2)}"
    return "mcp"
