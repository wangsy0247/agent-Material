"""
MCP 工具管理器 — 将外部 MCP Server 的工具集成到 Agent 中

设计思路：
    1. 初始化阶段（lifespan startup）：连接各 MCP Server，发现工具元数据
    2. 运行阶段（Agent 推理）：将 MCP 工具转为 LangChain BaseTool，Agent 通过标准接口调用
    3. 每次工具调用时创建临时 MCP 连接（简单可靠，无需管理长连接生命周期）

使用：
    # 在 api_server.py lifespan 中初始化
    await init_mcp_tools()

    # 在 Agent 中获取工具
    from agent.tools.agent_tools import get_all_tools
    tools = get_all_tools()  # 原生工具 + MCP 工具
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import yaml
from langchain_core.tools import BaseTool, StructuredTool, Tool
from pydantic import BaseModel, Field, create_model

from mcp_servers.client import MCPClient
from utils.logger import logger

# 全局 MCP 工具管理器实例（单例）
_mcp_tool_manager: MCPToolManager | None = None


class MCPToolManager:
    """MCP 工具管理器。

    负责从 config/mcp_config.yaml 读取配置，连接启用的 MCP Server，
    将 MCP Tool 转换为 LangChain BaseTool，供 Agent 使用。

    采用"发现时连接、调用时重连"的策略：
    - discover()：建立临时连接获取工具元数据（名称、描述、参数Schema）
    - get_tools()：基于元数据创建 LangChain Tool，每次调用时重新建立 MCP 连接
    """

    def __init__(self):
        # server_name -> list[dict(name, description, parameters)]
        self._server_tools: dict[str, list[dict]] = {}
        # tool_name -> server_name
        self._tool_server_map: dict[str, str] = {}
        # server_name -> 连接参数
        self._connection_params: dict[str, dict] = {}

    async def discover(self) -> None:
        """发现所有启用的 MCP Server 的工具列表。

        读取 config/mcp_config.yaml，连接每个 enabled=true 的 Server，
        获取工具元数据后断开连接。
        """
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "config", "mcp_config.yaml"
        )
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        for server_cfg in cfg.get("servers", []):
            if not server_cfg.get("enabled", False):
                continue

            name = server_cfg["name"]
            transport = server_cfg.get("transport", "stdio")

            try:
                if transport == "stdio":
                    cmd = server_cfg["command"]
                    async with MCPClient.stdio(cmd[0], *cmd[1:]) as client:
                        self._register_tools(name, client, transport, command=cmd[0], args=cmd[1:])
                elif transport == "sse":
                    url = server_cfg["url"]
                    async with MCPClient.sse(url) as client:
                        self._register_tools(name, client, transport, url=url)

                logger.info(
                    f"[MCPToolManager] 发现 Server '{name}' | "
                    f"工具: {[t['name'] for t in self._server_tools.get(name, [])]}"
                )

            except Exception as e:
                logger.error(f"[MCPToolManager] 发现 Server '{name}' 失败: {e}")

    def _register_tools(
        self,
        server_name: str,
        client: MCPClient,
        transport: str,
        **conn_kwargs: Any,
    ) -> None:
        """将 Client 发现的工具注册到内部映射表。"""
        tool_metas = []
        for t in client.tools:
            tool_metas.append(
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
            )
            self._tool_server_map[t.name] = server_name

        self._server_tools[server_name] = tool_metas
        self._connection_params[server_name] = {"type": transport, **conn_kwargs}

    def get_tools(self) -> list[BaseTool]:
        """将所有发现的 MCP 工具转换为 LangChain BaseTool 列表。

        Returns:
            可直接传入 create_agent(tools=...) 的 BaseTool 列表
        """
        tools: list[BaseTool] = []
        for server_name, tool_metas in self._server_tools.items():
            conn_params = self._connection_params[server_name]
            for meta in tool_metas:
                lc_tool = self._create_langchain_tool(meta, conn_params)
                tools.append(lc_tool)
        return tools

    def _create_langchain_tool(
        self,
        meta: dict[str, Any],
        conn_params: dict[str, Any],
    ) -> BaseTool:
        """为单个 MCP Tool 创建 LangChain Tool。

        关键设计：_arun 中每次调用都新建 MCP 连接。
        这虽然有一定开销，但避免了长连接生命周期管理的复杂性，
        且与 stdio 子进程模式的设计理念一致。
        """
        name = meta["name"]
        description = meta["description"]
        parameters = meta["parameters"]

        # 从 JSON Schema 构建 Pydantic 输入模型
        args_schema: type[BaseModel] | None = None
        try:
            args_schema = _json_schema_to_pydantic(name, parameters)
        except Exception as e:
            logger.warning(f"[MCPToolManager] 工具 '{name}' 参数模型构建失败: {e}")

        async def _arun(**kwargs: Any) -> str:
            """异步调用：创建临时 MCP 连接并调用工具。"""
            if conn_params["type"] == "stdio":
                async with MCPClient.stdio(
                    conn_params["command"], *conn_params["args"]
                ) as client:
                    return await client.call_tool(name, kwargs)
            else:  # sse
                async with MCPClient.sse(conn_params["url"]) as client:
                    return await client.call_tool(name, kwargs)

        def _run(**kwargs: Any) -> str:
            """同步调用包装。"""
            return asyncio.run(_arun(**kwargs))

        if args_schema is not None:
            return StructuredTool.from_function(
                func=_run,
                coroutine=_arun,
                name=name,
                description=description,
                args_schema=args_schema,
                return_direct=False,
            )
        else:
            return Tool(
                name=name,
                description=description,
                func=lambda: _run(),
                coroutine=lambda: _arun(),
            )

    def list_tools(self) -> list[dict[str, Any]]:
        """返回发现的工具元数据列表（用于调试和展示）。"""
        result = []
        for server_name, tool_metas in self._server_tools.items():
            for meta in tool_metas:
                result.append({**meta, "server": server_name})
        return result


# ------------------------------------------------------------------
# 模块级便捷函数
# ------------------------------------------------------------------

async def init_mcp_tools() -> list[BaseTool]:
    """初始化 MCP 工具（在应用 lifespan startup 中调用）。

    用法：
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await init_mcp_tools()
            yield
    """
    global _mcp_tool_manager
    _mcp_tool_manager = MCPToolManager()
    await _mcp_tool_manager.discover()
    tools = _mcp_tool_manager.get_tools()
    logger.info(f"[MCPToolManager] 共加载 {len(tools)} 个 MCP 工具")
    return tools


def get_mcp_tools() -> list[BaseTool]:
    """获取已初始化的 MCP 工具（LangChain BaseTool 格式）。

    必须在 init_mcp_tools() 之后调用。
    """
    if _mcp_tool_manager is None:
        raise RuntimeError(
            "MCP 工具未初始化。请在应用 lifespan startup 中调用 init_mcp_tools()"
        )
    return _mcp_tool_manager.get_tools()


def get_mcp_tool_manager() -> MCPToolManager:
    """获取 MCP 工具管理器实例。"""
    if _mcp_tool_manager is None:
        raise RuntimeError("MCP 工具未初始化")
    return _mcp_tool_manager


# ------------------------------------------------------------------
# 内部辅助
# ------------------------------------------------------------------


def _json_schema_to_pydantic(
    name: str,
    schema: dict[str, Any],
) -> type[BaseModel]:
    """将 MCP Tool 的 JSON Schema 转换为 Pydantic BaseModel。

    支持 string / integer / number / boolean / array / object 类型，
    以及 required 字段标记和 description 映射。
    """
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    fields = {}
    for field_name, field_schema in properties.items():
        json_type = field_schema.get("type", "string")
        description = field_schema.get("description", "")
        default = field_schema.get("default", ...)

        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        py_type = type_mapping.get(json_type, str)

        if field_name not in required and default is ...:
            default = None
            py_type = py_type | None

        fields[field_name] = (py_type, Field(default=default, description=description))

    return create_model(f"{name}_Input", **fields)
