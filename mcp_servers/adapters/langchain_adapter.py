"""
MCP ↔ LangChain 双向桥接适配器

提供两个核心能力：
    1. MCPToolAdapter: 将 MCP Server 的工具转换为 LangChain BaseTool
       → Agent 可以像调用普通 LangChain Tool 一样调用 MCP Tool

    2. LangChainToolAdapter: 将 LangChain Tool 注册到 MCP Server
       → 现有 LangChain Tool 可以通过 MCP 协议被外部客户端调用

用法示例：
    # MCP → LangChain（消费外部 MCP Server 的工具）
    async with MCPClient.stdio("python", "-m", "mcp_servers.tools.rag_mcp_server") as client:
        adapter = MCPToolAdapter(client)
        lc_tools = await adapter.to_langchain_tools()
        # lc_tools 是 List[BaseTool]，可直接传给 LangChain Agent

    # LangChain → MCP（将现有工具暴露为 MCP Server）
    server = BaseMCPServer(name="lc-tools")
    adapter = LangChainToolAdapter(server)
    adapter.register_tool(rag_summary)  # 注册现有 LangChain Tool
    server.run_stdio()
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Callable

from langchain_core.tools import BaseTool, StructuredTool, Tool
from pydantic import BaseModel, Field, create_model

from mcp_servers.client import MCPClient, ToolInfo
from mcp_servers.server import BaseMCPServer

logger = logging.getLogger(__name__)


# ===================================================================
# MCP → LangChain
# ===================================================================


class MCPToolAdapter:
    """将 MCP Server 的工具转换为 LangChain BaseTool。

    通过 MCP Client 发现远端工具列表，为每个工具生成对应的
    LangChain StructuredTool，使得 LangChain Agent 可以无缝调用。

    Args:
        client: 已连接的 MCPClient 实例
    """

    def __init__(self, client: MCPClient):
        self._client = client

    @staticmethod
    def _json_schema_to_pydantic(
        name: str,
        schema: dict[str, Any],
    ) -> type[BaseModel]:
        """将 MCP Tool 的 JSON Schema 转换为 Pydantic BaseModel。

        支持：
            - type: string / integer / number / boolean / array / object
            - required 字段标记
            - description 映射为 Field(description=...)
        """
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        fields = {}
        for field_name, field_schema in properties.items():
            json_type = field_schema.get("type", "string")
            description = field_schema.get("description", "")
            default = field_schema.get("default", ...)

            # JSON Schema type → Python type
            type_mapping = {
                "string": str,
                "integer": int,
                "number": float,
                "boolean": bool,
                "array": list,
                "object": dict,
            }
            py_type = type_mapping.get(json_type, str)

            # 非 required 字段且有默认值
            if field_name not in required and default is ...:
                default = None
                py_type = py_type | None

            fields[field_name] = (
                py_type,
                Field(default=default, description=description),
            )

        return create_model(f"{name}_Input", **fields)

    async def to_langchain_tools(self) -> list[BaseTool]:
        """将 MCP Server 的所有工具转换为 LangChain Tool 列表。

        Returns:
            List[BaseTool]，可直接传给 LangChain Agent 的 tools 参数
        """
        tools: list[BaseTool] = []

        for tool_info in self._client.tools:
            name = tool_info.name
            description = tool_info.description
            parameters = tool_info.parameters

            # 构建 Pydantic 输入模型
            try:
                args_schema = self._json_schema_to_pydantic(name, parameters)
            except Exception as e:
                logger.warning(f"[MCPToolAdapter] 工具 '{name}' 的参数模型构建失败: {e}，退化为字符串输入")
                args_schema = None

            # 创建 LangChain Tool
            lc_tool = self._create_tool(name, description, args_schema)
            tools.append(lc_tool)
            logger.info(f"[MCPToolAdapter] 转换工具: {name}")

        return tools

    def _create_tool(
        self,
        name: str,
        description: str,
        args_schema: type[BaseModel] | None,
    ) -> BaseTool:
        """为单个 MCP Tool 创建对应的 LangChain Tool。"""

        async def _arun(**kwargs: Any) -> str:
            return await self._client.call_tool(name, kwargs)

        def _run(**kwargs: Any) -> str:
            # 同步包装：在已有事件循环中直接 await，否则新建循环
            try:
                loop = asyncio.get_running_loop()
                # 已有事件循环，创建 task 并等待（可能阻塞，仅用于演示）
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, _arun(**kwargs))
                    return future.result()
            except RuntimeError:
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
                func=lambda query: _run(query=query),
                coroutine=lambda query: _arun(query=query),
            )

    async def to_langchain_tool(self, name: str) -> BaseTool | None:
        """将指定名称的 MCP Tool 转换为单个 LangChain Tool。"""
        for tool_info in self._client.tools:
            if tool_info.name == name:
                args_schema = None
                try:
                    args_schema = self._json_schema_to_pydantic(name, tool_info.parameters)
                except Exception as e:
                    logger.warning(f"[MCPToolAdapter] 工具 '{name}' 参数模型构建失败: {e}")
                return self._create_tool(name, tool_info.description, args_schema)
        return None


# ===================================================================
# LangChain → MCP
# ===================================================================


class LangChainToolAdapter:
    """将 LangChain Tool 注册到 MCP Server。

    使得现有的 LangChain Tool（如 @tool 装饰的函数）可以通过 MCP 协议
    被外部客户端调用，实现平滑迁移。

    Args:
        server: BaseMCPServer 实例
    """

    def __init__(self, server: BaseMCPServer):
        self._server = server

    def register_tool(self, lc_tool: BaseTool | Callable) -> None:
        """将单个 LangChain Tool 注册到 MCP Server。

        Args:
            lc_tool: LangChain BaseTool 实例，或被 @tool 装饰的函数
        """
        # 如果被 @tool 装饰的原始函数，先转为 BaseTool
        if not isinstance(lc_tool, BaseTool):
            from langchain_core.tools import tool as lc_tool_decorator

            if callable(lc_tool):
                lc_tool = StructuredTool.from_function(
                    func=lc_tool,
                    name=lc_tool.__name__,
                    description=lc_tool.__doc__ or lc_tool.__name__,
                )
            else:
                raise TypeError(f"不支持的 Tool 类型: {type(lc_tool)}")

        name = lc_tool.name
        description = lc_tool.description or name

        # 获取输入参数的 JSON Schema（从 Pydantic model）
        try:
            input_schema = lc_tool.args_schema.model_json_schema() if lc_tool.args_schema else {"type": "object", "properties": {}}
        except Exception:
            input_schema = {"type": "object", "properties": {}}

        # 构造 MCP Tool 处理函数
        # 注意：MCP Tool 接收的是关键字参数，需要适配 LangChain Tool 的调用方式
        @self._server.tool(name=name, description=description)
        async def _mcp_tool(**kwargs: Any) -> str:
            """MCP Tool 包装器，内部调用 LangChain Tool。"""
            try:
                # 优先使用异步接口
                if inspect.iscoroutinefunction(lc_tool.ainvoke) or hasattr(lc_tool, "coroutine"):
                    result = await lc_tool.ainvoke(kwargs)
                else:
                    result = lc_tool.invoke(kwargs)

                # 统一输出为字符串
                if isinstance(result, str):
                    return result
                elif hasattr(result, "content"):
                    return str(result.content)
                else:
                    return str(result)
            except Exception as e:
                logger.error(f"[LangChainToolAdapter] 工具 '{name}' 执行失败: {e}")
                raise

        logger.info(f"[LangChainToolAdapter] 注册工具到 MCP Server: {name}")

    def register_tools(self, tools: list[BaseTool | Callable]) -> None:
        """批量注册 LangChain Tool 列表。"""
        for tool in tools:
            self.register_tool(tool)


# ===================================================================
# 便捷函数
# ===================================================================


async def mcp_tools_to_langchain(
    command: str,
    *args: str,
) -> list[BaseTool]:
    """一键将 stdio MCP Server 的所有工具转为 LangChain Tools。

    用法：
        tools = await mcp_tools_to_langchain("python", "-m", "mcp_servers.tools.rag_mcp_server")
        agent = create_agent(model, tools=tools, ...)
    """
    async with MCPClient.stdio(command, *args) as client:
        adapter = MCPToolAdapter(client)
        return await adapter.to_langchain_tools()


def mcp_tools_to_langchain_sync(
    command: str,
    *args: str,
) -> list[BaseTool]:
    """同步版本：一键将 stdio MCP Server 的所有工具转为 LangChain Tools。"""
    return asyncio.run(mcp_tools_to_langchain(command, *args))
