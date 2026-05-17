"""
MCP (Model Context Protocol) 协议集成模块

提供：
- BaseMCPServer: MCP Server 基类，支持 stdio / sse 两种传输模式
- MCPClient: MCP Client，连接外部 MCP Server
- MCPToolAdapter / LangChainToolAdapter: 双向桥接适配器

项目 MCP 目录已从 `mcp` 重命名为 `mcp_servers`，避免与官方 `mcp` Python SDK 包名冲突。
"""

from mcp_servers.server import BaseMCPServer
from mcp_servers.client import MCPClient
from mcp_servers.adapters.langchain_adapter import (
    MCPToolAdapter,
    LangChainToolAdapter,
    mcp_tools_to_langchain,
    mcp_tools_to_langchain_sync,
)

__all__ = [
    "BaseMCPServer",
    "MCPClient",
    "MCPToolAdapter",
    "LangChainToolAdapter",
    "mcp_tools_to_langchain",
    "mcp_tools_to_langchain_sync",
]
