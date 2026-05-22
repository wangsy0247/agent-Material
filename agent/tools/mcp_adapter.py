"""
MCP 工具适配器 — 基于 langchain-mcp-adapters 官方库

特性：
    - 同时连接多个 MCP Server，支持 SSE 和 stdio 两种传输
    - 使用 langchain_mcp_adapters.tools.convert_mcp_tool_to_langchain_tool
      将 MCP Tool 转换为 LangChain StructuredTool，完整支持 JSON Schema
    - 同名校验：自动加 server_name 前缀区分
    - 单个连接失败不影响其他 Server
    - 支持 callbacks、tool interceptors 等官方扩展机制（预留接口）

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
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import convert_mcp_tool_to_langchain_tool
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client, StdioServerParameters

from utils.logger import logger

# 全局状态
_adapters: dict[str, "_MCPConnection"] = {}
_mcp_tools: list[BaseTool] = []
_tool_server_map: dict[str, str] = {}  # tool_name -> server_name 映射

# 从配置读取全局默认值（失败时使用硬编码默认值）
def _get_mcp_global_config(key: str, default):
    try:
        from utils.config_handler import agent_config
        return agent_config.get("mcp_global", {}).get(key, default)
    except Exception:
        return default


@dataclass
class _MCPConnection:
    """单个 MCP Server 连接（内部类，不对外暴露）。"""

    name: str
    transport: str
    _session: ClientSession | None = field(default=None, repr=False)
    _transport_ctx: Any = field(default=None, repr=False)
    # 运行时健康状态
    healthy: bool = False
    _reconnect_attempts: int = field(default=0, repr=False)
    _last_error: str | None = field(default=None, repr=False)

    # 重连配置（可从 agent_config 覆盖）
    max_reconnect_attempts: int = field(
        default_factory=lambda: _get_mcp_global_config("max_reconnect_attempts", 5)
    )
    initial_backoff_ms: float = field(
        default_factory=lambda: _get_mcp_global_config("initial_backoff_ms", 1000)
    )
    max_backoff_ms: float = field(
        default_factory=lambda: _get_mcp_global_config("max_backoff_ms", 30000)
    )

    async def connect_stdio(
        self,
        command: list[str],
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        """
        建立 stdio 传输连接。

        进程生命周期由 ``mcp.client.stdio`` 内部管理：
        - 启动时通过 ``anyio.open_process`` 创建子进程
        - 关闭时先关闭 stdin，等待 2s 后若未退出则调用 ``_terminate_process_tree``
          （Unix: ``os.killpg`` 进程组终止；Windows: Job Objects）
        - 因此无需在外层手动管理 SIGINT/SIGTERM/SIGKILL 序列
        """
        if not command:
            raise ValueError(f"'{self.name}': stdio 模式需要 command 参数")
        # 若未显式配置 env，则继承当前进程环境变量（确保 API Key 等能传递到子进程）
        effective_env = env if env is not None else os.environ.copy()
        params = StdioServerParameters(
            command=command[0],
            args=command[1:] if len(command) > 1 else [],
            env=effective_env,
        )
        self._transport_ctx = stdio_client(params)
        read, write = await self._transport_ctx.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()
        self.healthy = True
        self._reconnect_attempts = 0
        self._last_error = None

    async def connect_sse(self, url: str, auth: dict | None = None) -> None:
        """建立 SSE 传输连接。"""
        if not url:
            raise ValueError(f"'{self.name}': SSE 模式需要 url 参数")
        headers = {}
        if auth:
            auth_type = auth.get("type", "").lower()
            if auth_type == "bearer":
                token = auth.get("token", "")
                if token.startswith("$") and not token.startswith("${"):
                    token = os.environ.get(token[1:], "")
                elif token.startswith("${") and token.endswith("}"):
                    token = os.environ.get(token[2:-1], "")
                if token:
                    headers["Authorization"] = f"Bearer {token}"
        self._transport_ctx = sse_client(url, headers=headers or None)
        read, write = await self._transport_ctx.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()
        self.healthy = True
        self._reconnect_attempts = 0
        self._last_error = None

    async def _check_health(self) -> bool:
        """发送轻量调用检测连接是否存活。"""
        if self._session is None:
            self.healthy = False
            return False
        try:
            await asyncio.wait_for(self._session.list_tools(), timeout=5.0)
            if not self.healthy:
                logger.info(f"[MCPAdapter] '{self.name}' 连接已恢复")
            self.healthy = True
            self._reconnect_attempts = 0
            self._last_error = None
            return True
        except Exception as e:
            if self.healthy:
                transport_hint = "stdio" if self.transport == "stdio" else "SSE/HTTP"
                logger.warning(
                    f"[MCPAdapter] '{self.name}' ({transport_hint}) 健康检查失败: {e}"
                )
                if self.transport == "stdio":
                    logger.warning(
                        f"[MCPAdapter] '{self.name}' stdio 子进程可能已退出或僵死，"
                        f"将尝试重连"
                    )
            self.healthy = False
            self._last_error = str(e)
            return False

    async def reconnect(self, config: dict) -> bool:
        """尝试重新建立连接（带指数退避）。"""
        if self._reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(
                f"[MCPAdapter] '{self.name}' 重连次数已达上限 "
                f"({self.max_reconnect_attempts})"
            )
            return False

        backoff = min(
            self.initial_backoff_ms * (2 ** self._reconnect_attempts),
            self.max_backoff_ms,
        )
        self._reconnect_attempts += 1
        logger.info(
            f"[MCPAdapter] '{self.name}' 将在 {backoff / 1000:.1f}s 后第 "
            f"{self._reconnect_attempts} 次重连..."
        )
        await asyncio.sleep(backoff / 1000)

        # 先清理旧连接
        await self.close()

        try:
            if self.transport == "stdio":
                await self.connect_stdio(
                    command=config.get("command", []),
                    env=config.get("env"),
                    cwd=config.get("cwd"),
                )
            else:
                await self.connect_sse(
                    url=config.get("url", ""),
                    auth=config.get("auth"),
                )
            self.healthy = True
            self._reconnect_attempts = 0
            self._last_error = None
            # 重连成功后重新发现工具
            await self._refresh_tools()
            logger.info(f"[MCPAdapter] '{self.name}' 重连成功")
            return True
        except Exception as e:
            logger.error(f"[MCPAdapter] '{self.name}' 重连失败: {e}")
            self._last_error = str(e)
            return False

    async def _refresh_tools(self) -> None:
        """重新发现该服务器的工具并更新全局工具列表。"""
        global _mcp_tools, _tool_server_map
        if self._session is None:
            return
        try:
            response = await self._session.list_tools()
            # 移除该服务器旧的工具及其映射
            old_tools = [t for t in _mcp_tools if _tool_server_map.get(t.name) == self.name]
            for t in old_tools:
                _tool_server_map.pop(t.name, None)
            _mcp_tools = [t for t in _mcp_tools if t not in old_tools]

            for tool in response.tools:
                lc_tool = convert_mcp_tool_to_langchain_tool(
                    session=self._session,
                    tool=tool,
                    server_name=self.name,
                    tool_name_prefix=False,
                )
                lc_tool.name = f"{self.name}__{tool.name}"
                _tool_server_map[lc_tool.name] = self.name
                _mcp_tools.append(lc_tool)
            logger.info(
                f"[MCPAdapter] '{self.name}' 刷新工具完成: "
                f"{len(response.tools)} 个工具"
            )
        except Exception as e:
            logger.error(f"[MCPAdapter] '{self.name}' 刷新工具失败: {e}")

    async def close(self) -> None:
        """优雅关闭连接。

        stdio 模式下，``mcp.client.stdio`` 的 context manager 会在 ``__aexit__``
        中自动执行关闭 stdin → 等待 2s → ``_terminate_process_tree`` 的清理序列，
        因此外层只需确保 session 和 transport 的 ``__aexit__`` 被正确调用即可。
        """
        if self._session is not None:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception:
                pass
            self._session = None
        if self._transport_ctx is not None:
            try:
                await self._transport_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            self._transport_ctx = None
        self.healthy = False


# ================================================================
# 模块级便捷函数
# ================================================================


# 常量：工具描述截断长度
MAX_DESCRIPTION_LENGTH = 2048


async def init_mcp_tools(servers: list[dict]) -> list[BaseTool]:
    """连接多个 MCP Server（SSE / stdio），汇聚所有工具。

    使用 langchain-mcp-adapters 官方库的 ``convert_mcp_tool_to_langchain_tool``
    进行工具转换，自动处理 JSON Schema、重名前缀、返回格式等细节。
    单个 Server 连接失败不会中断整体初始化。

    Args:
        servers: [
            {"transport": "sse",  "url": "http://...", "name": "..."},
            {"transport": "stdio", "command": ["python", "-m", "..."], "name": "..."},
        ]

    Returns:
        所有 Server 的 LangChain Tool 汇总列表
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

    total = 0
    for cfg in servers:
        # 支持 enabled 开关，默认启用
        if not cfg.get("enabled", True):
            logger.info(f"[MCPAdapter] '{cfg.get('name', '?')}' 已禁用，跳过")
            continue

        raw_name = cfg.get("name", "") or _default_name(cfg)
        name = _normalize_server_name(raw_name)
        transport = cfg.get("transport", "sse")
        conn = _MCPConnection(name=name, transport=transport)

        try:
            if transport == "stdio":
                await conn.connect_stdio(
                    command=cfg.get("command", []),
                    env=cfg.get("env"),
                    cwd=cfg.get("cwd"),
                )
            else:
                await conn.connect_sse(
                    url=cfg.get("url", ""),
                    auth=cfg.get("auth"),
                )

            response = await conn._session.list_tools()
            for tool in response.tools:
                # 截断过长的描述，防止 OpenAPI 生成的大描述撑爆上下文
                if tool.description and len(tool.description) > MAX_DESCRIPTION_LENGTH:
                    tool.description = tool.description[:MAX_DESCRIPTION_LENGTH] + "... [截断]"

                lc_tool = convert_mcp_tool_to_langchain_tool(
                    session=conn._session,
                    tool=tool,
                    server_name=conn.name,
                    tool_name_prefix=False,  # 我们手动设置名称，使用双下划线分隔
                )
                # 使用双下划线作为 server 与 tool 的分隔符，避免与工具名内部的单下划线冲突
                # 例如：material_project__mp_download（原始工具名 mp_download 内部有下划线）
                lc_tool.name = f"{conn.name}__{tool.name}"
                # 使用全局字典维护 tool_name -> server_name 映射
                # StructuredTool 是 Pydantic BaseModel，不允许动态添加字段
                _tool_server_map[lc_tool.name] = conn.name
                _mcp_tools.append(lc_tool)
                total += 1

            _adapters[conn.name] = conn
            logger.info(
                f"[MCPAdapter] '{conn.name}' 发现 {len(response.tools)} 个工具: "
                f"{[t.name for t in response.tools]}"
            )
        except Exception as e:
            logger.error(f"[MCPAdapter] 连接 '{conn.name}' 失败: {e}")
            conn._last_error = str(e)
            await conn.close()

    logger.info(f"[MCPAdapter] 汇总: {len(_adapters)} 个 Server, {total} 个工具")
    return _mcp_tools


async def close_mcp() -> None:
    """关闭所有 MCP Server 连接。"""
    global _adapters, _mcp_tools, _tool_server_map
    for adapter in _adapters.values():
        await adapter.close()
    _adapters.clear()
    _mcp_tools.clear()
    _tool_server_map.clear()


def get_mcp_tools() -> list[BaseTool]:
    """获取已缓存的所有 MCP 工具（同步接口）。"""
    if not _mcp_tools:
        raise RuntimeError("MCP 工具未初始化，请在 lifespan 中调用 init_mcp_tools()")
    return _mcp_tools


def _normalize_server_name(name: str) -> str:
    """将 Server 名称规范化为 API 安全格式（字母、数字、下划线）。

    借鉴 Claude-Code 的 normalizeNameForMCP：将连字符、空格等替换为下划线，
    避免与工具名内部的单下划线分隔产生歧义。
    """
    # 先将常见分隔符统一替换为下划线
    normalized = re.sub(r"[\-\s.]", "_", name)
    # 去除连续下划线
    normalized = re.sub(r"_+", "_", normalized)
    # 去除首尾下划线
    normalized = normalized.strip("_")
    return normalized or "mcp"


def _default_name(cfg: dict) -> str:
    """根据配置生成默认 Server 名称。"""
    transport = cfg.get("transport", "sse")
    if transport == "stdio":
        cmd = cfg.get("command", ["stdio"])
        return _normalize_server_name(f"stdio_{cmd[0]}")
    url = cfg.get("url", "")
    match = re.search(r"([\d.]+):(\d+)", url)
    if match:
        return f"mcp_{match.group(2)}"
    return "mcp"
