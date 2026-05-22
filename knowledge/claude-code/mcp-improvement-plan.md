# MCP 子系统改进计划 —— 借鉴 Claude-Code 实现

> 目标：基于 Claude-Code 的生产级 MCP 实现，识别我们项目（agent-material）中可落地的具体改进点  
> 状态：待实施  
> 文件位置：`knowledge/mcp-improvement-plan.md`

---

## 当前现状速览

我们的 MCP 实现正处于**架构简化后的早期可用阶段**：
- 使用 `langchain-mcp-adapters` 官方库做工具转换
- 支持 SSE 和 stdio 两种传输
- 仅在 `api_server.py` lifespan 中做一次初始化，无运行时监控
- 配置已合并到 `agent_config.yaml`，但丢失了 per-server 控制项

**核心差距**：缺少连接韧性、安全边界细化、运行时可观测性。

---

## 改进项清单（按优先级排序）

---

### 🔴 P0 — 修复敏感工具检测漏洞（MCP 工具名前缀绕过 HITL）

**问题**：`agent/graph/agent_graph.py` 中 `_SENSITIVE_TOOLS = {"mp_download"}`，但 MCP 模式下工具名被前缀化为 `material-project_mp_download`，导致 HITL **完全失效**。

**Claude-Code 做法**：使用 `getToolNameForPermissionCheck()`，以 `mcp__server__tool` 全名进行权限匹配，同时维护一个独立的 display-name → 敏感规则映射。

**我们的修复方案**：

```python
# agent/graph/agent_graph.py

def _get_sensitive_tools() -> set[str]:
    """读取敏感工具列表（支持运行时热更新）。"""
    try:
        from utils.config_handler import agent_config
        keywords = agent_config.get("sensitive_keywords", [])
        if keywords:
            return set(keywords)
    except Exception:
        pass
    return _SENSITIVE_TOOLS


def _is_sensitive_tool(tool_name: str) -> bool:
    """
    判断工具是否为敏感操作。
    支持两种匹配模式：
    1. 精确匹配：tool_name 在敏感列表中（兼容无前缀的本地工具）
    2. 后缀匹配：去掉 server 前缀后的工具名在敏感列表中（兼容 MCP 前缀化工具）
    """
    sensitive = _get_sensitive_tools()
    if tool_name in sensitive:
        return True
    # MCP 工具名格式: "<server_name>_<tool_name>"
    # 取最后一个下划线后的部分作为原始工具名进行匹配
    # 注意：工具名本身可能包含下划线，所以用 rsplit(..., 1) 只分割一次
    raw_name = tool_name.rsplit("_", 1)[-1] if "_" in tool_name else tool_name
    return raw_name in sensitive
```

然后在 `human_approval_node` 中：

```python
# 替换原来的 tc["name"] in sensitive
pending = [tc for tc in tool_calls if _is_sensitive_tool(tc["name"])]
```

**同时需要在 `agent_config.yaml` 中恢复敏感关键词配置**：

```yaml
sensitive_keywords:
  - "mp_download"
  - "filesystem_write"   # 未来可能接入的 filesystem MCP
  - "filesystem_delete"
```

---

### 🔴 P0 — SSE 连接保活与自动重连

**问题**：当前 `mcp_adapter.py` 在 lifespan 中初始化一次后永不检查连接健康。SSE 连接因网络抖动、MCP Server 重启而断开后，Agent 调用工具会直接报错，且**不会自动恢复**。

**Claude-Code 做法**：
- SSE 传输内置 `onclose` 回调，触发后清除 memo cache，下次调用时透明重连
- 远程连接使用指数退避重连：`INITIAL_BACKOFF_MS=1000`, `MAX_BACKOFF_MS=30000`, `MAX_RECONNECT_ATTEMPTS=5`
- 捕获 `Session not found` (-32001) 错误后自动清除缓存并重试一次

**我们的修复方案**（在 `mcp_adapter.py` 中增加连接健康层）：

```python
# agent/tools/mcp_adapter.py

import asyncio
from dataclasses import dataclass

@dataclass
class _MCPConnection:
    name: str
    transport: str
    _session: ClientSession | None = None
    _transport_ctx: Any = None
    _healthy: bool = False           # 新增：连接健康状态
    _reconnect_attempts: int = 0     # 新增：当前重连次数
    _last_error: str | None = None   # 新增：上次错误信息

    # 重连配置（可从 agent_config 读取）
    MAX_RECONNECT_ATTEMPTS: int = 5
    INITIAL_BACKOFF_MS: float = 1000
    MAX_BACKOFF_MS: float = 30000

    async def _check_health(self) -> bool:
        """发送 ping 检测连接是否存活。"""
        if self._session is None:
            return False
        try:
            # MCP SDK 没有原生 ping，但 list_tools 是轻量调用
            await asyncio.wait_for(
                self._session.list_tools(), timeout=5.0
            )
            self._healthy = True
            self._reconnect_attempts = 0
            return True
        except Exception as e:
            self._healthy = False
            self._last_error = str(e)
            return False

    async def reconnect(self, config: dict) -> bool:
        """尝试重新建立连接。"""
        if self._reconnect_attempts >= self.MAX_RECONNECT_ATTEMPTS:
            logger.error(
                f"[MCPAdapter] '{self.name}' 重连次数已达上限 ({self.MAX_RECONNECT_ATTEMPTS})"
            )
            return False

        # 指数退避
        backoff = min(
            self.INITIAL_BACKOFF_MS * (2 ** self._reconnect_attempts),
            self.MAX_BACKOFF_MS,
        )
        self._reconnect_attempts += 1
        logger.info(
            f"[MCPAdapter] '{self.name}' 将在 {backoff/1000:.1f}s 后第 "
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
                await self.connect_sse(url=config.get("url", ""))
            self._healthy = True
            self._last_error = None
            logger.info(f"[MCPAdapter] '{self.name}' 重连成功")
            return True
        except Exception as e:
            logger.error(f"[MCPAdapter] '{self.name}' 重连失败: {e}")
            self._last_error = str(e)
            return False
```

**健康检查触发点**：

1. **定时心跳**：在 `api_server.py` lifespan 中启动后台任务，每 30s 检查一次
2. **调用前检测**：在 `get_mcp_tools()` 或工具调用前检查 `_healthy`，不健康则尝试重连
3. **异常捕获**：工具调用捕获 `McpError` / `ConnectionError` 后标记不健康并触发重连

```python
# api_server.py lifespan 中添加心跳任务

async def _mcp_health_monitor():
    """后台任务：定期检查 MCP 连接健康。"""
    from agent.tools.mcp_adapter import _adapters
    from utils.config_handler import agent_config

    servers_cfg = {cfg.get("name", ""): cfg for cfg in agent_config.get("mcp_servers", [])}

    while True:
        await asyncio.sleep(30)
        for name, conn in list(_adapters.items()):
            healthy = await conn._check_health()
            if not healthy:
                cfg = servers_cfg.get(name)
                if cfg:
                    success = await conn.reconnect(cfg)
                    if not success and conn._reconnect_attempts >= conn.MAX_RECONNECT_ATTEMPTS:
                        logger.warning(
                            f"[MCPHealth] '{name}' 已离线，Agent 将回退到本地模式"
                        )

# 在 lifespan 中启动
if mp_tool_mode == "mcp":
    # ... 现有初始化代码 ...
    asyncio.create_task(_mcp_health_monitor())
```

---

### 🟡 P1 — 配置分离：恢复 per-server 控制项

**问题**：`mcp_config.yaml` 被删除后，配置合并到 `agent_config.yaml`，但丢失了：
- 单 server `enabled` 开关
- 连接超时配置
- 敏感操作关键词列表
- stdio 进程的 env/cwd 控制

**Claude-Code 做法**：7 层配置作用域（enterprise → local → project → user → plugin → claudeai → dynamic），每 server 都有完整控制项。

**我们的修复方案**：

```yaml
# config/agent_config.yaml

mp_tool_mode: "mcp"

mcp_servers:
  - transport: "sse"
    url: "http://127.0.0.1:8002/sse"
    name: "material-project"
    enabled: true              # 新增：可单独禁用
    timeout: 60                # 新增：单次工具调用超时（秒）
    health_check_interval: 30  # 新增：心跳间隔（秒）

  # stdio 模式示例
  - transport: "stdio"
    command: ["python", "-m", "mp_material_server"]
    name: "material-stdio"
    enabled: false
    env:                       # 新增：可覆盖/扩展环境变量
      MP_API_KEY: "${MP_API_KEY}"
    cwd: "workspace"

# 敏感操作关键词（全局）
sensitive_keywords:
  - "mp_download"
  - "write"
  - "delete"
  - "exec"

# MCP 全局配置
mcp_global:
  max_reconnect_attempts: 5
  initial_backoff_ms: 1000
  max_backoff_ms: 30000
  tool_description_max_length: 2048   # 借鉴：防止 OpenAPI 大描述撑爆上下文
```

---

### 🟡 P1 — 大结果卸载（避免上下文爆炸）

**问题**：MCP 工具返回的大段文本（如 MP 查询返回大量材料数据、RAG 检索返回长文档）直接作为 `ToolMessage.content` 塞进对话历史，容易撑爆上下文窗口。

**Claude-Code 做法**：
- 结果超过阈值时，调用 `persistToolResult()` 写入磁盘
- Agent 收到的是 `"结果已保存到 <path>，请读取该文件"`
- 图片自动 resize/compress，二进制 blob 持久化

**我们的修复方案**（在工具调用结果处理层增加卸载逻辑）：

```python
# agent/tools/mcp_adapter.py

import tempfile
from pathlib import Path

# 阈值配置（可从 agent_config 读取）
MAX_INLINE_RESULT_CHARS = 100_000   # 10万字符以上卸载到文件

def _offload_large_result(content: str, tool_name: str) -> str:
    """如果结果过大，将其卸载到文件并返回文件路径提示。"""
    if len(content) <= MAX_INLINE_RESULT_CHARS:
        return content

    # 写入 workspace 临时文件
    offload_dir = Path("workspace/mcp_results")
    offload_dir.mkdir(parents=True, exist_ok=True)

    timestamp = asyncio.get_event_loop().time()
    filename = f"{tool_name}_{int(timestamp)}.txt"
    filepath = offload_dir / filename

    filepath.write_text(content, encoding="utf-8")

    logger.info(
        f"[MCPAdapter] 工具 '{tool_name}' 结果过大 ({len(content)} 字符)，"
        f"已卸载到 {filepath}"
    )

    return (
        f"[系统提示：工具返回结果过大，已保存到文件]\n"
        f"文件路径: {filepath}\n"
        f"文件大小: {len(content)} 字符\n"
        f"请使用 FileReadTool 读取该文件获取完整内容。"
    )
```

**如何接入**：需要包装 `convert_mcp_tool_to_langchain_tool` 生成的 tool，或修改调用后的结果处理。更简单的做法是在 Agent 的 ToolNode 层面拦截：

```python
# agent/graph/agent_graph.py — _make_tool_node_with_logging 中

async def _tool_node(state: AgentState) -> dict:
    result = await _tn.ainvoke(state)

    for msg in result.get("messages", []):
        if isinstance(msg, ToolMessage):
            # 大结果卸载
            if len(msg.content) > MAX_INLINE_RESULT_CHARS:
                from agent.tools.mcp_adapter import _offload_large_result
                msg.content = _offload_large_result(msg.content, msg.name)
            logger.info(f"[工具{msg.name}]调用成功")
    return result
```

---

### 🟡 P1 — stdio 进程生命周期管理

**问题**：当前 stdio 模式下，如果子进程崩溃或僵死，没有任何检测和清理机制。

**Claude-Code 做法**：三级信号清理：`SIGINT` → 100ms → `SIGTERM` → 400ms → `SIGKILL`，并轮询进程是否存在。

**我们的修复方案**：

```python
# agent/tools/mcp_adapter.py

import signal
import subprocess

class _MCPConnection:
    # ... 现有代码 ...
    _proc: subprocess.Popen | None = None   # 新增：跟踪 stdio 子进程

    async def connect_stdio(self, command, env=None, cwd=None):
        # ... 现有连接代码 ...
        # mcp SDK 的 stdio_client 内部会创建子进程，但我们拿不到 Popen 对象
        # 需要换种方式：先自己启动进程，再包装为 transport

        effective_env = env if env is not None else os.environ.copy()
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=effective_env,
            cwd=cwd,
        )
        # 然后用 mcp.client.stdio 的底层机制包装这个进程...
        # （或继续使用 stdio_client，但在 close 时增加进程存活检测）

    async def close(self) -> None:
        """优雅关闭连接，stdio 模式下尝试三级信号终止。"""
        if self.transport == "stdio" and self._proc is not None:
            proc = self._proc
            if proc.poll() is None:  # 进程仍在运行
                logger.info(f"[MCPAdapter] '{self.name}' 发送 SIGINT...")
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=0.5)  # 500ms 等待
                except subprocess.TimeoutExpired:
                    logger.warning(f"[MCPAdapter] '{self.name}' SIGINT 超时，发送 SIGTERM...")
                    proc.terminate()
                    try:
                        proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        logger.error(f"[MCPAdapter] '{self.name}' SIGTERM 超时，强制 SIGKILL")
                        proc.kill()
                        proc.wait()
            self._proc = None

        # 原有清理逻辑
        if self._session is not None:
            await self._session.__aexit__(None, None, None)
            self._session = None
        if self._transport_ctx is not None:
            await self._transport_ctx.__aexit__(None, None, None)
            self._transport_ctx = None
```

> 注：`mcp` SDK 的 `stdio_client` 内部已经管理了子进程，上述代码需要与其配合。当前建议先增加 **进程退出检测**（通过 transport 的 `onclose` 或定期检查 `poll()`），而非完全重写进程管理。

---

### 🟢 P2 — 连接健康检查与状态 API

**问题**：运维人员无法知道 MCP Server 是否在线、哪些工具可用、上次错误是什么。

**Claude-Code 做法**：`/mcp` 命令提供完整的 TUI 状态面板；内部状态机追踪 `pending → connected → failed → needs-auth → disabled`。

**我们的修复方案**：

```python
# api/routes/system.py（新增 endpoint）

from fastapi import APIRouter
from agent.tools.mcp_adapter import _adapters, _mcp_tools

router = APIRouter()

@router.get("/system/mcp/status", tags=["System"])
async def mcp_status():
    """返回所有 MCP Server 的连接状态。"""
    return {
        "mode": "mcp",
        "servers": [
            {
                "name": conn.name,
                "transport": conn.transport,
                "healthy": conn._healthy if hasattr(conn, "_healthy") else "unknown",
                "reconnect_attempts": getattr(conn, "_reconnect_attempts", 0),
                "last_error": getattr(conn, "_last_error", None),
            }
            for conn in _adapters.values()
        ],
        "tools": [
            {"name": t.name, "server": getattr(t, "server_name", "unknown")}
            for t in _mcp_tools
        ],
    }
```

前端可以在右栏参考资料区增加一个 "MCP 状态" 小面板，显示各 Server 的在线状态。

---

### 🟢 P2 — 认证层预留设计

**问题**：SSE 传输没有任何认证。虽然目前只是本地跑，但未来如果暴露到网络或被其他团队使用，就是安全漏洞。

**Claude-Code 做法**：
- OAuth 2.0 + OIDC：自动发现、Token 刷新、Step-up Auth
- Claude.ai Proxy：Bearer Token + 401 自动重试
- Token 存系统安全存储（Keychain / Windows Credential / Secret Service）

**我们的修复方案**（预留接口，暂不实现完整 OAuth）：

```yaml
# config/agent_config.yaml

mcp_servers:
  - transport: "sse"
    url: "http://127.0.0.1:8002/sse"
    name: "material-project"
    auth:                         # 新增：认证配置（预留）
      type: "bearer"              # 未来可扩展为 "oauth", "api_key"
      token: "${MCP_MP_TOKEN}"    # 从环境变量读取，不硬编码
      # oauth 模式示例（预留）
      # type: "oauth"
      # client_id: "xxx"
      # token_url: "https://..."
```

```python
# agent/tools/mcp_adapter.py — connect_sse 中增加 header 注入

async def connect_sse(self, url: str, auth: dict | None = None) -> None:
    headers = {}
    if auth:
        auth_type = auth.get("type", "").lower()
        if auth_type == "bearer":
            token = auth.get("token", "")
            # 支持环境变量展开
            if token.startswith("$"):
                token = os.environ.get(token[1:], "")
            elif token.startswith("${") and token.endswith("}"):
                token = os.environ.get(token[2:-1], "")
            if token:
                headers["Authorization"] = f"Bearer {token}"
        # oauth 模式可在此扩展...

    self._transport_ctx = sse_client(url, headers=headers)
    # ... 后续不变
```

---

### 🟢 P2 — 工具描述截断

**问题**：如果接入的 MCP Server 是从 OpenAPI 自动生成的（如 Swagger-to-MCP），工具 description 可能有几千字，直接撑爆 LLM 上下文窗口。

**Claude-Code 做法**：描述硬截断到 `MAX_MCP_DESCRIPTION_LENGTH = 2048` 字符。

**我们的修复方案**（一行修改）：

```python
# agent/tools/mcp_adapter.py

MAX_DESCRIPTION_LENGTH = 2048

async def init_mcp_tools(servers):
    # ... 在 convert_mcp_tool_to_langchain_tool 之后 ...
    for tool in response.tools:
        # 截断过长的描述
        if tool.description and len(tool.description) > MAX_DESCRIPTION_LENGTH:
            tool.description = tool.description[:MAX_DESCRIPTION_LENGTH] + "... [截断]"

        lc_tool = convert_mcp_tool_to_langchain_tool(...)
        _mcp_tools.append(lc_tool)
```

---

## 实施路线图

| 阶段 | 内容 | 预计改动文件数 | 风险 |
|------|------|-------------|------|
| **Phase 1（本周）** | P0 安全漏洞修复 + 敏感关键词配置恢复 | 2-3 | 低 |
| **Phase 2（下周）** | SSE 重连 + 健康检查 + 配置增强 | 3-4 | 中（需测试重连逻辑） |
| **Phase 3（下下周）** | 大结果卸载 + stdio 进程管理 | 2-3 | 中 |
| **Phase 4（后续）** | 状态 API + 认证预留 + 描述截断 | 2-3 | 低 |

---

## 与 Claude-Code 暂不采纳的差异

以下特性当前不需要，但未来可评估：

| 特性 | 暂不采纳原因 |
|------|------------|
| **XAA 企业认证** | 项目面向科研个人用户，无企业 SSO |
| **Channel Relay（手机审批）** | 需要外部消息渠道基础设施 |
| **IDE 专用传输（sse-ide/ws-ide）** | 无 IDE 插件生态 |
| **SDK 控制传输** | Python FastAPI 架构不同，无需 CLI↔SDK 进程间通信 |
| **参数诱导（Elicitation）** | 当前接入的 MP Server 无此需求 |
| **7 层配置作用域** | 过度设计，当前只需 user + project 两层 |
| **两阶段配置加载** | 无云端配置源（如 Claude.ai connectors）|

---

*本计划基于对 Claude-Code MCP 源码的完整分析和当前项目代码现状评估制定。*
