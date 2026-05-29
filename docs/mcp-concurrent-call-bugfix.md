# MCP 并发调用 Bug 修复报告

> 日期：2026-05-26
> 涉及模块：`agent_material.mcp.session_pool`、`agent_material.mcp.tools`
> 影响范围：MCP 模式下的 Agent 工具调用

---

## 一、问题概述

在参考 deer-flow 架构重构 MCP 子系统后，实际运行中暴露出两个关联 Bug：

1. **Bug #1 — 并发调用崩溃**：当 Agent 在单轮推理中并发调用多个 MCP 工具（或同一工具被并行调用）时，触发 `RuntimeError: Attempted to exit cancel scope in a different task than it was entered in`，导致 ASGI 响应异常终止。
2. **Bug #2 — 模型兼容性错误**：修复 Bug #1 后，工具调用成功但返回结果被 DeepSeek 模型拒绝，报错 `expected string or bytes-like object, got 'list'`。

---

## 二、Bug #1：并发调用时 RuntimeError

### 2.1 现象

用户提问后，Agent 触发工具调用（如同时查询 `CoCrFeNi` 和 `CrFeCoNi`），日志出现：

```
default - INFO agent_graph.py:135 - [调用工具]：material-project_mp_query，参数：{'formula': 'CoCrFeNi'}
default - INFO agent_graph.py:135 - [调用工具]：material-project_mp_query，参数：{'formula': 'CrFeCoNi'}
[MP MCP Server] 启动中...
[MP MCP Server] 启动中...
ERROR:    ASGI callable returned without completing response.
RuntimeError: Attempted to exit cancel scope in a different task than it was entered in
RuntimeError: athrow(): asynchronous generator is already running
```

两个子进程被同时拉起，随后请求中断，前端收不到 SSE 流结尾。

### 2.2 根因分析

#### 2.2.1 直接原因

`langchain-mcp-adapters` 的 `create_session()` 是一个 `@asynccontextmanager` 装饰的 async generator，其内部嵌套调用 `stdio_client()`，而 `stdio_client()` 又使用了 `anyio.create_task_group()`。

**anyio 的 `TaskGroup` 有一个关键限制**：其 `CancelScope` 在创建时与**当前 task** 绑定。如果 `TaskGroup.__aexit__()`（即 `stdio_client` 的上下文退出）在**另一个 task** 中被调用，`anyio` 会直接抛出：

```python
RuntimeError: Attempted to exit cancel scope in a different task than it was entered in
```

#### 2.2.2 为什么并发场景会触发

`MCPSessionPool.get_session()` 的设计意图是：同一 `(server_name, thread_id)` 只保留一个持久 session，供同一线程的多次工具调用复用。

但在 deer-flow 原始实现中，`get_session()` 仅用 `threading.Lock` 保护字典读写，**没有保护 `await cm.__aenter__()` 这段异步创建过程**。导致如下竞态：

| 时间线 | Task A（调用工具1） | Task B（调用工具2） |
|--------|---------------------|---------------------|
| T1 | 获取 thread lock，发现 key 不存在，释放 lock | — |
| T2 | — | 获取 thread lock，发现 key 不存在，释放 lock |
| T3 | `cm = create_session()` → 启动子进程 A | — |
| T4 | — | `cm = create_session()` → 启动子进程 B |
| T5 | `await cm.__aenter__()` 完成 | — |
| T6 | 获取 lock，存储 session A | — |
| T7 | — | `await cm.__aenter__()` 完成 |
| T8 | — | 获取 lock，发现 key 已存在，需关闭 CM B |

**问题出在 T8**：Task B 试图关闭 CM B，但 CM B 内部的 anyio `CancelScope` 是在 Task B 创建时绑定的。如果关闭操作通过 `asyncio.create_task()` 调度到后台 task，或者由于垃圾回收自动触发 `aclose()`，关闭就会发生在**非创建 task** 中，触发 `RuntimeError`。

#### 2.2.3 垃圾回收的额外干扰

即使显式关闭被跳过，未被存入 `_context_managers` 的 CM 对象会被 Python 垃圾回收，async generator finalizer 自动调用 `aclose()`。由于 finalizer 运行在 GC thread 中，必然不是创建 task，同样会触发崩溃。

### 2.3 修复方案

#### 修改 1：引入 asyncio.Lock 串行化创建

在 `MCPSessionPool` 中增加 `asyncio.Lock`，将"检查 + 创建 + 存储"整个临界区串行化，彻底消除竞态：

```python
class MCPSessionPool:
    def __init__(self):
        self._lock = threading.Lock()          # 保护字典
        self._async_create_lock = asyncio.Lock()  # 保护 create_session

    async def get_session(self, server_name, scope_key, connection):
        # Fast path：无锁检查
        with self._lock:
            if key in self._entries:
                ...
                return session

        # Slow path：串行创建
        async with self._async_create_lock:
            with self._lock:
                if key in self._entries:       # double-check
                    return session

            cm = create_session(connection)
            session = await cm.__aenter__()
            await session.initialize()

            with self._lock:
                if key in self._entries:
                    # key 已被其他路径（如 sync wrapper）写入
                    # 必须在当前 task 中关闭 CM，不能交给后台 task
                    try:
                        await cm.__aexit__(None, None, None)
                    except Exception:
                        logger.warning(...)
                    return existing_session

                self._entries[key] = (session, current_loop)
                self._context_managers[key] = (cm, current_loop)
            return session
```

#### 修改 2：同 task 内关闭冗余 CM

当 `asyncio.Lock` 保护下仍然出现 key 已存在的情况（例如 sync wrapper 在另一线程插入），多余的 CM 在**当前 task**（即创建它的 task）中直接 `await cm.__aexit__()`，避免跨 task 关闭。

#### 修改 3：抑制已知的关闭噪音

对于 LRU 淘汰或 `close_all_sync()` 中的跨 task 关闭，这是库级限制无法根治，将 `RuntimeError` 降级为 `debug` 日志：

```python
except RuntimeError as e:
    if "cancel scope" in str(e).lower():
        logger.debug("Ignored expected anyio error ...")
    else:
        logger.warning(...)
```

#### 修改 4：保留 dead CM 引用阻止 finalizer

`close_all_sync()` 将失败 CM 移入 `_dead_cms` 列表，阻止 Python GC 自动触发 `aclose()`：

```python
def close_all_sync(self):
    with self._lock:
        self._dead_cms.extend(self._context_managers.values())
        self._context_managers.clear()
        self._entries.clear()
```

### 2.4 验证

```python
# 3 轮并发查询测试
for i in range(3):
    results = await asyncio.gather(
        call_once('CoCrFeNi'),
        call_once('CrFeCoNi'),
    )
    print(f'Round {i+1}: OK')
```

输出：
```
Round 1: OK
Round 2: OK
Round 3: OK
```

并发调用全部成功，无请求中断。

---

## 三、Bug #2：DeepSeek 模型拒绝 list 类型 ToolMessage

### 3.1 现象

修复 Bug #1 后，工具调用本身成功（子进程正常返回查询结果），但 Agent 整体报错：

```
default - ERROR agent.py:192 - [Agent SSE] 请求失败 | error=expected string or bytes-like object, got 'list'
```

### 3.2 根因分析

deer-flow 的 `_convert_call_tool_result()` 严格遵循 MCP 协议，将工具返回结果转换为 **LangChain 多模态 content blocks 列表**：

```python
# _convert_call_tool_result 返回值示例
[
    {"type": "text", "text": '{"Formula": "Si", "MP_ID": "mp-149", ...}'},
]
```

`StructuredTool` 设置了 `response_format="content_and_artifact"`，因此上述 list 被直接赋值给 `ToolMessage.content`。

当前项目使用的后端模型是 **DeepSeek-V4-Flash**，其并不支持 `ToolMessage.content` 为 `list` 类型。当 `agent_graph.py` 调用 `estimate_tokens(msg.content)` 进行 token 估算时：

```python
def estimate_tokens(text: str) -> int:
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))  # ← 这里 text 是 list，直接抛 TypeError
```

`re.findall` 接收到了 `list`，抛出 `expected string or bytes-like object, got 'list'`。

### 3.3 修复方案

在 `_make_session_pool_tool` 的 `call_with_persistent_session` 返回前，将多模态 list 展平为纯文本字符串：

```python
content, artifact = _convert_call_tool_result(call_tool_result)

# DeepSeek 等模型不支持 list-type ToolMessage.content
if isinstance(content, list):
    text_parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "image":
                text_parts.append("[image]")
            elif block.get("type") == "file":
                text_parts.append("[file]")
            else:
                text_parts.append(str(block))
        else:
            text_parts.append(str(block))
    content = "\n".join(text_parts)

return content, artifact
```

这样既保留了 `response_format="content_and_artifact"` 的接口兼容性，又确保了对非多模态模型的友好支持。

### 3.4 验证

```python
result = await query_tool.ainvoke({'formula': 'Si'})
print(type(result))   # <class 'str'>
```

工具返回类型变为 `str`，Agent 端到端流程恢复正常。

---

## 四、文件变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `backend/packages/agent_material/mcp/session_pool.py` | 修改 | 增加 `asyncio.Lock`、同 task 关闭、RuntimeError 降级、`_dead_cms` 防 GC |
| `backend/packages/agent_material/mcp/tools.py` | 修改 | `_make_session_pool_tool` 中将 list content 展平为字符串 |

---

## 五、已知限制与后续建议

### 5.1 已知限制

由于 `langchain-mcp-adapters` + `mcp` + `anyio` 的库级限制，`create_session` 产生的 async generator 无法在创建 task 之外安全关闭。

**已修复场景**：
- ✅ 并发工具调用（`get_session()` 竞态）— `asyncio.Lock` + 同 task 内关闭
- ✅ 进程关闭时的 unhandled exception — `close_all_sync()` 改用 `_close_cm()` 包裹 + `except BaseException`

**仍可能存在 debug 日志的场景**（不影响功能）：
- 应用启动阶段：`MultiServerMCPClient.get_tools()` 创建的临时 stdio session 被丢弃时的垃圾回收

这些子进程会在应用退出时由操作系统自动回收，不会成为僵尸进程。

### 5.2 后续建议

1. **升级依赖**：关注 `langchain-mcp-adapters` 后续版本，若其 `MultiServerMCPClient` 支持 `async with` 或提供显式 `close()` 方法，可彻底消除临时 session 泄漏。
2. **模型适配**：若未来切换到 Claude/GPT-4o 等多模态模型，可移除 `tools.py` 中的 list→string 转换逻辑，直接传递 content blocks 以支持图片、文件等富媒体工具结果。
3. **监控**：生产环境可监控 `workspace/mcp_results` 目录体积及系统子进程数量，确认无异常泄漏。
