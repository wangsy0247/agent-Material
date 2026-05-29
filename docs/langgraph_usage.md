# 本项目 LangGraph 使用方式总结

> 生成时间：2026-05-22  
> 对应代码版本：agent/graph/agent_graph.py, agent/React_agent.py

---

## 一、架构概览

本项目使用 **LangGraph** 作为 Agent 编排的核心框架，构建了一个支持**长时记忆、工具调用、人工审核（HITL）和流式输出**的 ReAct 风格智能体。

核心设计思想：
- **显式状态机**：用 `StateGraph` 明确定义节点和边，替代黑盒式的 `create_react_agent`
- **人工介入（HITL）**：敏感操作前自动暂停，等待人工审批后继续
- **迭代保护**：硬编码 10 轮上限，防止无限循环
- **状态隔离**：每次调用生成新 `thread_id`，避免历史消息污染

---

## 二、核心文件职责

| 文件 | 职责 |
|------|------|
| `agent/graph/agent_graph.py` | **图定义中心**：定义 `StateGraph`、`AgentState`、所有节点和边 |
| `agent/React_agent.py` | **图执行封装**：编译图、处理流式输出、实现 HITL 中断与恢复 |
| `agent/graph/__init__.py` | 导出 `AgentState` 和 `build_agent_graph` |
| `api/routes/agent.py` | FastAPI 路由层，通过 SSE 向前端暴露图的流式能力 |
| `config/agent_config.yaml` | 配置敏感关键词（触发 HITL）和 MCP 全局参数 |

---

## 三、状态定义：AgentState

```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    memory_context: str        # 长时记忆上下文
    tool_calls: list[dict]     # 待执行的工具调用列表
    iteration_count: int       # 迭代计数（防循环）
    session_id: str            # 会话 ID
    pending_approval: dict | None  # 挂起状态
    query: str                 # 用户原始查询
    final_answer: str          # 最终答案
```

**关键设计**：
- `messages` 使用 `add_messages` **reducer**，自动合并消息列表而非覆盖
- `iteration_count` 用于条件边判断，防止工具调用循环无限进行
- `pending_approval` 记录 HITL 审批状态

---

## 四、节点（Nodes）详解

```python
graph.add_node("memory", partial(memory_node, memory_manager=memory_manager))
graph.add_node("agent", partial(agent_node, chat_model=chat_model, tools=tools))
graph.add_node("human_approval", human_approval_node)
graph.add_node("tools", _make_tool_node_with_logging(tools))
graph.add_node("finalize", finalize_node)
```

| 节点 | 功能 | 关键实现 |
|------|------|----------|
| `memory` | 组装系统提示 + 长时记忆 + 对话历史 | 调用 `MemoryManager.build_context()`，将记忆注入到 `messages` |
| `agent` | 调用 LLM（已绑定工具）生成回复或工具调用 | `chat_model.bind_tools(tools)` + `ainvoke()` |
| `human_approval` | 检测敏感工具，触发人工审核 | 使用 `interrupt()` 暂停图执行 |
| `tools` | 执行工具调用 | 基于 `ToolNode` 预构建节点，含日志和结果卸载 |
| `finalize` | 提取最终答案 | 处理迭代上限截断，从最后一条 AIMessage 提取答案 |

---

## 五、图结构与条件边

### 5.1 图的可视化结构

```
entry → memory → agent → should_continue
              ├─ iteration > 10 ──→ finalize ──→ END
              ├─ has tool_calls ──→ human_approval
              │         ├─ approved ──→ tools ──→ agent (循环)
              │         └─ rejected ──→ agent (循环)
              └─ no tool_calls ──→ finalize ──→ END
```

### 5.2 条件边定义

**从 agent 节点分支**：
```python
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "max_iterations": "finalize",  # 超过10轮 → 强制结束
        "tools": "human_approval",     # 有工具调用 → 人工审核
        "end": "finalize"              # 无工具调用 → 正常结束
    },
)
```

**从 human_approval 节点分支**：
```python
graph.add_conditional_edges(
    "human_approval",
    after_approval,
    {
        "approved": "tools",   # 批准 → 执行工具
        "rejected": "agent"    # 拒绝 → 返回 LLM 重试
    },
)
```

---

## 六、人工介入（HITL）机制

### 6.1 中断：`interrupt()`

```python
from langgraph.types import interrupt

def human_approval_node(state: AgentState) -> dict:
    pending = [tc for tc in tool_calls if _is_sensitive_tool(tc["name"])]
    if not pending:
        return {}  # 无需审批，直接放行

    approval = interrupt({
        "tool_calls": [...],
        "message": "检测到敏感操作，请确认是否执行。"
    })
    # 图执行在此处暂停，状态保存到 checkpoint
```

### 6.2 恢复：`Command(resume=...)`

```python
from langgraph.types import Command

async for chunk in self.graph.astream(
    Command(resume=approval_result),  # 从 checkpoint 恢复
    config,
    stream_mode="updates"
):
    ...
```

### 6.3 HITL 执行流程

1. Agent 生成敏感工具调用（如 `mp_download`）
2. `human_approval_node` 调用 `interrupt()`，图暂停
3. 流式输出中出现 `"__interrupt__"` key，前端收到审批请求
4. 用户通过 `/agent/approve` 提交审批结果
5. `resume_stream()` 使用 `Command(resume=...)` 恢复执行
6. 根据审批结果：批准 → 执行工具；拒绝 → 生成拒绝 ToolMessage 返回给 LLM

---

## 七、Checkpointer 与会话隔离

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
return graph.compile(checkpointer=checkpointer)
```

**设计要点**：
- `MemorySaver` 是内存型检查点器，支持 HITL 的断点续行
- 通过 `config = {"configurable": {"thread_id": thread_id}}` 隔离不同会话
- **关键设计**：每次调用生成新 `thread_id`（`session_id + uuid[:8]`）
  - 优点：避免旧消息累积污染 prompt
  - 代价：牺牲了同一会话内的连续 checkpoint 历史

---

## 八、流式输出

```python
async for chunk in self.graph.astream(
    initial_state,
    config,
    stream_mode="updates"  # 以节点更新为单位流式输出
):
    if "__interrupt__" in chunk:
        # 检测 HITL 中断
        ...

    agent_update = chunk.get("agent")
    if agent_update:
        # 只处理 agent 节点输出，过滤中间节点噪音
        ...
```

**stream_mode 说明**：
- `"updates"`：每次节点完成后输出该节点的增量更新（本项目使用）
- `"values"`：每次状态更新后输出完整状态
- `"debug"`：包含内部调试信息

---

## 九、工具节点增强设计

### 9.1 大结果卸载

当工具返回超过阈值（默认 10 万字符）时：
- 将完整结果写入 `workspace/mcp_results/<tool_name>_<timestamp>.txt`
- 用文件路径提示替换原始内容，防止上下文窗口爆炸

### 9.2 敏感工具判定

支持两种匹配模式：
1. **精确匹配**：`tool_name` 在敏感列表中
2. **后缀匹配**：去掉 MCP server 前缀后的工具名匹配
   - MCP 工具名格式：`<server_name>__<tool_name>`
   - 例如：`material_project__mp_download` → 提取 `mp_download`

---

## 十、LangGraph API 速查表

### 10.1 核心类与函数

| API | 来源 | 用途 |
|-----|------|------|
| `StateGraph` | `langgraph.graph` | 显式状态机图定义 |
| `END` | `langgraph.graph` | 图的终止节点常量 |
| `add_messages` | `langgraph.graph.message` | 消息列表 reducer，自动合并 |
| `ToolNode` | `langgraph.prebuilt` | 预构建工具执行节点 |
| `MemorySaver` | `langgraph.checkpoint.memory` | 内存型 checkpoint |
| `interrupt` | `langgraph.types` | HITL 中断 |
| `Command` | `langgraph.types` | 恢复执行/向图发送命令 |

### 10.2 图操作方法

| 方法 | 说明 |
|------|------|
| `graph.add_node(name, func)` | 添加节点 |
| `graph.add_edge(source, target)` | 添加普通边 |
| `graph.add_conditional_edges(source, router, path_map)` | 添加条件边 |
| `graph.set_entry_point(node)` | 设置入口节点 |
| `graph.compile(checkpointer=...)` | 编译图为可执行对象 |

### 10.3 编译后图方法

| 方法 | 说明 |
|------|------|
| `compiled_graph.invoke(state, config)` | 同步调用 |
| `compiled_graph.ainvoke(state, config)` | 异步调用 |
| `compiled_graph.stream(state, config)` | 同步流式 |
| `compiled_graph.astream(state, config)` | 异步流式（本项目使用） |
| `compiled_graph.get_graph()` | 获取图结构用于可视化 |

---

## 十一、设计亮点与最佳实践

1. **状态隔离优于状态累积**：每次新调用生成独立 `thread_id`，避免历史消息污染，适合对话记忆由外部 `MemoryManager` 管理的架构。

2. **双重循环保护**：
   - 条件边 `should_continue` 的 `max_iterations` 分支
   - 硬编码 `iteration_count > 10` 强制截断

3. **HITL 零侵入设计**：敏感工具检测完全封装在 `human_approval_node` 中，不影响正常工具的执行路径。

4. **流式输出精确过滤**：只监听 `chunk.get("agent")`，避免将工具执行日志、记忆组装等内部节点输出暴露给用户。

5. **MCP 命名空间兼容**：通过 `__` 分隔符解析 MCP 工具名，实现对第三方工具的人工审批。

---

## 十二、可视化代码示例

在 Jupyter 中显示图结构：

```python
from agent.graph import build_agent_graph
from IPython.display import Image, display

graph = build_agent_graph(chat_model, tools)
display(Image(graph.get_graph().draw_mermaid_png()))
```

保存为文件：

```python
png_bytes = graph.get_graph().draw_mermaid_png()
with open("agent_graph.png", "wb") as f:
    f.write(png_bytes)
```

获取 Mermaid 文本：

```python
print(graph.get_graph().draw_mermaid())
```

---

## 十三、相关依赖

```toml
# pyproject.toml 中需包含
langchain>=0.3.0
langgraph>=0.2.0
```

> 注：本项目未显式声明 `langgraph` 版本，由 `langchain` 传递依赖引入。
