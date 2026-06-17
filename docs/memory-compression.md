# 记忆压缩机制技术文档

## 概述

Agent Material 的记忆压缩系统由**两层触发**和一个**四阶段压缩算法**组成。压缩不在 ShortTermMemory 内部执行，而是由 LangGraph 的 `summarization_node` 统一调度，复用 `ContextCompressor` 作为核心压缩引擎。

---

## 架构位置

```
entry → memory_node → should_summarize → summarization_node → agent_node → ...
                          │                      │
                   消息数 > 20?           调用 ContextCompressor
                          │                  .compress()
                     ┌────┴────┐
                     │         │
                   need    no_need
                   sum       sum
                     │         │
                     ▼         ▼
              summarize    agent_node
```

具体文件：

| 文件 | 角色 |
|------|------|
| `memory/context_compressor.py` | 四阶段压缩算法实现 |
| `memory/short_term.py` | 消息列表和 Token 计数维护（不执行压缩） |
| `memory/manager.py` | 三层记忆协调整合 |
| `memory/models.py` | ContextWindow 等数据模型 |
| `agents/graph/agent_graph.py` | LangGraph 的 `summarization_node`，压缩调度的入口 |
| `config/memory_config.yaml` | 压缩提示词和参数配置 |
| `config/agent_config.yaml` | summarization 触发阈值配置 |

---

## 触发条件

压缩有**两道防线**：

### 第一道：`should_summarize` 条件边

`agent_graph.py:602`

```python
chat_count = len([m for m in messages if not isinstance(m, SystemMessage)])
if chat_count > _SUMMARIZATION_THRESHOLD:  # 默认 20
    return "summarize"
```

### 第二道：`summarization_node` 入口守卫

`agent_graph.py:342`

```python
if len(chat_msgs) <= _SUMMARIZATION_PRESERVE_LAST_N + 2:  # 默认 6 + 2 = 8
    return {}  # 消息太少，不值得压缩
```

配置项（`config/agent_config.yaml`）：

| 配置键 | 默认值 | 说明 |
|--------|--------|------|
| `summarization.enabled` | true | 是否启用 summarization |
| `summarization.message_threshold` | 20 | 消息数阈值 |
| `summarization.preserve_last_n` | 6 | 压缩时保留的尾部消息数 |
| `summarization.max_tokens` | 4000 | 压缩后目标 Token 数 |

---

## 四阶段压缩算法

### Phase 1: 预处理（零 LLM Token 成本）

`_preprocess_messages()` — `context_compressor.py:340`

两种纯规则压缩，在消息进入 LLM 摘要之前执行：

#### Pass 1: Smart Collapse — 大型工具输出替换

当 `tool` 消息内容超过 200 字符时，替换为工具专属的信息化单行摘要。

`_get_tool_collapse_info()` — `context_compressor.py:85`

```
原始 (3847 chars):
  ToolMessage(rag_search → {huge JSON result})

替换为:
  [rag_search] query='LiFePO4 band gap' -> 3847 chars
```

已注册的工具摘要格式：

| 工具名 | 摘要格式 |
|--------|---------|
| `rag_search` / `rag_query` | `[rag_search] query='...' -> N chars` |
| `rag_load_documents` | `[rag_load_documents] -> N chars` |
| `rag_add_documents` | `[rag_add_documents] -> N chars` |
| `rag_delete_by_source` | `[rag_delete_by_source] source='...' -> N chars` |
| `rag_get_stats` | `[rag_get_stats] -> N chars` |
| `mp_query` | `[mp_query] formula='...' -> N chars` |
| `mp_download` | `[mp_download] formula='...' -> saved, N chars` |
| `mp_list` | `[mp_list] -> N chars` |
| 其他 MCP 工具 | `[name] args={...}... -> N chars` |

设计原则：**保留操作参数**（query/formula/source 的前 50 字符），让 LLM 能理解"做了什么"，而非仅知道"某工具返回了大量数据"。

#### Pass 2: 工具调用参数截断

当 assistant 消息的 `tool_calls[].function.arguments` 超过 500 字符时，截断到 200 字符并追加 `...[truncated]`。

`_truncate_tool_call_args()` — `context_compressor.py:376`

```
原始:
  {"function": {"arguments": "very long string > 500 chars ..."}}

截断后:
  {"function": {"arguments": "first 200 chars...[truncated]"}}
```

---

### Phase 2: 保护区分离与边界自适应

`compress()` — `context_compressor.py:233-254`

将消息列表切分为三个区域：

```
[head区: protect_first_n条] [middle区: 将被压缩为摘要] [tail区: protect_last_n条]
```

`summarization_node` 的调用参数：
- `protect_first_n=0` — system message 已预先排除，无需 head 保护
- `protect_last_n=6` — 保留最近 6 条非 system 消息

#### 自适应扩展 1：最后一条用户消息保护

`_find_last_user_message_index()` + 扩展逻辑 — `context_compressor.py:403` / `compress():234`

**问题**：如果最后一条 user 消息被划入 middle 区，压缩后 LLM 摘要会"总结"它，Agent 可能不会直接回复。

**解决**：检测到 `last_user_msg_index` 在 middle 区中时，自动扩大 `protect_last_n` 将该消息纳入 tail。

```
原始 protect_last_n=6:
  [...middle包含last_user..., T, T, T, T, T, T]
                              ↑ tail起始

扩展后:
  [...middle..., last_user, T, T, T, T, T, T]
                 ↑ tail起始(扩大)
```

#### 自适应扩展 2：工具调用对完整性

`_expand_tail_for_tool_integrity()` — `context_compressor.py:411`

**问题**：Anthropic API 要求 `ToolMessage`（tool_result）必须有对应的 `AIMessage.tool_calls`（tool_use）在其之前。如果 tail 保留了 ToolMessage 但其 parent AIMessage 在 middle 中被压缩，API 返回 400。

**算法**（递归向前扫描）：

```
1. 建立 tool_call_id → parent_AIMessage_index 映射
2. 扫描 tail 中的所有 ToolMessage
3. 若 ToolMessage.parent_AIMessage_index < tail_start:
      tail_start = parent_AIMessage_index
      重新扫描
4. 递归直到稳定

示例:
  消息: [AI(tc_id=A), Tool(A), User, AI, Tool(tc_id=B)]
                                        ↑ tail_start(默认)
  Tool(B)的parent不在tail中! → tail_start前移
  结果: [AI(tc_id=A), Tool(A), User, AI, Tool(tc_id=B)]
        ↑ 新的tail_start
```

---

### Phase 3: 结构化 LLM 摘要（核心）

`compress()` — `context_compressor.py:266-290`

#### 两种模式

| 模式 | 条件 | 行为 |
|------|------|------|
| 首次压缩 | `_previous_summary is None` | 全量 11 章节结构化摘要 |
| 迭代更新 | 已有旧摘要 | 合并旧摘要 + 新对话 |

#### 消息格式化

`_format_messages_for_compression()` — `context_compressor.py:458`

将 middle 区的消息序列化为带角色标签的文本：

```
[用户] LiFePO4的带隙是多少？
[助手] 让我搜索一下相关资料 [Tool calls: rag_search]
[工具] [rag_search] query='LiFePO4 band gap' -> 3847 chars
[助手] 根据查询结果，LiFePO4的带隙约为0.6-0.7eV [Tool calls: mp_query]
[工具] [mp_query] formula='LiFePO4' -> 1243 chars
```

单条消息超过 3000 字符时保留头尾（前 2/3 + 后 1/4），中间插入截断标记。

#### 首次压缩的 11 章节提示词

配置于 `config/memory_config.yaml` 的 `compression.prompt_template`：

| 章节 | 说明 |
|------|------|
| Goal | 用户正在解决的材料科学问题或任务 |
| Constraints & Preferences | 用户偏好、约束、重要决策 |
| Completed Actions | 已完成动作的编号列表（含工具名、参数、结果） |
| Active State | 当前状态：涉及的材料、计算结果、文件 |
| In Progress | 压缩触发时正在进行的工作 |
| Blocked | 未解决的阻塞或错误（含完整错误消息） |
| Key Decisions | 重要技术决策及理由 |
| Resolved Questions | 已问已答的问题（含答案，避免重复回答） |
| Pending User Asks | 已问但未答的问题 |
| Relevant Files/Data | 涉及的文献、材料结构ID、数据文件 |
| Remaining Work | 未完成的任务 |
| Critical Context | 具体值、错误消息、配置细节等不可丢失的信息 |

#### 迭代更新提示词

配置于 `compression.prompt_template_update`：

核心指令：
1. **ADD** 新的 completed actions 到编号列表（继续编号，不重置）
2. 把 "In Progress" 中已完成的事项移到 "Completed Actions"
3. 更新 "Active State" 反映当前状态
4. 仅在明显过时时才移除旧信息

每次压缩产生**累积摘要**而非替换，避免信息丢失。

---

### Phase 4: 组装与清洗

#### ContextCompressor 内部

`compress()` — `context_compressor.py:292-312`

```
压缩结果 = head 消息 + 摘要消息(role=system) + tail 消息
```

摘要消息格式：
```
【上文摘要】
<LLM 生成的 11 章节结构化摘要>
【摘要结束，继续对话】
```

#### summarization_node 中的二次清洗

`agent_graph.py:371-387`

1. 用 `RemoveMessage(id=...)` 删除所有旧的对话消息（由 LangGraph checkpointer 生效）
2. 将摘要的 SystemMessage 转换为 AIMessage：
   ```
   原始: SystemMessage(content="【上文摘要】...")
   转换: AIMessage(content="【上文摘要】\n...\n【摘要结束，继续对话】")
   ```
   转换原因：避免与 `memory_node` 注入的 system message 冲突
3. 插入转换后的压缩结果

---

## 反颠簸机制

`should_compress()` — `context_compressor.py:172`

### 问题

如果压缩节省的 Token 太少（比如 500 token 的历史压完还是 480 token），说明用户消息本身就短或信息密度高，继续压缩只是在浪费 LLM 调用。

### 机制

```
配置常量:
  _ANTITHRASHING_THRESHOLD = 0.10   # 节省比例阈值
  _ANTITHRASHING_MAX_COUNT = 2      # 连续低效次数上限

每次 compress() 成功后:
  saved_tokens = original_tokens - compressed_tokens
  saved_ratio = saved_tokens / original_tokens

  if saved_ratio < 0.10:
      ineffective_count += 1
      if ineffective_count >= 2:
          记录: "反颠簸锁定 — 建议清空会话重新开始"
          后续 should_compress() 返回 False (拒绝压缩)
  else:
      ineffective_count = 0  # 重置计数器
```

### 状态机

```
  ┌──────────┐  节省≥10%   ┌──────────┐
  │  正常    │──────────→│  count=0  │
  │ count=0  │←──────────│  (重置)   │
  └────┬─────┘           └──────────┘
       │ 节省<10%
       ▼
  ┌──────────┐  节省<10%   ┌──────────┐  节省<10%   ┌──────────┐
  │ count=1  │──────────→│ count=2  │──────────→│  锁定    │
  │ (警告)   │           │ (警告)   │           │ (不再压缩) │
  └──────────┘           └──────────┘           └──────────┘
       │ 节省≥10%              │ 节省≥10%
       └───────────────────────┴──────────→ 重置 count=0
```

---

## 分级冷却机制

`_set_cooldown()` — `context_compressor.py:503`

### 问题

如果 LLM 摘要失败（网络故障、模型不可用、Token 超限），不停重试只会加重系统负载。

### 机制

根据异常类型设置不同冷却时间：

| 错误类型 | 异常类 | 冷却时长 | 原因 |
|---------|--------|---------|------|
| 配置类错误 | `RuntimeError`, `ConfigurationError`, `ValueError` | 600 秒 | 不会自愈，需人工介入 |
| 瞬态错误 | 其他所有异常（网络、超时、限流等） | 60 秒 | 可能快速自愈 |

### 行为

```
should_compress() 检查:
  if time.monotonic() < self._cooldown_until:
      return False  # 跳过压缩，不抛出异常

compress() 成功:
  self._cooldown_until = 0.0  # 重置冷却
```

使用 `time.monotonic()`（单调时钟）而非 `time.time()`，确保冷却时长不受系统时间调整影响。

### 回退策略

`_truncate_tool_results()` — `context_compressor.py:531`

当消息太少不值得压缩或 LLM 摘要失败时，执行轻量级截断：
- 超过 3000 字符的工具结果保留头尾、中间省略
- 保证压缩完全失效时消息历史仍然可用

---

## 完整生命周期

```
summarization_node 触发
  │
  ▼
should_compress() 检查
  ├─ Token 不足阈值?     → 跳过
  ├─ 冷却中?             → 跳过
  ├─ 反颠簸锁定?          → 跳过，建议清空会话
  └─ 通过 → compress()
              │
              ├─ 消息太少 (< protect_first + protect_last + 2)?
              │     └─ _truncate_tool_results() 仅截断
              │
              ├─ Phase 1: _preprocess_messages()
              │    ├─ Smart Collapse: tool 结果 > 200 chars → 一行摘要
              │    └─ 参数截断: arguments > 500 chars → 200 + [truncated]
              │
              ├─ Phase 2: 边界自适应
              │    ├─ 最后一条用户消息保护 → 扩大 tail
              │    └─ 工具对完整性检查 → 递归前移 tail_start
              │
              ├─ Phase 3: LLM 摘要
              │    ├─ 首次压缩? → 11 章节全量摘要模板
              │    └─ 迭代更新? → 合并旧摘要模板
              │    └─ 失败? → _set_cooldown() + _truncate_tool_results()
              │
              ├─ Phase 4: 组装
              │    └─ head + summary_msg + tail
              │
              └─ 反颠簸评估
                   ├─ 节省 < 10%? → ineffective_count++
                   └─ 节省 >= 10%? → ineffective_count = 0
```
