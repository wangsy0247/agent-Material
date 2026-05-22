# Hermes Agent Context Engine & Context Compressor 深度分析报告

> 分析对象：Nous Research Hermes Agent 的上下文管理系统  
> 核心源码：`agent/context_engine.py` (211 行) / `agent/context_compressor.py` (1575+ 行)  
> 参考来源：Hermes Wiki、官方文档、mem0.ai 生产环境分析、hermes-lcm 插件源码  
> 生成日期：2026-05-20  
> 报告位置：`knowledge/hermes-context-engine-compressor-analysis.md`

---

## 目录

1. [总体架构定位](#1-总体架构定位)
2. [Context Engine — 可插拔接口层](#2-context-engine--可插拔接口层)
3. [Context Compressor — 四阶段压缩算法](#3-context-compressor--四阶段压缩算法)
4. [核心组件详解](#4-核心组件详解)
5. [生产环境故障模式与边界案例](#5-生产环境故障模式与边界案例)
6. [设计演进：v1 → v2 → v3](#6-设计演进v1--v2--v3)
7. [与 Claude Code (OpenClaw) 压缩机制对比](#7-与-claude-code-openclaw-压缩机制对比)
8. [对 agent-material 项目的借鉴建议](#8-对-agent-material-项目的借鉴建议)

---

## 1. 总体架构定位

Hermes Agent 的上下文管理系统采用 **"双层压缩 + 可插拔引擎"** 架构：

```
┌──────────────────────────────────────────────────────────────┐
│  Layer 2: Gateway Session Hygiene (网关层卫生检查)            │
│  位置: gateway/run.py                                        │
│  触发阈值: 85% 上下文窗口                                    │
│  职责: 捕获跨轮次膨胀的会话（如 Telegram 隔夜累积）          │
├──────────────────────────────────────────────────────────────┤
│  Layer 1: Agent ContextCompressor (主压缩引擎)                │
│  位置: agent/context_compressor.py                           │
│  触发阈值: 50% 上下文窗口（可配置）                          │
│  职责: 正常运行时的上下文管理                                │
├──────────────────────────────────────────────────────────────┤
│  底座: ContextEngine ABC (可插拔接口)                         │
│  位置: agent/context_engine.py                               │
│  职责: 定义上下文管理合约，支持插件替换内置实现              │
└──────────────────────────────────────────────────────────────┘
```

**关键设计决策**：两层阈值故意错开（50% vs 85%）。早期将网关层也设为 50% 时，导致长会话每轮都触发过早压缩。85% 阈值的存在意义就是**不干扰主压缩引擎的正常工作**。

---

## 2. Context Engine — 可插拔接口层

### 2.1 抽象基类合约

`ContextEngine` ABC 定义了上下文管理必须实现的接口：

| 方法/属性 | 类型 | 职责 |
|-----------|------|------|
| `name` | property | 引擎标识字符串（如 `"compressor"`、`"lcm"`） |
| `should_compress(prompt_tokens)` | required | 判断是否达到压缩条件 |
| `compress(messages, current_tokens)` | required | 执行压缩，返回新消息列表 |
| `update_from_response(usage)` | optional | 每轮 API 调用后追踪 Token 用量 |
| `on_session_start/end/reset()` | optional | 会话生命周期钩子 |
| `get_tool_schemas()` | optional | 向 Agent 暴露引擎专用工具（如 `lcm_grep`） |
| `handle_tool_call(tool_call)` | optional | 处理 Agent 对引擎工具的调用 |
| `update_model(model_name)` | optional | 切换模型时重新校准参数 |

### 2.2 插件注册与发现

引擎通过 `config.yaml` 中的一行配置切换：

```yaml
context:
  engine: "compressor"   # 默认内置有损摘要
  engine: "lcm"          # 无损上下文管理插件示例
```

**解析顺序**：
1. 检查 `plugins/context_engine/<name>/` 目录
2. 检查通用插件系统的 `register_context_engine()` 注册表
3. 回退到内置 `ContextCompressor`

**重要约束**：
- 插件引擎**永不自动激活**，必须显式配置
- 任意时刻**只允许一个引擎活跃**
- 内置 `"compressor"` 永远指向默认实现，不受插件影响

### 2.3 已知插件实现

| 插件名 | 策略 | 存储结构 | 典型场景 |
|--------|------|----------|----------|
| `compressor` (内置) | LLM 结构化摘要 | 内存消息列表 | 通用默认 |
| `lcm` (社区) | 无损 DAG + SQLite FTS | DAG 节点 + 向量检索 | 超长会话精确回忆 |
| `dcp` (PR #20774) | 动态上下文修剪 | 压缩块 + 稳定引用 | 模型自主管理上下文 |

---

## 3. Context Compressor — 四阶段压缩算法

`ContextCompressor.compress()` 是核心方法，采用 **"先预处理瘦身，再 LLM 摘要"** 的策略。当前生产版本为 v3（2026-04-14+）。

### 3.1 配置参数与计算值

```yaml
compression:
  enabled: true
  threshold: 0.50          # 触发阈值（占上下文窗口比例）
  target_ratio: 0.20       # 尾部保护预算 = threshold_tokens × target_ratio
  protect_last_n: 20       # 尾部硬底线消息数

auxiliary:
  compression:
    model: null            # 摘要专用模型（null = 自动选择）
    provider: auto         # 模型提供方
```

**以 200K 上下文模型为例的计算值**：

| 指标 | 公式 | 结果 |
|------|------|------|
| threshold_tokens | 200,000 × 0.50 | **100,000** |
| tail_token_budget | 100,000 × 0.20 | **20,000** |
| max_summary_tokens | min(200,000 × 0.05, 12,000) | **10,000** |

### 3.2 四阶段流程

```
Phase 1: 廉价预处理（纯本地，零 Token 成本）
    ├── Pass 1: MD5 去重
    ├── Pass 2: Smart Collapse（信息化单行摘要）
    └── Pass 3: tool_call 参数截断

Phase 2: 边界确定
    ├── 保护头部（system + 前 3 条非 system 消息）
    └── 按 Token 预算保护尾部（硬底线 3 条消息）

Phase 3: LLM 结构化摘要
    ├── 序列化中间轮次为带标签文本
    ├── 计算摘要预算（content_tokens × 0.20）
    └── 调用辅助 LLM 生成结构化摘要

Phase 4: 组装与清洗
    ├── 选择摘要消息角色（避免连续同角色）
    ├── 修复孤儿 tool_call / tool_result 对
    └── Session 分裂（保留原始消息到 DB）
```

### 3.3 Phase 1：廉价预处理（v3 核心改进）

这是 v3 相比 v2 最重要的增强——**在调用 LLM 之前，通过纯本地操作砍掉 30-50% 的 Token**。

#### Pass 1 — MD5 去重

**机制**：对长度 >200 字符的工具结果计算 MD5，相同内容只保留最新一份。

```
旧副本替换为:
[Duplicate tool output — same content as a more recent call]
```

**典型场景**：反复 `read_file` 同一文件、反复 `search_files` 相同模式。

#### Pass 2 — Smart Collapse

v2 使用通用占位符 `_PRUNED_TOOL_PLACEHOLDER`，v3 改为**按工具类型生成信息化单行摘要**：

| 工具类型 | 摘要模板示例 |
|----------|-------------|
| `terminal` | `[terminal] ran \`npm test\` -> exit 0, 47 lines output` |
| `read_file` | `[read_file] read config.py from line 1 (1,200 chars)` |
| `search_files` | `[search_files] content search for 'compress' in agent/ -> 12 matches` |
| `patch` | `[patch] replace in config.py (1,500 chars result)` |
| `web_search` | `[web_search] query='cache control' (5,200 chars result)` |
| `delegate_task` | `[delegate_task] 'refactor auth module' (8,400 chars result)` |
| `memory` | `[memory] save on long-term` |
| fallback | `[tool_name] arguments... (N chars result)` |

**设计意义**：相比通用占位符，Smart Collapse 保留了**具体命令、文件路径、结果规模**，使模型在阅读历史时仍能知道"之前做过什么"。

#### Pass 3 — tool_call 参数截断

 Assistant 消息中的 `tool_calls.function.arguments` 长度 >500 字符时，截断到前 200 字符 + `...[truncated]`。

**修复场景**：`write_file(content=50KB)` 这类调用，即使工具结果被修剪，参数本身仍占大量上下文。

**多模态保护**：三个 Pass 都检测 `isinstance(content, list)`，跳过多模态消息，避免破坏图像/音频内容。

### 3.4 Phase 2：边界确定

#### 头部保护

固定保护前 3 条消息（`protect_first_n`，硬编码）：
- `[0]` system prompt
- `[1]` 用户首轮消息
- `[2]` 助手首轮响应

**保护原因**：头部包含任务定义和初始约束，压缩后容易丢失"原始意图"。

#### 尾部保护（Token 预算 + 硬底线）

```python
def _find_tail_cut_by_tokens(messages, head_end, token_budget):
    min_tail = min(3, n - head_end - 1)        # 硬底线：至少 3 条
    soft_ceiling = int(token_budget * 1.5)     # 软上限：允许 1.5 倍超额
    
    # 从末尾向前累加 token，直到超过 soft_ceiling 且满足 min_tail
    # 预算不足 → 回退到 n - min_tail（强制保留 3 条）
    # 预算覆盖全部 → 强制在 head 后切割（确保压缩仍执行）
```

**关键改进（2026-04-09）**：从固定消息数改为 **Token 预算动态 + 硬底线 3 条**。对长消息和短消息都更合理——超大工具输出不会独占尾部预算。

#### 边界对齐

```python
def _align_boundary_backward(messages, idx):
    """如果边界落在 tool call/result 组中间，向后拉回完整包含该组"""
```

避免拆分 assistant + tool_results 组，否则 `_sanitize_tool_pairs` 会移除尾部孤儿结果导致**静默数据丢失**。

### 3.5 Phase 3：LLM 结构化摘要

#### 序列化格式

中间轮次被序列化为带标签文本：

```
[TOOL RESULT xxx]: 内容 (截断: 前2000 + ... + 后800)
[ASSISTANT]: 内容 + [Tool calls: tool_name(args), ...]
[USER]: 内容 (截断为 3000 chars)
```

**关键设计**：序列化时包含工具调用名称和参数，使摘要器能保留具体的文件路径、命令和输出。

#### v3 摘要模板（11 段 action-log 风格）

```markdown
## Goal
[用户要完成什么]

## Constraints & Preferences
[用户偏好、编码风格、约束、重要决策]

## Completed Actions
[编号动作列表, 每条格式: N. ACTION target — outcome [tool: name]
例:
1. READ config.py:45 — found `==` should be `!=` [tool: read_file]
2. PATCH config.py:45 — changed `==` to `!=` [tool: patch]
3. TEST `pytest tests/` — 3/50 failed: test_parse, test_validate, test_edge [tool: terminal]]

## Active State
[当前工作状态: 工作目录和分支、已修改文件、测试状态、运行中进程]

## In Progress
[压缩触发时正在进行的工作]

## Blocked
[未解决的阻塞/错误, 包含完整错误消息]

## Key Decisions
[重要技术决策及其 WHY]

## Resolved Questions
[用户问过且已回答的问题 — 包含答案]

## Pending User Asks
[用户问过但尚未回答的问题]

## Relevant Files
[读取/修改/创建的文件]

## Remaining Work
[未完成的任务]

## Critical Context
[具体值、错误消息、配置细节等不能丢失的信息]
```

**v3 相比 v2 的核心改进**：

| 维度 | v2 | v3 |
|------|-----|-----|
| 完成记录 | `## Progress > ### Done` 自由文本 | `## Completed Actions` 强制编号 + 固定格式 |
| 当前状态 | 散落在 Progress 中 | 独立 `## Active State` 段落 |
| 格式示例 | 无 | 3 条 Few-shot 示例 |
| 具体度要求 | "Be specific — include file paths..." | "Be CONCRETE... **Avoid vague descriptions like 'made some changes'**" |
| 迭代更新 | "ADD new progress. Move from In Progress to Done" | "continue numbering... **Remove information only if it is clearly obsolete**" |
| 摘要预算 max_tokens | budget × 2.0 | budget × **1.3**（防膨胀） |

#### 迭代更新机制

首次压缩：从头生成完整摘要。  
后续压缩：传入 `_previous_summary`，要求**更新**而非重写：

```
PREVIOUS SUMMARY: [旧摘要]
NEW TURNS TO INCORPORATE: [新轮次]

更新摘要，保留所有仍有用的旧信息。
ADD 新的 completed actions 到编号列表（继续编号）。
把 "In Progress" 移到 "Completed Actions"（完成时）。
更新 "Active State" 反映当前状态。
仅在明显过时时才移除信息。
```

**优势**：避免每次压缩后编号重置导致信息丢失；保持摘要连续性；减少 LLM 输入量。

#### 摘要预算计算

```python
def _compute_summary_budget(turns_to_summarize):
    content_tokens = estimate_messages_tokens_rough(turns_to_summarize)
    budget = int(content_tokens * 0.20)      # 压缩到 20%
    return max(2000, min(budget, max_summary_tokens))
```

摘要预算与待压缩内容成比例，但受上下限约束（最低 2000 tokens，最高 12K tokens）。

### 3.6 Phase 4：组装与清洗

#### 摘要角色选择

为避免连续相同角色消息导致 API 错误：

```python
if last_head_role in ("assistant", "tool"):
    summary_role = "user"
else:
    summary_role = "assistant"

# 若与尾部冲突 → 尝试翻转
# 若两种角色都冲突 → 合并到第一条尾部消息中
```

#### 工具对完整性保障（`_sanitize_tool_pairs`）

| 故障模式 | 原因 | 解决方案 |
|----------|------|----------|
| 孤儿 tool result | 引用的 `call_id` 对应的 assistant `tool_call` 被移除 | 删除该孤儿结果 |
| 孤儿 tool call | assistant 有 `tool_calls` 但对应结果被丢弃 | 插入存根结果 `"[Result from earlier conversation]"` |

**重要性**：不修复会导致 OpenAI 兼容 API 返回 400 错误，整个消息列表被拒绝。

#### 最后一条用户消息保护（v0.10.0 修复 #10896）

```python
def _ensure_last_user_message_in_tail():
    """确保最后一条用户消息永远留在尾部"""
```

修复前场景：用户发送活跃任务指令后触发压缩，该指令被压入摘要区域 → Agent 丢失当前任务上下文 → 停滞或重复已完成工作。

#### Session 分裂

压缩时**拆分 session** 以保留完整原始消息供 `session_search` 日后检索：

```
压缩前:
  session "abc" (DB 中 msg 0-49 完整原始消息)
  内存中 msg 2-40 即将被压缩

压缩后:
  session "abc" (结束, reason="compression")
    → DB 中 msg 0-49 完整保留 ← session_search 可搜到原始内容
    
  session "abc-2" (新建, parent_session_id="abc")
    → 摘要 + 尾部消息 + 后续新消息
    → _last_flushed_db_idx 重置为 0
```

多次压缩形成链：`abc → abc-2 → abc-3 → ...`

**为什么不在原地替换？** 原地替换会导致 DB 中前半段是原始消息、后半段是摘要，session_search 搜到的是不一致数据。分裂保证每个 session 片段内容完整一致。

---

## 4. 核心组件详解

### 4.1 Token 预算管理器

```python
class ContextCompressor:
    def __init__(self, model, threshold_percent=0.50):
        self.context_length = get_model_context_length(model)
        self.threshold_tokens = int(self.context_length * 0.50)
        self.tail_token_budget = int(self.threshold_tokens * 0.20)
        self.max_summary_tokens = min(int(self.context_length * 0.05), 12_000)
```

**缩放设计**：尾部预算和摘要上限都与模型上下文窗口成比例，大窗口模型自动获得更丰富的摘要。

### 4.2 自适应失败冷却机制

```python
_SUMMARY_FAILURE_COOLDOWN_SECONDS = 600   # 10 分钟（配置类错误）
_TRANSIENT_COOLDOWN_SECONDS      = 60     # 1 分钟（瞬态错误）

def _generate_summary(self, turns):
    if time.monotonic() < self._summary_failure_cooldown_until:
        return None  # 冷却期内跳过

    try:
        response = call_llm(task="compression", ...)
        self._summary_failure_cooldown_until = 0.0  # 成功重置
    except RuntimeError:
        # 无 provider 配置 — 10 分钟内不会自己恢复
        self._summary_failure_cooldown_until = time.monotonic() + 600
    except Exception:
        # 瞬态错误（超时/限流/网络）— 短冷却快速重试
        self._summary_failure_cooldown_until = time.monotonic() + 60
```

**设计考量（2026-04-14）**：区分两类失败。`RuntimeError` 表示配置问题走长冷却；其他异常默认是瞬态问题走短冷却，让压缩更快从短暂故障中恢复。

### 4.3 反颠簸保护（Anti-Thrashing）

```python
def should_compress(self, prompt_tokens=None) -> bool:
    if tokens < self.threshold_tokens:
        return False
    # 连续 2 次压缩节省 <10%，跳过本次
    if self._ineffective_compression_count >= 2:
        logger.warning(
            "Compression skipped — last %d compressions saved <10%% each. "
            "Consider /new to start a fresh session, or /compress <topic> ..."
        )
        return False
    return True
```

每次压缩后计算实际节省百分比：
- `>= 10%` → 重置 `_ineffective_compression_count = 0`
- `< 10%` → `_ineffective_compression_count += 1`

**解决的问题**：某些场景下（头部 + 尾部 + 摘要本身已经很大），压缩只能挤出 1-2 条消息，每轮都触发但几乎没用，形成压缩抖动循环。连续两次无效就放弃，提示用户 `/new` 或 `/compress <topic>` 手动处理。

### 4.4 引导式压缩（`/compress <topic>`）

支持用户主动干预压缩方向：

```
用户: /compress 只保留与数据库迁移相关的内容
```

压缩器将摘要聚焦于特定主题，丢弃无关信息。这需要 CLI/交互层支持斜杠命令。

### 4.5 消息持久化机制

消息**不是实时逐条写入 DB**，而是在退出点批量 flush：

```python
def _flush_messages_to_session_db(self, messages, conversation_history):
    flush_from = max(start_idx, self._last_flushed_db_idx)
    for msg in messages[flush_from:]:
        db.append_message(session_id, role, content, ...)
    self._last_flushed_db_idx = len(messages)  # 更新水位线
```

**触发时机**（代码中 20 个调用点，覆盖所有退出路径）：

| 场景 | 保证 |
|------|------|
| 对话正常完成 | ✅ 写入 |
| API 错误 max retry 耗尽 | ✅ 放弃前写入 |
| 用户中断（Ctrl+C） | ✅ 中断前写入 |
| Rate limit 等待中被中断 | ✅ 写入 |
| 413/context overflow 压缩失败 | ✅ 写入 |
| 工具执行异常 | ✅ 写入 |
| Fallback provider 全部失败 | ✅ 写入 |

**水位线防重复**：`_last_flushed_db_idx` 记录已写入位置，多个退出路径重复调用也不会重复写入（修复 issue #860）。

---

## 5. 生产环境故障模式与边界案例

### 5.1 静默摘要丢弃（最常见）

**现象**：`_generate_summary()` 不处理 `json.JSONDecodeError`。当辅助 LLM 返回非 JSON 响应（如配置错误的端点、HTML 错误页、限流响应）时，JSON 解析静默失败。

**后果**：压缩器丢弃中间轮次**且没有生成摘要**，会话继续运行但丢失了压缩区域的所有上下文。

**日志表现**：
```
WARNING: Failed to generate context summary: Expecting value: line 631 column 1
```

### 5.2 工具排序崩溃

**触发条件**：尾部的第一条消息恰好是 `tool` 角色，插入的摘要消息排在它前面。

**后果**：OpenAI 兼容 API 要求每个 `tool` 消息必须紧跟在包含 `tool_calls` 的 `assistant` 消息之后。违反此规则导致**每轮请求返回 HTTP 400**，会话完全崩溃。

### 5.3 反颠簸永久锁定

**触发条件**：连续两次压缩节省 Token < 10%。

**后果**：`should_compress()` 永久返回 `False`，直到用户运行 `/new` 重置会话。**没有超时或衰减机制**，一旦锁定，即使后续消息大量涌入也不会自动恢复压缩。

### 5.4 辅助模型上下文不足

**触发条件**：摘要模型的上下文窗口小于主 Agent 模型。

**后果**：`_generate_summary()` 捕获上下文长度错误，记录 warning，返回 `None`。中间轮次被**无摘要丢弃**。

**要求**：摘要模型的上下文窗口必须**至少与主模型一样大**。

### 5.5 压缩导致的精确值丢失

所有基于摘要的压缩器都会系统性丢失以下五类信息：

1. **精确数值**（端口号、ID、版本号、特定配置值）
2. **硬约束**（第二轮给出的"必须使用 Python 3.11"可能在第 20 轮压缩后消失）
3. **否定性信息**（"不要修改这个文件"比"修改了那个文件"更容易被摘要遗漏）
4. **时间顺序**（摘要会重排为逻辑结构，丢失实际执行顺序）
5. **失败历史**（重试次数、曾经失败的方法通常被压缩为"最终成功"）

---

## 6. 设计演进：v1 → v2 → v3

| 改进项 | v1 | v2 | v3 (2026-04-14+) |
|--------|-----|-----|------------------|
| 摘要模板 | 无结构 | Goal/Progress/Decisions/Files/Next Steps | **编号的 Completed Actions + Active State**（action-log 风格） |
| 摘要更新 | 每次从头生成 | 迭代更新 | 迭代更新（continue numbering） |
| 尾部保护 | 固定消息数 | Token 预算按比例缩放 | 同 v2 |
| 工具输出修剪 | 无 | 通用占位符 `_PRUNED_TOOL_PLACEHOLDER` | **Smart Collapse**：按工具类型生成信息化单行摘要 |
| 去重 | 无 | 无 | **MD5 去重**：相同 tool result 只保留最新 |
| tool_call 参数 | 原样保留 | 原样保留 | **>500 字符自动截断到 200** |
| 摘要预算 max_tokens | 固定 | 按内容比例缩放 | 同 v2，但从 2× 降到 **1.3×** |
| 反颠簸 | 无 | 无 | **连续 2 次 <10% → 跳过** |
| 多模态消息 | 可能崩溃 | 可能崩溃 | dedup/prune 路径跳过 list content |
| 压缩 note 幂等 | 仅首次追加 | 同 v1 | **检测已存在则不重复追加** |
| Failure cooldown | 10 分钟固定 | 10 分钟固定 | **配置错误 10 分钟 / 瞬态错误 60 秒** |
| 工具调用完整性 | 可能丢失 | `_sanitize_tool_pairs` 修复孤儿对 | 同 v2 |
| 最后用户消息保护 | 无 | 无 | **v0.10.0 新增 `_ensure_last_user_message_in_tail`** |

**v3 的核心设计思路**：v2 的模板留给 LLM 太多自由度，产出质量不稳定；v3 通过强制编号、具体示例、显式禁令把"怎么写摘要"从开放式变成填空式，压缩产出更可预测、信息密度更高。

---

## 7. 与 Claude Code (OpenClaw) 压缩机制对比

OpenClaw 的压缩实现位于 `src/agents/compaction.ts`，采用**分块摘要**策略，与 Hermes 的**三阶段预处理 + 单次摘要**形成鲜明对比。

### 7.1 整体架构差异

| 维度 | Hermes v3 | OpenClaw |
|------|-----------|----------|
| 整体策略 | 本地预处理 → 边界划分 → 单次 LLM 摘要 | 分块 + 多次 LLM 摘要 |
| LLM 调用次数 | **1 次**（只对瘦身后的中间部分） | **多次**（滚动式 N 次，或并行式 N+1 次） |
| 预处理 | MD5 去重 + Smart Collapse + 参数截断（零 Token） | `stripToolResultDetails()` 去掉工具详情（轻量） |
| 分块 | 不分块，头-中-尾三段 | 两种分块策略 |

OpenClaw 两条压缩路径：
- **`summarizeChunks`（滚动式）**：按 token 上限切块，串行处理——chunk1 摘要作为 chunk2 的 `previousSummary` 传入，逐步滚动。LLM 调用 N 次。
- **`summarizeInStages`（并行+合并）**：切 N 块（默认 2 块），每块独立摘要，最后用 `MERGE_SUMMARIES_INSTRUCTIONS` 合并。LLM 调用 N+1 次。

### 7.2 摘要模板对比

**Hermes v3（11 段）**：
```
Goal / Constraints & Preferences / Completed Actions（编号+格式）/
Active State / In Progress / Blocked / Key Decisions /
Resolved Questions / Pending User Asks / Relevant Files /
Remaining Work / Critical Context
```

**OpenClaw（5 段）**：
```
Decisions / Open TODOs / Constraints/Rules /
Pending user asks / Exact identifiers
```

Hermes 模板更细致（Active State、Relevant Files、编号的 Completed Actions），OpenClaw 更精炼，但有 `Exact identifiers` 段显式要求保留 IDs/URLs/哈希/端口等字面值。

### 7.3 逐项对比

| 维度 | Hermes v3 | OpenClaw |
|------|-----------|----------|
| 操作记录 | `Completed Actions` 编号列表 `N. ACTION target — outcome [tool: name]` | 无专门段落，融入 Decisions |
| 运行时状态 | `Active State`（分支、测试状态、运行进程） | 无 |
| 精确值保留 | `Critical Context` 段 | `Exact identifiers` 段 |
| 未答问题追踪 | `Pending User Asks` + `Resolved Questions` | `Pending user asks`（只跟踪未答） |
| 文件追踪 | `Relevant Files` 独立段落 | 无，靠 Exact identifiers 保留路径 |
| 质量校验 | 无（信任 LLM 输出） | `auditSummaryQuality()` 检查 5 段落，不通过重试，兜底生成骨架 |
| 迭代更新 | "continue numbering" 接续旧摘要编号 | `previousSummary` 传入下一块 |
| 摘要上限 | 压缩内容 × 0.2，上限 12K tokens | 硬性 16,000 字符 |
| 防抖 | 连续 2 次 <10% → 跳过 | 6 种 skip reason 分类 |
| 失败处理 | RuntimeError 600s / 瞬态 60s 冷却 | 15 分钟安全超时 + 3 次重试 + 结构化兜底 |
| 工具对修复 | `_sanitize_tool_pairs()` 补孤儿对 | `repairToolUseResultPairing()` 删孤儿 |
| 尾部保护 | token 预算动态 + 硬底线 3 条 | `DEFAULT_RECENT_TURNS_PRESERVE=3`（上限 12） |

### 7.4 各有所长

**Hermes 优势**：
- 本地预处理（MD5 去重 + Smart Collapse）在 LLM 之前砍掉 30-50% token
- 单次 LLM 调用，不管对话多长只调 1 次
- 模板更细致（11 段 vs 5 段），续接 agent 拿到的上下文更丰富

**OpenClaw 优势**：
- 质量校验闭环（审查 → 重试 → 兜底骨架）
- 分块策略天然适应超长对话（单次 LLM 输入窗口有限，分块避免溢出）
- `Exact identifiers` 段显式保留关键字面值
- 跳过原因分类更细（6 种 reason），有助于调试

---

## 8. 对 agent-material 项目的借鉴建议

### 8.1 短期可落地（1-3 天）

#### 改进 1：结构化摘要模板

当前 `memory_config.yaml` 中的压缩提示词为自由文本：

```yaml
prompt_template: |
  你是一个对话压缩助手。请将以下对话历史压缩成简洁的摘要...
```

建议改为 Hermes v3 风格的结构化模板（选取适合材料科学场景的 5-6 个章节）：

```yaml
prompt_template: |
  请对以下对话历史进行结构化摘要，供后续 Agent 接续对话使用：

  ## 当前任务
  [用户正在解决的材料科学问题]

  ## 已完成操作
  [编号动作列表，格式: N. ACTION — outcome [tool: name]]

  ## 关键决策
  [重要技术决策及理由]

  ## 工具调用结果
  [关键工具返回的数据、文献、计算结果]

  ## 待解决问题
  [尚未回答的用户问题或阻塞事项]

  ## 相关文献/数据
  [检索到的关键文献或 Materials Project 结构 ID]

  对话历史：
  {conversation}

  摘要（用中文输出，严格按上述章节组织）：
```

#### 改进 2：工具输出预处理

在 `ContextCompressor.compress()` 的 Phase 1 增加零成本预处理：

```python
def _preprocess_tool_results(self, messages):
    # 1. 去重相同 tool result（按内容 hash）
    # 2. 大型工具输出替换为信息化单行摘要
    #    - MP 查询: "[mp_query] found 12 structures for Ag-Au (total 3.4KB)"
    #    - RAG 检索: "[rag_retrieve] query='deep potential' -> 5 docs, top score 0.87"
    #    - 文件读取: "[read_file] read config.py (1,200 chars)"
    # 3. 截断过长的 tool_call 参数
```

#### 改进 3：Token 预算尾部保护

当前 `memory_config.yaml`：
```yaml
short_term:
  protect_last_n: 4     # 固定消息数
```

建议改为 Token 预算：

```python
def _find_tail_cut_by_tokens(messages, head_end, token_budget):
    min_tail = 3
    soft_ceiling = int(token_budget * 1.5)
    # 从末尾向前累加，直到超过 soft_ceiling 且满足 min_tail
```

### 8.2 中期可落地（1-2 周）

#### 改进 4：迭代摘要更新

当前实现每次压缩都从头生成摘要。改为保留上轮摘要，与新消息合并更新：

```python
class ContextCompressor:
    def __init__(self, ...):
        self._previous_summary: str | None = None
    
    def compress(self, messages, ...):
        if self._previous_summary:
            # 传入旧摘要，要求更新而非重写
            prompt = UPDATE_SUMMARY_PROMPT.format(
                previous=self._previous_summary,
                new_turns=new_turns_text
            )
        else:
            prompt = FIRST_SUMMARY_PROMPT.format(conversation=conversation_text)
        
        summary = self.model.invoke(prompt)
        self._previous_summary = summary
```

#### 改进 5：压缩效果追踪与反颠簸

```python
class ContextCompressor:
    def __init__(self, ...):
        self._ineffective_count: int = 0
    
    def should_compress(self, current_tokens) -> bool:
        if current_tokens < self.threshold:
            return False
        if self._ineffective_count >= 2:
            logger.warning("Compression skipped: last 2 compressions saved <10%")
            return False
        return True
    
    def compress(self, ...):
        # ... 执行压缩 ...
        saved_percent = (original - new) / original
        if saved_percent < 0.10:
            self._ineffective_count += 1
        else:
            self._ineffective_count = 0
```

#### 改进 6：自适应失败冷却

```python
def _generate_summary(self, turns):
    if time.monotonic() < self._cooldown_until:
        return None
    
    try:
        return self._call_summary_llm(turns)
    except ConfigurationError:
        self._cooldown_until = time.monotonic() + 600  # 10 分钟
    except (Timeout, RateLimitError):
        self._cooldown_until = time.monotonic() + 60   # 1 分钟
```

### 8.3 长期可探索

#### 改进 7：Context Engine 可插拔化

参考 Hermes 的 `ContextEngine` ABC，将当前项目中硬编码的压缩逻辑抽离为接口：

```python
from abc import ABC, abstractmethod

class ContextEngine(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...
    
    @abstractmethod
    def should_compress(self, prompt_tokens: int) -> bool: ...
    
    @abstractmethod
    def compress(self, messages: list[MessageRecord], current_tokens: int) -> ContextWindow: ...
```

这样未来可以：
- 替换为基于向量检索的上下文选择器
- 接入 LCM 这类无损 DAG 引擎
- 让模型通过工具调用自主决定压缩时机和范围

#### 改进 8：Session 分裂与历史检索

当前项目压缩时直接在内存中替换消息，原始消息丢失。可借鉴 Hermes 的 Session 分裂：

```
压缩前: session "abc" (msg 0-49)
压缩后: 
  - session "abc" 结束 (reason="compression")，DB 保留完整原始消息
  - session "abc-2" 新建，内存中只有摘要 + 尾部
```

配合 `session_search` 工具，Agent 可以检索被压缩的历史原始内容。

---

## 附录：关键文件索引

| 文件路径（Hermes 源码） | 职责 |
|------------------------|------|
| `agent/context_engine.py` | ContextEngine ABC 定义 |
| `agent/context_compressor.py` | ContextCompressor 默认实现（四阶段算法） |
| `agent/prompt_caching.py` | Anthropic Prompt Caching 策略 |
| `gateway/run.py` | Gateway Session Hygiene（85% 安全网） |
| `run_agent.py` | Agent 主循环，搜索 `_compress_context` |
| `plugins/context_engine/__init__.py` | 引擎插件注册基础设施 |

---

*报告完成。本报告基于 Hermes Agent 公开源码、官方文档、Wiki 及第三方生产环境分析独立整理，未引用项目内已有的 `hermes-agent-prompt-analysis.md` 文件。*
