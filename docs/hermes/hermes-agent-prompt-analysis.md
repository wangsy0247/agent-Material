# Hermes Agent 提示词系统深度分析报告

> 分析对象：`hermes-agent` 源码中的 Prompt / Context / Compression / Caching 体系  
> 分析日期：2026-05-20  
> 报告位置：`knowledge/hermes-agent-prompt-analysis.md`

---

## 1. 总体架构概览

Hermes Agent 的提示词系统是一个**分层、模块化、模型无关**的流水线，核心设计哲学：

> **"OpenAI Message Format 作为内部通用语（Lingua Franca），Provider 转换只在 API 边界发生"**

```
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 6: Provider Transport（API 边界转换）                         │
│  Anthropic / Bedrock / ChatCompletions / Codex                      │
│  ── convert_messages() / convert_tools() / build_kwargs()          │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 5: Prompt Caching（Anthropic 专属缓存标记注入）               │
│  ── apply_anthropic_cache_control() → system + 最近 3 条消息        │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 4: Context Compression（上下文压缩）                          │
│  ── ContextCompressor: 保护头部 + 保护尾部 + LLM 摘要中部           │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 3: Context Engine（可插拔接口）                               │
│  ── update_from_response() / should_compress() / compress()        │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 2: Prompt Builder（系统提示组装）                             │
│  ── Identity + Model Guidance + Platform + Environment + Skills    │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 1: Context Files（项目级上下文文件）                          │
│  ── .hermes.md / AGENTS.md / .cursorrules / SOUL.md               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 核心模块详解

### 2.1 Prompt Builder (`agent/prompt_builder.py`, 1456 行)

这是**系统提示的源头**，`AIAgent._build_system_prompt()` 调用这里的函数组装完整的系统提示。

#### 2.1.1 模块化系统提示组件

| 组件 | 内容 | 说明 |
|------|------|------|
| `DEFAULT_AGENT_IDENTITY` | 基础身份定义 | "You are Hermes, an expert software engineer..." |
| `MEMORY_GUIDANCE` | 记忆工具使用指南 | 如何读取/写入长期记忆 |
| `SESSION_SEARCH_GUIDANCE` | 会话搜索指南 | 如何利用历史会话上下文 |
| `SKILLS_GUIDANCE` | Skill 使用指南 | 何时调用 Skill、参数规范 |
| `KANBAN_GUIDANCE` | 看板任务管理 | TODO 状态流转规则 |
| `TOOL_USE_ENFORCEMENT_GUIDANCE` | 工具使用强制约束 | 对特定模型的强制工具调用指令 |

#### 2.1.2 模型专属行为注入

根据模型名称**动态注入**不同的行为指导：

**OpenAI 模型**：
- `<tool_persistence>` — 工具调用后保持状态
- `<mandatory_tool_use>` — 强制使用工具而非纯文本回答
- `<act_dont_ask>` — 先执行再问
- `<prerequisite_checks>` — 执行前检查前置条件
- `<verification>` — 执行后验证结果

**Google 模型**：
- 强制使用绝对路径
- `verify first` — 先验证再操作
- 鼓励并行工具调用

#### 2.1.3 平台感知格式化 (`PLATFORM_HINTS`)

支持 **16+ 平台**（WhatsApp、Telegram、Discord、Slack、CLI、SMS、WebUI、飞书、企业微信、元宝等）：
- 控制是否支持 Markdown
- `MEDIA:` 语法（图片/文件交付方式）
- 消息长度限制
- 代码块渲染规则

#### 2.1.4 环境感知 (`build_environment_hints()`)

自动探测运行环境并注入描述：
- WSL / 纯 Linux / macOS / Windows
- 远程后端（Docker、Modal、SSH）
- Windows Bash Shell
- 实时后端状态探测

#### 2.1.5 上下文文件加载与注入扫描

自动发现项目中的上下文文件（从当前目录向上遍历到 git root）：
- `.hermes.md` — 项目级自定义提示
- `AGENTS.md` — Agent 使用指南
- `CLAUDE.md` — Claude 专属上下文
- `.cursorrules` — Cursor IDE 规则
- `SOUL.md` — 高级人格定义

**安全机制 — Prompt Injection 扫描** (`_scan_context_content`)：
- 检测不可见 Unicode 字符（零宽空格、方向覆盖符等）
- 检测欺骗性模式（如 "ignore previous instructions"、"system prompt override"）
- 检测数据外泄模式（如要求发送敏感信息到外部 URL）
- 每份上下文文件加载前必须经过扫描，**未通过则拒绝注入**

#### 2.1.6 Skills 系统提示 (`build_skills_system_prompt()`)

**双层缓存机制**：
1. **内存 LRU 缓存**：最多 8 条（`_SKILLS_PROMPT_CACHE_MAX = 8`），带线程锁
2. **磁盘快照**：`.skills_prompt_snapshot.json`，基于 mtime/size 的 manifest 校验

**条件激活**：Skills 可声明 `requires_tools`、`fallback_for_toolsets`，Prompt Builder 只把与当前可用工具/平台兼容的 Skills 注入提示词。

#### 2.1.7 上下文文件截断

单文件 **20K 字符上限**，采用 **70/20 头尾比例**：
- 头部保留 70%（约 14K）
- 尾部保留 20%（约 4K）
- 中间 10% 用 `[...truncated...]` 替代

---

### 2.2 Prompt Caching (`agent/prompt_caching.py`, 79 行)

**目标 Provider**：仅 Anthropic（其他 Provider 不支持显式 prompt caching）

**策略：`system_and_3`**

最多设置 **4 个 cache_control 断点**：
1. System prompt（如果存在）
2. 最近 3 条非 system 消息（用户/助手/工具）

每个断点共享相同的 TTL：
- 默认 `5m`
- 可选 `1h`

**效果**：多轮对话中，输入 Token 成本降低约 **75%**

**实现**：
```python
def apply_anthropic_cache_control(messages, system, ttl="5m"):
    # 深度拷贝消息列表，在最后一个 content block 注入 {"type": "ephemeral"}
    # 不修改内部消息表示，仅在 Anthropic SDK 调用前最后一刻应用
```

---

### 2.3 Context Engine (`agent/context_engine.py`, 211 行)

**可插拔接口（Abstract Base）**，定义上下文管理合约：

| 方法 | 职责 |
|------|------|
| `update_from_response(usage)` | 每轮 API 调用后追踪 Token 用量 |
| `should_compress(prompt_tokens)` | 判断是否需要压缩（默认阈值 75% 上下文窗口） |
| `compress(messages, ...)` | 压缩消息列表，返回合法 OpenAI 格式序列 |
| `on_session_start/end/reset()` | 会话生命周期钩子 |
| `get_tool_schemas()` | 可选：向 Agent 暴露引擎专用工具（如 `lcm_grep`） |
| `update_model()` | 用户切换模型时重新校准参数 |

**默认实现**：`ContextCompressor`

---

### 2.4 Context Compressor (`agent/context_compressor.py`, 1575 行)

这是整个提示词系统中最复杂的模块，使用**辅助 LLM（便宜/快速模型）**进行有损摘要。

#### 2.4.1 五阶段压缩算法

**Phase 1: 廉价预处理（无 LLM）**
- 大型工具输出替换为单行摘要（如 `[terminal] ran \`npm test\` -> exit 0, 47 lines output`）
- 去重相同工具结果
- 剥离旧 Computer-Use 截图
- 截断大型 tool_call 参数但保持 JSON 合法性

**Phase 2: 保护头部（Head Protection）**
- System prompt
- 前 N 条非 system 消息（默认 3 条）

**Phase 3: 按 Token 预算保护尾部（Tail Protection）**
- 不是固定消息数，而是**按 Token 数量**保护最近消息
- 保护量 = 上下文窗口的 ~20%，且有硬下限 3 条消息
- 使用 `_find_tail_cut_by_tokens()` 精确计算切割点
- **自适应**：32K → 1M+ 上下文模型自动缩放

**Phase 4: 中部摘要（Middle Summarization）**

使用结构化模板让辅助 LLM 生成摘要：

```
## Active Task
## Goal
## Constraints
## Completed Actions
## Active State
## In Progress
## Blocked
## Key Decisions
## Resolved/Pending Questions
## Relevant Files
## Remaining Work
## Critical Context
```

**迭代更新**：保留上一轮摘要，与新消息合并更新，而非从头重新生成。

**引导式压缩**：支持 `/compress <focus_topic>`，让摘要聚焦于特定主题。

**数据脱敏**：摘要前对内容进行 redact 处理，防止秘密泄露给辅助模型。

**Phase 5: 组装与清洗**
- 清理孤儿 tool_call / tool_result 对（防止 API 400）
- 确保最后一条用户消息永远不会被压缩掉（修复 #10896）
- 避免连续相同角色的消息（通过合并或角色选择）
- 追加 `--- END OF CONTEXT SUMMARY ---` 显式标记

#### 2.4.2 抗抖动与容错

| 机制 | 说明 |
|------|------|
| **压缩效果追踪** | 若最近 2 次压缩节省 <10%，则跳过本次压缩 |
| **失败冷却** | 根据错误类型，30s ~ 600s 冷却期 |
| **模型降级** | 摘要模型失效时，自动降级到主模型进行摘要 |
| **静态兜底** | 若全部失败，插入静态标记 `[Context compressed — previous turns summarized]` |

---

### 2.5 Transport 层 (`agent/transports/`)

**设计原则**：Transport 只负责**格式转换**，不负责客户端构造、流式处理、凭证刷新、Prompt Caching、中断处理或重试逻辑。

**核心接口**（Abstract Base）：

| 方法 | 职责 |
|------|------|
| `convert_messages(messages)` | OpenAI 格式 → Provider 原生格式 |
| `convert_tools(tools)` | OpenAI Tool Schema → Provider 原生格式 |
| `build_kwargs(model, messages, tools, **params)` | 组装完整 API 调用参数 |
| `normalize_response(response)` | Provider 响应 → 统一 `NormalizedResponse` |
| `extract_cache_stats(response)` | Provider 专属缓存命中/创建 Token 提取 |
| `map_finish_reason(raw)` | Provider 停止原因 → OpenAI 等价物 |

**支持的 Provider**：

| api_mode | Transport | 关键适配点 |
|----------|-----------|-----------|
| `chat_completions` | `ChatCompletionsTransport` | ~16 个 OpenAI 兼容 Provider；Gemini thinkingConfig、Qwen、OpenRouter 专属参数 |
| `anthropic_messages` | `AnthropicTransport` | 完整消息/工具/推理块转换；Thinking 签名管理；Beta Headers；OAuth |
| `bedrock_converse` | `BedrockTransport` | boto3 Converse API 格式；Guardrails |
| `codex_responses` | `ResponsesApiTransport` | OpenAI Codex Responses API |

---

### 2.6 Anthropic 适配器 (`agent/anthropic_adapter.py`)

最复杂的 Provider 适配器，处理 Anthropic 独有的特性：

#### 2.6.1 消息转换 (`convert_messages_to_anthropic`)

- 提取 system prompt 为独立参数
- OpenAI `tool_calls` / `tool` → Anthropic `tool_use` / `tool_result` blocks
- 多模态内容转换（图片 URL、base64 data URL）
- **强制角色交替**：合并连续相同角色的消息

#### 2.6.2 Thinking Block 签名管理

Anthropic 对 thinking blocks 有**内容签名验证**：

| 场景 | 处理方式 |
|------|---------|
| 原生 Anthropic 端点 | 仅保留最新 assistant 消息的签名，其余 strip |
| 第三方端点（Kimi/DeepSeek 等） | 全部 strip 签名 |
| Kimi/DeepSeek 兼容 | 保留 unsigned blocks |
| 其他第三方 | unsigned blocks 降级为纯文本 |

#### 2.6.3 图片驱逐

Tool result 中**仅保留最近 3 张截图**，防止视觉上下文无限膨胀。

#### 2.6.4 工具 Schema 规范化

- 剥离 Anthropic 拒绝的 nullable union（`anyOf: [string, null]`）
- 移除顶层 `oneOf` / `allOf` / `anyOf`
- 防御性去重工具名

#### 2.6.5 模型专属参数解析

- `max_tokens` 按模型族解析（Claude 3/3.5/4/4.5/4.6/4.7，输出限制 4K ~ 128K）
- Claude 4.6+ 支持自适应 thinking；旧版支持手动 thinking budget
- 4.7+ 模型完全 strip `temperature` / `top_p` / `top_k`（API 拒绝非空值）
- `xhigh` effort 对 4.7 前模型降级
- Fast mode (`speed: "fast"`) 仅对 Opus 4.6 原生端点生效

---

## 3. 独特特性总结

| 特性 | 说明 | 借鉴价值 |
|------|------|---------|
| **Prompt Injection 扫描** | 加载 `.hermes.md` / `AGENTS.md` 前扫描不可见 Unicode、欺骗模式、数据外泄 | ⭐⭐⭐⭐⭐ 高：可直接在 AGENTS.md 加载前增加扫描 |
| **模型专属行为注入** | OpenAI / Google / Anthropic 分别注入不同的行为指导块 | ⭐⭐⭐⭐⭐ 高：我们项目目前只有一套 system prompt，可按模型拆分 |
| **平台感知格式化** | 16+ 平台控制 Markdown、MEDIA 语法、长度限制 | ⭐⭐⭐ 中：我们项目前端是 Web，暂不需要 |
| **双层 Skills 缓存** | 内存 LRU (8条) + 磁盘快照 (mtime/size 校验) | ⭐⭐⭐⭐ 高：RAG prompt 或 Skill 提示可借鉴 |
| **Token 预算尾部保护** | 不是固定消息数，而是按 Token 比例保护尾部 | ⭐⭐⭐⭐⭐ 高：我们的短期记忆压缩可改用 Token 预算而非固定消息数 |
| **结构化 LLM 摘要模板** | 12 个固定章节（Active Task、Blocked、Key Decisions 等） | ⭐⭐⭐⭐ 高：我们的 context_compressor 可改用结构化模板 |
| **迭代摘要更新** | 保留上一轮摘要，与新消息合并更新 | ⭐⭐⭐⭐ 高：避免每次从头生成摘要 |
| **引导式压缩** | `/compress <topic>` 聚焦特定主题 | ⭐⭐⭐ 中：需要 CLI 交互支持 |
| **数据脱敏** | 摘要前 redact 敏感内容 | ⭐⭐⭐⭐ 高：我们的记忆提取也应增加脱敏 |
| **Thinking Block 签名管理** | Anthropic 特有的推理块签名验证 | ⭐⭐ 低：我们目前不用 Claude 的 reasoning 特性 |
| **工具对完整性校验** | 压缩后清理孤儿 tool_call / tool_result | ⭐⭐⭐⭐ 高：LangGraph 中工具消息完整性检查 |
| **角色交替强制** | Anthropic 要求严格 user/assistant 交替，合并连续同角色消息 | ⭐⭐⭐ 中：OpenAI 不强制，但有助于减少消息数 |

---

## 4. 与我们项目的对比与借鉴

### 4.1 当前差距

| 维度 | Hermes Agent | 我们的项目 (agent-material) |
|------|------------|---------------------------|
| **系统提示组装** | 模块化、多来源（Identity + Model + Platform + Environment + Skills + Context Files） | 单一 `system_prompt.txt` 文件 |
| **模型适配** | 按模型名称动态注入不同行为指导 | 无模型专属适配 |
| **上下文压缩** | 5 阶段结构化压缩，辅助 LLM 摘要 | 仅保留头尾消息 + 简单截断 |
| **Prompt Caching** | Anthropic `system_and_3` 策略，~75% 成本节省 | 不支持 |
| **Prompt Injection 扫描** | 加载任何外部文件前强制扫描 | 无 |
| **Skills 缓存** | LRU + 磁盘快照双层缓存 | 无 Skills 体系 |
| **Token 预算尾部保护** | 按 Token 比例自适应保护尾部 | 固定消息数保护 |
| **环境感知** | 自动探测 WSL/Docker/SSH/Windows | 无 |

### 4.2 可直接借鉴的改进

#### 4.2.1 短期：Prompt Injection 扫描（1 天可落地）

在 `utils/file_handler.py` 或 `utils/prompt_loader.py` 中增加扫描函数：

```python
import re
import unicodedata

SUSPICIOUS_PATTERNS = [
    re.compile(r"ignore\s+(previous|all)\s+instructions", re.I),
    re.compile(r"system\s+prompt\s+(override|leak|expose)", re.I),
    re.compile(r"send\s+(this|the)\s+to\s+\w+@\w+\.\w+", re.I),
]

def scan_prompt_injection(text: str) -> list[str]:
    """扫描文本中的 prompt injection 迹象。"""
    findings = []
    # 不可见 Unicode
    for char in text:
        if unicodedata.category(char) in ("Cf", "Cn", "Co"):
            findings.append(f"不可见字符: U+{ord(char):04X}")
    # 欺骗模式
    for pattern in SUSPICIOUS_PATTERNS:
        if pattern.search(text):
            findings.append(f"可疑模式: {pattern.pattern}")
    return findings
```

在加载 `AGENTS.md`、`prompt/system_prompt.txt` 等外部文件前调用。

#### 4.2.2 短期：模型专属行为注入（2 天可落地）

在 `model/model_factory.py` 中识别模型名称，返回对应的模型族标识：

```python
def get_model_guidance(model_name: str) -> str:
    if "deepseek" in model_name.lower():
        return "...DeepSeek 专属指导..."
    if "qwen" in model_name.lower():
        return "...Qwen 专属指导..."
    return ""
```

然后在 `agent/graph/agent_graph.py` 的 `memory_node` 中，将模型指导拼接到 system prompt 中。

#### 4.2.3 中期：上下文压缩结构化模板（3-5 天）

我们的 `memory/context_compressor.py` 当前使用的是简单截断。可借鉴 Hermes 的 12 章节结构化模板，改为：

```python
COMPRESSION_TEMPLATE = """
请对以下对话历史进行摘要，保留关键信息：

## 当前任务
## 已完成操作
## 关键决策
## 待解决问题
## 相关文件/数据

对话历史：
{conversation}

摘要：
"""
```

#### 4.2.4 中期：Token 预算尾部保护（2-3 天）

在 `memory/short_term.py` 中，将当前的固定消息数保护改为 Token 预算保护：

```python
# 当前：固定保留最近 N 条消息
# 改进：按 Token 比例保留
protected_token_budget = int(max_tokens * 0.2)  # 保留 20% Token 预算给尾部
```

#### 4.2.5 长期：Prompt Caching（依赖模型支持）

如果未来切换到 Anthropic Claude 或支持 prompt caching 的模型，可在 `model/model_factory.py` 输出层增加 `apply_anthropic_cache_control()` 类似的逻辑。

#### 4.2.6 长期：双层 Skills 缓存

如果未来引入 Skill 体系，可借鉴 LRU + 磁盘快照的缓存策略。

---

## 5. 关键文件索引

| 文件路径 | 职责 |
|---------|------|
| `agent/prompt_builder.py` | 系统提示组装：Identity、Model Guidance、Platform、Environment、Skills、Context Files |
| `agent/prompt_caching.py` | Anthropic Prompt Caching：`system_and_3` 策略 |
| `agent/context_engine.py` | 可插拔上下文引擎接口 |
| `agent/context_compressor.py` | 默认上下文压缩器：5 阶段算法 |
| `agent/transports/base.py` | Transport 抽象基类 |
| `agent/transports/anthropic.py` | Anthropic Transport 包装器 |
| `agent/anthropic_adapter.py` | Anthropic 消息/工具/Thinking 块转换 |
| `agent/transports/chat_completions.py` | OpenAI 兼容 Provider 传输 |
| `agent/transports/bedrock.py` | AWS Bedrock Converse API 传输 |

---

*报告完成。如需针对某个具体模块（如 Context Compressor 的 12 章节模板、Prompt Injection 扫描器）进行更深入的源码级分析或落地实现，请告知。*
