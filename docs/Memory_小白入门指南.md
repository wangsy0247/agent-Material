# 企业落地 AI 记忆力机制小白入门指南（LangChain/LangGraph 实战版）

> **目标读者**：正在构建 AI Agent 或对话系统、希望让 AI"记住"事情的企业开发者。
>
> **学习目标**：理解 AI 记忆的分层架构（短期/长期/缓存），掌握各种记忆压缩策略，能在 LangChain/LangGraph 中实现生产级记忆管理，并了解主流产品的记忆机制差异。

---

## 目录

- [企业落地 AI 记忆力机制小白入门指南（LangChain/LangGraph 实战版）](#企业落地-ai-记忆力机制小白入门指南langchainlanggraph-实战版)
  - [目录](#目录)
  - [一、什么是 AI 的记忆？一句话理解](#一什么是-ai-的记忆一句话理解)
  - [二、短期记忆与上下文窗口](#二短期记忆与上下文窗口)
    - [2.1 短期记忆是什么？](#21-短期记忆是什么)
    - [2.2 上下文窗口（Context Window）](#22-上下文窗口context-window)
    - [2.3 在 LangChain/LangGraph 中管理短期记忆](#23-在-langchainlanggraph-中管理短期记忆)
  - [三、分层压缩策略 ⭐](#三分层压缩策略-)
    - [3.1 为什么需要压缩？](#31-为什么需要压缩)
    - [3.2 常见压缩策略详解](#32-常见压缩策略详解)
    - [3.3 压缩参数如何调整？](#33-压缩参数如何调整)
  - [四、长期记忆](#四长期记忆)
    - [4.1 长期记忆是什么？](#41-长期记忆是什么)
    - [4.2 长期记忆的触发机制](#42-长期记忆的触发机制)
    - [4.3 长期记忆的作用](#43-长期记忆的作用)
    - [4.4 在 LangChain 中实现长期记忆](#44-在-langchain-中实现长期记忆)
    - [4.5 向量库作为长期记忆](#45-向量库作为长期记忆)
  - [五、会话窗口](#五会话窗口)
    - [5.1 会话窗口是什么？](#51-会话窗口是什么)
    - [5.2 滑动窗口 vs 固定窗口](#52-滑动窗口-vs-固定窗口)
    - [5.3 会话隔离与共享](#53-会话隔离与共享)
  - [六、缓存机制](#六缓存机制)
    - [6.1 为什么需要缓存？](#61-为什么需要缓存)
    - [6.2 缓存类型详解](#62-缓存类型详解)
    - [6.3 LangChain 缓存实现](#63-langchain-缓存实现)
  - [七、参数调整](#七参数调整)
    - [7.1 记忆相关参数汇总](#71-记忆相关参数汇总)
    - [7.2 各参数的影响和调优建议](#72-各参数的影响和调优建议)
  - [八、主流产品记忆机制对比](#八主流产品记忆机制对比)
    - [8.1 Claude Code](#81-claude-code)
    - [8.2 ChatGPT](#82-chatgpt)
    - [8.3 Hermes](#83-hermes)
    - [8.4 开源记忆框架（以 OpenClaw 为例）](#84-开源记忆框架以-openclaw-为例)
    - [8.5 四者对比总结](#85-四者对比总结)
  - [九、Agent 自我进化](#九agent-自我进化)
    - [9.1 什么是自我进化？](#91-什么是自我进化)
    - [9.2 实现方案](#92-实现方案)
    - [9.3 LangGraph 中的自我进化实现](#93-langgraph-中的自我进化实现)
  - [十、完整实战代码](#十完整实战代码)
  - [附录：记忆机制全景图](#附录记忆机制全景图)

---

## 一、什么是 AI 的记忆？一句话理解

**AI 的记忆 = 让模型"记得"之前发生过什么。**

大语言模型（LLM）本身是无状态的——每次你发消息，它都是"从头开始"处理，不会自动记住你们上一轮聊了什么。记忆机制就是解决这个问题的一套工程方案。

```
没有记忆：
  用户: "我叫张三"
  AI:   "你好张三，有什么可以帮你的？"
  用户: "我叫什么名字？"
  AI:   "抱歉，我不知道你的名字。"  ← 😅 刚说的就忘了

有记忆：
  用户: "我叫张三"
  AI:   "你好张三，有什么可以帮你的？"
  用户: "我叫什么名字？"
  AI:   "你叫张三。"  ← ✅ 记住了
```

**记忆的层次架构**：

```
┌─────────────────────────────────────────────────────────────┐
│                     AI 记忆分层架构                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  超高速层（纳秒级）                                          │
│  ├── 模型内部 Attention 缓存（KV Cache）                     │
│  └── 单次推理中的中间状态                                     │
│                                                             │
│  高速层（毫秒级） ──► 短期记忆 / 上下文窗口                   │
│  ├── 当前对话历史（最近 N 轮）                                │
│  ├── 系统 Prompt / 角色设定                                   │
│  └── 压缩后的摘要（Summary）                                  │
│                                                             │
│  中速层（百毫秒级） ──► 缓存                                 │
│  ├── LLM 响应缓存（相同问题直接返回）                         │
│  ├── Embedding 缓存（避免重复计算）                           │
│  └── 检索结果缓存（向量搜索加速）                             │
│                                                             │
│  低速层（秒级） ──► 长期记忆 / 持久化存储                     │
│  ├── 向量数据库（语义记忆）                                   │
│  ├── 关系数据库（结构化记忆：用户画像、偏好）                  │
│  ├── 键值存储（简单事实缓存）                                 │
│  └── 文件/对象存储（文档、历史记录）                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

> **核心原则**：越快的记忆层容量越小、越贵；越慢的记忆层容量越大、越便宜。好的记忆系统设计 = **让"热数据"留在高速层，"冷数据"沉入低速层。**

---

## 二、短期记忆与上下文窗口

### 2.1 短期记忆是什么？

短期记忆 = **当前对话中，模型能直接"看到"的所有信息。**

当你和 AI 聊天时，每次请求发给模型的不只是你最新的一条消息，而是**整个对话历史**（加上系统指令）。这些被一起塞进模型输入里的内容，就是短期记忆。

```
实际发给模型的输入：

[系统指令] 你是一个 helpful assistant.

[用户] 你好
[AI]   你好！有什么可以帮你的？

[用户] 我想订一张去北京的机票
[AI]   好的，请问您想什么时候出发？

[用户] 明天  ← 这是用户最新发的消息

─────────────────────────────────────
这整个东西 = 短期记忆 = 上下文（Context）
```

**短期记忆的特点**：

| 特点 | 说明 |
|------|------|
| **速度快** | 直接放在 Prompt 里，模型一眼就能看到 |
| **容量有限** | 受模型上下文窗口限制（4K~200K tokens） |
| **生命周期短** | 对话结束就没了（除非持久化） |
| **精度高** | 原始信息无压缩，细节完整 |

### 2.2 上下文窗口（Context Window）

上下文窗口 = **模型一次能处理的 Token 上限**。它是短期记忆的"物理天花板"。

| 模型 | 上下文窗口 | 大约能容纳 |
|------|-----------|-----------|
| GPT-3.5 | 16K | 约 12,000 汉字 |
| GPT-4o | 128K | 约 96,000 汉字 |
| GPT-4o-mini | 128K | 约 96,000 汉字 |
| Claude 3.5 Sonnet | 200K | 约 150,000 汉字 |
| Claude 3 Opus | 200K | 约 150,000 汉字 |
| Llama 3 | 8K~128K（依版本） | 依配置 |
| Qwen2.5 | 128K | 约 96,000 汉字 |

**上下文窗口的分配**：

```
总上下文窗口（如 128K）
├─ System Prompt（系统指令）     占 500~2000 tokens
├─ 历史对话（多轮消息）           占 20K~100K tokens（主要部分）
├─ 检索文档（RAG 上下文）         占 5K~20K tokens
├─ 工具调用结果                  占 1K~5K tokens
├─ 用户最新输入                  占 100~1000 tokens
└─ 输出预留                      占 2K~10K tokens
```

**常见问题**：

- **上下文溢出**：对话太长，超过了窗口上限 → 最早的消息被截断或丢弃
- **注意力稀释**：虽然窗口有 128K，但模型对**开头和中间**的内容注意力会衰减（"Lost in the Middle"问题）
- **成本爆炸**：上下文越长，Token 消耗越多，API 费用越贵

### 2.3 在 LangChain/LangGraph 中管理短期记忆

LangChain 提供了多种 **Memory 组件** 来管理短期记忆：

```python
from langchain.memory import (
    ConversationBufferMemory,       # 原始缓冲：全量保存
    ConversationBufferWindowMemory, # 窗口缓冲：只保留最近 K 轮
    ConversationSummaryMemory,      # 摘要缓冲：历史压缩成摘要
    ConversationEntityMemory,       # 实体记忆：提取关键实体
    VectorStoreRetrieverMemory,     # 向量检索记忆：语义搜索历史
)

# ========== 方式1：全量缓冲（最简单，最容易溢出）==========
memory = ConversationBufferMemory(
    memory_key="chat_history",  # 在 Prompt 中的变量名
    return_messages=True,       # 返回 Message 对象列表
)

# 保存对话
memory.save_context(
    {"input": "我叫张三"},
    {"output": "你好张三！"}
)
memory.save_context(
    {"input": "我喜欢吃川菜"},
    {"output": "川菜很辣很香！"}
)

# 查看记忆内容
print(memory.load_memory_variables({}))
# {'chat_history': [HumanMessage(content='我叫张三'),
#                   AIMessage(content='你好张三！'),
#                   HumanMessage(content='我喜欢吃川菜'),
#                   AIMessage(content='川菜很辣很香！')]}


# ========== 方式2：窗口缓冲（只保留最近 K 轮）==========
memory = ConversationBufferWindowMemory(
    k=3,  # 只保留最近 3 轮（6 条消息）
    memory_key="chat_history",
    return_messages=True,
)


# ========== 方式3：摘要缓冲（历史压缩成摘要）==========
from langchain_openai import ChatOpenAI

memory = ConversationSummaryMemory(
    llm=ChatOpenAI(model="gpt-4o-mini"),  # 用 LLM 做摘要
    memory_key="chat_history",
    return_messages=True,
)

memory.save_context({"input": "我叫张三，今年30岁，在北京做软件工程师..."}, 
                    {"output": "很高兴认识你！"})
memory.save_context({"input": "我养了一只猫叫咪咪..."}, 
                    {"output": "咪咪一定很可爱！"})

print(memory.load_memory_variables({}))
# {'chat_history': [SystemMessage(content='张三是一位30岁的软件工程师，住在北京。
#                   他养了一只叫咪咪的猫。')]}
# 注意：原始对话被压缩成了摘要！


# ========== 方式4：在 LangGraph 中管理记忆（推荐生产环境）==========
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import operator

class ConversationState(TypedDict):
    """对话状态"""
    messages: Annotated[List, operator.add]  # 消息历史（只增不减）
    summary: str                             # 对话摘要（压缩后的记忆）
    user_info: dict                          # 提取的用户信息（实体记忆）

# 节点1：处理用户输入
def user_input(state: ConversationState):
    # 用户消息已在 State 中
    return {}

# 节点2：管理记忆（决定是否压缩）
def manage_memory(state: ConversationState):
    messages = state["messages"]
    summary = state.get("summary", "")
    
    # 计算当前 Token 数（简化版，实际用 tiktoken）
    total_chars = sum(len(m.content) for m in messages)
    
    # 如果超过阈值，进行摘要压缩
    if total_chars > 8000:  # 约 2000 tokens 的汉字
        # 保留最近 2 轮对话，把更早的压缩成摘要
        recent_messages = messages[-4:]  # 最近 2 轮 = 4 条消息
        older_messages = messages[:-4]
        
        # 用 LLM 生成摘要
        summary_prompt = f"""将以下对话历史压缩成简短摘要，保留关键信息：
{format_messages(older_messages)}

现有摘要：{summary}

更新后的摘要："""
        
        llm = ChatOpenAI(model="gpt-4o-mini")
        new_summary = llm.invoke(summary_prompt).content
        
        return {
            "summary": new_summary,
            "messages": recent_messages  # 替换为保留的最近消息
        }
    
    return {}

# 节点3：生成回复
def generate(state: ConversationState):
    messages = state["messages"]
    summary = state.get("summary", "")
    
    # 构建 Prompt：摘要 + 最近对话 + 系统指令
    system_msg = SystemMessage(content=f"你是一个助手。对话摘要：{summary}")
    all_messages = [system_msg] + messages
    
    llm = ChatOpenAI(model="gpt-4o")
    response = llm.invoke(all_messages)
    
    return {"messages": [response]}

# 构建图
workflow = StateGraph(ConversationState)
workflow.add_node("user_input", user_input)
workflow.add_node("manage_memory", manage_memory)
workflow.add_node("generate", generate)

workflow.set_entry_point("user_input")
workflow.add_edge("user_input", "manage_memory")
workflow.add_edge("manage_memory", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# 运行
result = app.invoke({
    "messages": [HumanMessage(content="我叫张三，做软件的")],
    "summary": "",
    "user_info": {}
})
```

---

## 三、分层压缩策略 ⭐

### 3.1 为什么需要压缩？

假设你和 AI 聊了 50 轮，每轮平均 200 字：

```
50 轮 × 200 字 × 2（用户+AI）= 20,000 字 ≈ 6,000 tokens

加上系统 Prompt、RAG 文档、工具结果...
很容易超过 8K/16K 的上下文限制！

不压缩的结果：
  ❌ 最早的消息被截断（"我叫张三"可能丢失）
  ❌ API 费用飙升（按 Token 收费）
  ❌ 响应变慢（模型处理长文本更慢）
  ❌ 注意力稀释（模型"看不见"重要信息）

压缩的目标：
  ✅ 保留关键信息（你是谁、你的偏好、重要事实）
  ✅ 丢弃低价值信息（寒暄、重复确认）
  ✅ 控制总 Token 在预算内
```

### 3.2 常见压缩策略详解

#### 3.2.1 最近 K 轮保留（Last K / Sliding Window）

**原理**：只保留最近 K 轮对话，更早的直接丢弃。

```
对话历史: [R1, R2, R3, R4, R5, R6, R7, R8, R9, R10]
K=3 保留:              [R8, R9, R10]  ← 只保留最近3轮
                       R1~R7 被丢弃
```

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=3)

# 或手动实现
class SlidingWindowMemory:
    def __init__(self, k: int = 5):
        self.k = k  # 保留轮数
        self.messages = []
    
    def add(self, user_msg: str, ai_msg: str):
        self.messages.append(("user", user_msg))
        self.messages.append(("ai", ai_msg))
        # 只保留最后 2*k 条消息（k轮 = 2k条）
        self.messages = self.messages[-(2 * self.k):]
    
    def get_context(self) -> str:
        return "\n".join([f"{role}: {msg}" for role, msg in self.messages])
```

**优点**：简单、快、确定性高
**缺点**：可能丢失重要信息（比如第1轮用户说了自己的名字）

**适用场景**：客服对话、闲聊、对历史依赖不强的场景

---

#### 3.2.2 Token 预算截断（Token Budget Truncation）

**原理**：设定一个 Token 上限，从最早的消息开始删除，直到总 Token 数低于预算。

```python
import tiktoken

class TokenBudgetMemory:
    def __init__(self, budget: int = 4000, model: str = "gpt-4o"):
        self.budget = budget  # Token 预算
        self.encoder = tiktoken.encoding_for_model(model)
        self.messages = []
    
    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self._enforce_budget()
    
    def _enforce_budget(self):
        """从最早的消息开始删除，直到满足预算"""
        while self._total_tokens() > self.budget and len(self.messages) > 2:
            # 保留系统消息，删除最早的用户/AI消息
            self.messages.pop(1)  # 删除第1条（系统消息是第0条）
    
    def _total_tokens(self) -> int:
        text = "\n".join([f"{m['role']}: {m['content']}" for m in self.messages])
        return len(self.encoder.encode(text))
    
    def get_messages(self):
        return self.messages
```

**优点**：精确控制 Token 消耗
**缺点**：可能把重要信息截断

**改进——优先保留高价值消息**：

```python
class SmartTokenBudgetMemory(TokenBudgetMemory):
    def _enforce_budget(self):
        """智能截断：优先删除低价值消息"""
        while self._total_tokens() > self.budget:
            # 计算每条消息的"重要性分数"
            scores = []
            for i, msg in enumerate(self.messages):
                score = self._importance_score(msg)
                scores.append((i, score))
            
            # 删除分数最低的非系统消息
            scores = [(i, s) for i, s in scores if self.messages[i]["role"] != "system"]
            if not scores:
                break
            
            min_idx = min(scores, key=lambda x: x[1])[0]
            self.messages.pop(min_idx)
    
    def _importance_score(self, msg) -> float:
        """计算消息重要性（启发式规则）"""
        content = msg["content"]
        score = 1.0
        
        # 包含关键实体（名字、地点、数字）加分
        if any(kw in content for kw in ["叫", "姓名", "年龄", "地址", "电话"]):
            score += 3.0
        
        # 用户明确强调的信息加分
        if "记住" in content or "重要" in content:
            score += 2.0
        
        # 越新的消息默认越重要（时间衰减）
        # 这里简化处理，实际可以用消息索引
        
        return score
```

---

#### 3.2.3 摘要压缩（Summarization）⭐ 最常用

**原理**：用 LLM 把历史对话压缩成一段摘要，只保留摘要+最近几轮原始对话。

```
原始对话（10轮）: [R1, R2, R3, R4, R5, R6, R7, R8, R9, R10]
                       │
                       ▼ 用 LLM 摘要 R1~R7
摘要: "用户叫张三，30岁，软件工程师，喜欢川菜和猫。"
                       │
                       ▼ 保留摘要 + 最近3轮
最终上下文: [摘要, R8, R9, R10]
```

```python
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI

memory = ConversationSummaryMemory(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.0),
    memory_key="history",
    return_messages=True,
)

# LangChain 会自动在每次 save_context 时检查是否需要摘要
# 你也可以手动控制摘要频率

# ========== 自定义摘要策略（更精细控制）==========
class CustomSummarizer:
    def __init__(self, llm, trigger_threshold: int = 3000):
        self.llm = llm
        self.trigger_threshold = trigger_threshold  # 触发摘要的 Token 阈值
        self.messages = []
        self.summary = ""
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        
        # 检查是否需要摘要
        total_tokens = self._estimate_tokens()
        if total_tokens > self.trigger_threshold:
            self._summarize()
    
    def _summarize(self):
        """执行摘要"""
        # 保留最近 2 轮，摘要更早的
        recent = self.messages[-4:]  # 最近2轮
        to_summarize = self.messages[:-4]
        
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in to_summarize])
        
        prompt = f"""将以下对话历史压缩成简洁摘要，保留所有关键事实（姓名、偏好、重要事件）。
现有摘要：{self.summary}

新对话：
{history_text}

更新后的摘要（保持简短）："""
        
        self.summary = self.llm.invoke(prompt).content.strip()
        self.messages = recent  # 替换为保留的最近消息
        
        print(f"[Summarized] New summary: {self.summary[:100]}...")
    
    def get_context(self) -> List[dict]:
        """获取最终上下文：摘要作为系统消息 + 最近原始消息"""
        result = []
        if self.summary:
            result.append({"role": "system", "content": f"对话历史摘要：{self.summary}"})
        result.extend(self.messages)
        return result
    
    def _estimate_tokens(self) -> int:
        # 简化估算：1 token ≈ 0.75 个汉字 ≈ 4 个英文字符
        total_chars = sum(len(m["content"]) for m in self.messages)
        return int(total_chars / 3)


# ========== 增量摘要（更省 Token）==========
class IncrementalSummarizer(CustomSummarizer):
    """增量摘要：不重新摘要全部历史，只把新消息合并进旧摘要"""
    
    def _summarize(self):
        recent = self.messages[-4:]
        new_messages = self.messages[:-4]
        
        # 只对新消息做摘要，再和旧摘要合并
        new_text = "\n".join([f"{m['role']}: {m['content']}" for m in new_messages])
        
        prompt = f"""将以下新对话合并进现有摘要，保持简洁。

现有摘要：{self.summary}

新对话：
{new_text}

合并后的摘要："""
        
        self.summary = self.llm.invoke(prompt).content.strip()
        self.messages = recent
```

**摘要策略的调优要点**：

| 参数 | 作用 | 建议值 |
|------|------|--------|
| `trigger_threshold` | 触发摘要的 Token 阈值 | 上下文窗口的 50%~70% |
| `保留轮数` | 摘要后保留的最近对话轮数 | 2~4 轮（保持对话连贯性） |
| `摘要模型` | 执行摘要的 LLM | 轻量模型（gpt-4o-mini）即可 |
| `摘要频率` | 多久执行一次摘要 | 每次对话后检查，或按轮数触发 |

---

#### 3.2.4 语义压缩（Semantic Compression）

**原理**：用 Embedding 向量判断消息之间的语义相似度，合并或删除冗余信息。

```
对话历史:
  用户: "北京天气怎么样？"
  AI:   "北京今天晴天，25度。"
  用户: "那上海呢？"
  AI:   "上海今天多云，23度。"
  用户: "北京明天呢？"  ← 和第一条语义相关
  AI:   "北京明天阴天，22度。"

语义压缩：
  发现"北京今天..."和"北京明天..."语义相关
  可以合并为："用户询问了北京和上海的天气，今天北京晴25度、上海多云23度，
               明天北京阴22度。"
```

```python
from langchain_openai import OpenAIEmbeddings
import numpy as np

class SemanticCompressor:
    def __init__(self, similarity_threshold: float = 0.85):
        self.embedding = OpenAIEmbeddings()
        self.threshold = similarity_threshold
        self.messages = []
        self.message_embeddings = []
    
    def add_message(self, role: str, content: str):
        # 计算新消息的 embedding
        emb = self.embedding.embed_query(content)
        
        # 检查是否与最近的消息语义重复
        if self.message_embeddings:
            last_emb = self.message_embeddings[-1]
            similarity = self._cosine_similarity(emb, last_emb)
            
            if similarity > self.threshold:
                # 语义重复：合并或替换
                print(f"[Semantic Merge] Similarity={similarity:.3f}")
                self._merge_with_last(role, content)
                return
        
        self.messages.append({"role": role, "content": content})
        self.message_embeddings.append(emb)
    
    def _cosine_similarity(self, a, b) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _merge_with_last(self, role: str, content: str):
        """合并语义重复的消息"""
        last = self.messages[-1]
        # 简单的拼接合并（实际可用 LLM 智能合并）
        last["content"] += f" [补充: {content}]"
    
    def deduplicate_history(self):
        """对整个历史去重"""
        unique_indices = []
        for i, emb in enumerate(self.message_embeddings):
            is_duplicate = False
            for j in unique_indices:
                if self._cosine_similarity(emb, self.message_embeddings[j]) > self.threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_indices.append(i)
        
        self.messages = [self.messages[i] for i in unique_indices]
        self.message_embeddings = [self.message_embeddings[i] for i in unique_indices]
```

**优点**：能识别语义重复（如"你好""在吗""请问"），比关键词匹配更智能
**缺点**：每次都要计算 embedding，有一定延迟和成本

---

#### 3.2.5 分层记忆（Hierarchical Memory）⭐⭐ 企业级方案

**原理**：把记忆分成多个层次，每层有不同的粒度和压缩程度。

```
┌─────────────────────────────────────────────────────────────┐
│                    分层记忆架构                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Layer 1: 原始消息层（Raw）                                  │
│  ├── 保留最近 2-4 轮完整对话                                  │
│  └── 作用：保持对话连贯性                                     │
│                                                             │
│  Layer 2: 摘要层（Summary）                                  │
│  ├── 本轮会话的逐步摘要                                       │
│  └── 作用：保留本轮会话的上下文                               │
│                                                             │
│  Layer 3: 主题层（Topics）                                   │
│  ├── 从多轮对话中提取的实体和主题                             │
│  └── 如：{用户姓名: "张三", 偏好: ["川菜", "猫"], ...}        │
│                                                             │
│  Layer 4: 跨会话层（Cross-Session）                          │
│  ├── 持久化存储的跨会话知识                                   │
│  └── 存储在向量库 / 关系数据库中                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘

构建 Prompt 时的优先级：
  系统指令 > Layer 4（跨会话知识）> Layer 3（主题实体）> 
  Layer 2（本轮摘要）> Layer 1（最近原始消息）
```

```python
class HierarchicalMemory:
    """分层记忆管理器"""
    
    def __init__(self, llm, vector_store=None):
        self.llm = llm
        self.vector_store = vector_store
        
        # Layer 1: 原始消息
        self.raw_messages = []
        self.raw_limit = 4  # 保留最近2轮
        
        # Layer 2: 本轮摘要
        self.session_summary = ""
        
        # Layer 3: 主题实体
        self.topics = {}  # {key: value}
        
        # Layer 4: 跨会话向量存储
        self.vector_store = vector_store
    
    def add_turn(self, user_msg: str, ai_msg: str):
        """添加一轮对话"""
        # Layer 1: 加入原始消息
        self.raw_messages.append(("user", user_msg))
        self.raw_messages.append(("ai", ai_msg))
        
        # Layer 1 溢出时，触发 Layer 2 摘要
        if len(self.raw_messages) > self.raw_limit:
            self._update_summary()
            self.raw_messages = self.raw_messages[-self.raw_limit:]
        
        # Layer 3: 提取实体（每 N 轮执行一次）
        if len(self.raw_messages) % 4 == 0:
            self._extract_topics()
    
    def _update_summary(self):
        """更新 Layer 2 摘要"""
        to_summarize = self.raw_messages[:-self.raw_limit]
        text = "\n".join([f"{r}: {m}" for r, m in to_summarize])
        
        prompt = f"""将以下对话合并进现有摘要，保持简洁。
现有摘要：{self.session_summary}
新对话：{text}
更新后摘要："""
        
        self.session_summary = self.llm.invoke(prompt).content.strip()
    
    def _extract_topics(self):
        """提取 Layer 3 主题实体"""
        all_text = "\n".join([f"{r}: {m}" for r, m in self.raw_messages])
        
        prompt = f"""从以下对话中提取关键信息，输出 JSON 格式：
{{"姓名": "...", "职业": "...", "偏好": [...], "重要事件": [...]}}

对话：{all_text}
"""
        
        try:
            import json
            extracted = json.loads(self.llm.invoke(prompt).content)
            self.topics.update(extracted)
        except:
            pass
    
    def build_prompt_context(self) -> str:
        """构建最终上下文（按优先级拼接）"""
        parts = []
        
        # Layer 4: 跨会话知识（如果有向量库）
        if self.vector_store:
            # 用最近一条用户消息检索相关历史
            recent_user_msg = next(
                (m for r, m in reversed(self.raw_messages) if r == "user"),
                ""
            )
            if recent_user_msg:
                relevant = self.vector_store.similarity_search(recent_user_msg, k=2)
                if relevant:
                    parts.append("[历史相关记忆]\n" + "\n".join([d.page_content for d in relevant]))
        
        # Layer 3: 主题实体
        if self.topics:
            parts.append(f"[已知信息] {self.topics}")
        
        # Layer 2: 本轮摘要
        if self.session_summary:
            parts.append(f"[本轮对话摘要] {self.session_summary}")
        
        # Layer 1: 最近原始消息
        if self.raw_messages:
            parts.append("[最近对话]\n" + "\n".join([f"{r}: {m}" for r, m in self.raw_messages]))
        
        return "\n\n".join(parts)
```

### 3.3 压缩参数如何调整？

**决策树**：

```
你的对话平均轮数是多少？
    │
    ├─ < 10 轮 ──► 窗口保留（K=5~10）或全量
    │
    └─ >= 10 轮 ──► 需要压缩
            │
            ├─ 对历史精度要求高？（如医疗问诊）
            │       └─ 是 ──► Token 预算 + 智能保留重要消息
            │
            ├─ 对话主题多变？（如闲聊）
            │       └─ 是 ──► 语义压缩 + 分层记忆
            │
            ├─ 单主题深度讨论？（如技术支持）
            │       └─ 是 ──► 摘要压缩（保留核心论点）
            │
            └─ 跨会话需要记忆？（如个人助手）
                    └─ 是 ──► 分层记忆 + 向量库存储
```

**参数调优表**：

| 参数 | 适用场景 | 建议值 | 调大效果 | 调小效果 |
|------|----------|--------|----------|----------|
| `window_size (K)` | 滑动窗口保留轮数 | 3~10 | 保留更多上下文，更连贯 | 省 Token，可能丢失信息 |
| `token_budget` | Token 预算上限 | 4K~8K | 信息更完整 | 更省成本，更省时间 |
| `summary_threshold` | 触发摘要的 Token 阈值 | 预算的 60%~80% | 摘要频率低，原始信息多 | 摘要更频繁，压缩更早 |
| `semantic_threshold` | 语义去重阈值 | 0.80~0.90 | 更激进合并，更省空间 | 保留更多细节，更安全 |
| `hierarchy_levels` | 分层记忆层数 | 2~4 | 更细粒度，更灵活 | 更简单，开销更小 |


---

## 四、长期记忆

### 4.1 长期记忆是什么？

长期记忆 = **跨会话、跨时间持久保存的信息**。即使对话结束了、浏览器关闭了、服务器重启了，这些信息依然存在。

```
短期记忆 vs 长期记忆

短期记忆（Session Memory）:
  用户: "我叫张三"
  AI:   "你好张三"
  ...（聊了很久）...
  【浏览器关闭】
  【重新打开】
  用户: "我叫什么？"
  AI:   "抱歉，我不知道"  ← 短期记忆随会话消失

长期记忆（Persistent Memory）:
  用户: "我叫张三"
  AI:   "记住了，你叫张三"
  ...（聊了很久，浏览器关闭）...
  【一周后重新打开】
  用户: "我叫什么？"
  AI:   "你叫张三，我们上周聊过。"  ← 长期记忆还在！
```

**长期记忆存储什么？**

| 记忆类型 | 例子 | 存储方式 |
|----------|------|----------|
| **用户画像** | 姓名、年龄、职业、偏好 | 关系数据库 / KV 存储 |
| **事实知识** | "张三喜欢川菜"、"公司在 20 楼" | 向量数据库 |
| **历史对话** | 之前聊过的话题和结论 | 向量数据库 / 文件存储 |
| **任务状态** | 未完成的待办事项 | 关系数据库 |
| **自定义规则** | "每次回复用简洁风格" | KV 存储 |

### 4.2 长期记忆的触发机制

信息不是随意存进长期记忆的——需要**触发条件**：

```
┌─────────────────────────────────────────────────────────────┐
│                   长期记忆触发机制                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  触发方式1: 显式存储（用户主动要求）                         │
│  用户: "请记住我喜欢喝美式咖啡"                               │
│  AI:   "好的，已记录。"                                       │
│       │                                                     │
│       ▼ 立即写入长期记忆                                     │
│                                                             │
│  触发方式2: 隐式提取（AI 自动识别关键信息）                  │
│  用户: "我是做 Java 的，干了 5 年了"                         │
│       │                                                     │
│       ▼ AI 自动提取实体                                       │
│       {职业: "Java工程师", 经验: "5年"}                       │
│       │                                                     │
│       ▼ 满足置信度阈值后存入                                 │
│                                                             │
│  触发方式3: 会话结束总结（End-of-Session Summary）           │
│  【会话结束】                                                 │
│       │                                                     │
│       ▼ LLM 总结本轮会话关键信息                             │
│       "用户张三咨询了年假政策，对5天的规定表示满意"           │
│       │                                                     │
│       ▼ 存入向量库                                          │
│                                                             │
│  触发方式4: 定时归档（Batch Processing）                     │
│  每天凌晨 3 点：                                              │
│       1. 扫描当日所有会话                                     │
│       2. 提取关键事实和主题                                   │
│       3. 更新用户画像和知识库                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**隐式提取的实现**：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json

class ImplicitMemoryExtractor:
    """自动从对话中提取关键信息并存入长期记忆"""
    
    def __init__(self, llm, confidence_threshold: float = 0.7):
        self.llm = llm
        self.confidence_threshold = confidence_threshold
        self.facts = {}  # 提取到的事实
    
    def extract(self, conversation_history: List[dict]) -> List[dict]:
        """
        从对话历史中提取值得记忆的事实
        返回: [{"fact": "...", "confidence": 0.9, "type": "preference"}, ...]
        """
        text = "\n".join([f"{m['role']}: {m['content']}" for m in conversation_history])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个信息提取专家。请分析以下对话，提取所有值得长期记忆的事实。

输出 JSON 数组格式：
[
  {
    "fact": "具体事实陈述",
    "type": "profile|preference|fact|task",
    "confidence": 0-1 的置信度,
    "key": "用于索引的键名"
  }
]

只提取明确陈述的信息，不要推测。"""),
            ("human", "对话：\n{conversation}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"conversation": text}).content
        
        try:
            extracted = json.loads(response)
            # 过滤低置信度
            return [f for f in extracted if f.get("confidence", 0) >= self.confidence_threshold]
        except:
            return []
    
    def should_store(self, new_facts: List[dict]) -> bool:
        """判断是否应该存入长期记忆"""
        # 规则1: 高置信度（> 0.8）直接存
        high_conf = [f for f in new_facts if f["confidence"] > 0.8]
        if high_conf:
            return True
        
        # 规则2: 同一事实多次出现（跨会话验证）
        for fact in new_facts:
            if self._has_seen_before(fact):
                return True
        
        # 规则3: 用户明确说"记住"
        for fact in new_facts:
            if "记住" in fact.get("source", ""):
                return True
        
        return False
    
    def _has_seen_before(self, fact: dict) -> bool:
        """检查是否之前提取过类似事实"""
        # 简化为检查 key 是否已存在
        return fact.get("key") in self.facts
```

### 4.3 长期记忆的作用

| 作用 | 说明 | 例子 |
|------|------|------|
| **个性化体验** | 记住用户偏好，提供定制回复 | "上次你说喜欢简洁风格，这次我也简短回答" |
| **跨会话连贯** | 不同时间聊天也能接上话 | "上周你提到在准备面试，结果怎么样？" |
| **知识积累** | 系统越用越"懂"用户/业务 | "根据你之前反馈，这类问题通常是因为..." |
| **效率提升** | 不用重复介绍自己 | 不用每次都说"我是张三，做 Java 的" |
| **关系建立** | 让用户感觉被重视 | "生日快乐！你之前说今天是你的生日" |

### 4.4 在 LangChain 中实现长期记忆

```python
from langchain.memory import (
    ConversationEntityMemory,
    ConversationKGMemory,  # 知识图谱记忆
)
from langchain_openai import ChatOpenAI

# ========== 方式1: 实体记忆（自动提取实体）==========
memory = ConversationEntityMemory(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    memory_key="entity_memory",
)

memory.save_context(
    {"input": "我叫张三，30岁，喜欢川菜"},
    {"output": "很高兴认识你！"}
)

print(memory.load_memory_variables({"input": "我叫什么名字？"}))
# {'entity_memory': '张三是一位30岁的人，喜欢川菜。'}


# ========== 方式2: 知识图谱记忆（提取关系）==========
memory = ConversationKGMemory(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    memory_key="kg_memory",
)

memory.save_context(
    {"input": "张三在北京工作，他喜欢去海底捞吃川菜"},
    {"output": "好的"}
)

print(memory.load_memory_variables({"input": "张三在哪里工作？"}))
# 会输出知识图谱形式的关系


# ========== 方式3: 自定义长期记忆（Redis + 向量库）==========
import redis
import json
from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings

class PersistentMemory:
    """基于 Redis + pgvector 的长期记忆"""
    
    def __init__(self, user_id: str, redis_host="localhost"):
        self.user_id = user_id
        self.redis = redis.Redis(host=redis_host, decode_responses=True)
        self.vector_store = PGVector(
            collection_name=f"user_{user_id}_memory",
            embedding_function=OpenAIEmbeddings(),
            connection_string="postgresql+psycopg://user:password@localhost:5432/memory_db"
        )
    
    def store_fact(self, key: str, value: str, fact_type: str = "general"):
        """存储结构化事实到 Redis"""
        data = {
            "value": value,
            "type": fact_type,
            "timestamp": time.time(),
            "access_count": 0
        }
        self.redis.hset(f"user:{self.user_id}:facts", key, json.dumps(data))
    
    def store_conversation(self, text: str, metadata: dict = None):
        """存储对话片段到向量库"""
        from langchain_core.documents import Document
        doc = Document(
            page_content=text,
            metadata=metadata or {"user_id": self.user_id, "timestamp": time.time()}
        )
        self.vector_store.add_documents([doc])
    
    def get_fact(self, key: str) -> str:
        """获取结构化事实"""
        data = self.redis.hget(f"user:{self.user_id}:facts", key)
        if data:
            fact = json.loads(data)
            fact["access_count"] += 1
            self.redis.hset(f"user:{self.user_id}:facts", key, json.dumps(fact))
            return fact["value"]
        return None
    
    def get_all_facts(self) -> dict:
        """获取用户所有事实"""
        facts = self.redis.hgetall(f"user:{self.user_id}:facts")
        return {k: json.loads(v) for k, v in facts.items()}
    
    def search_memory(self, query: str, k: int = 3) -> List[Document]:
        """语义搜索历史记忆"""
        return self.vector_store.similarity_search(query, k=k)
    
    def get_user_profile(self) -> str:
        """生成用户画像文本（用于 Prompt）"""
        facts = self.get_all_facts()
        if not facts:
            return ""
        
        lines = []
        for key, fact in facts.items():
            lines.append(f"- {key}: {fact['value']}")
        
        return "用户已知信息：\n" + "\n".join(lines)


# ========== 使用示例 ==========
memory = PersistentMemory(user_id="user_123")

# 存储事实
memory.store_fact("name", "张三", "profile")
memory.store_fact("preference_food", "川菜", "preference")
memory.store_fact("job", "Java工程师", "profile")

# 存储对话
memory.store_conversation(
    "用户询问了年假政策，对5天的规定表示满意",
    {"topic": "年假", "sentiment": "positive"}
)

# 后续会话中加载记忆
profile = memory.get_user_profile()
print(profile)
# 用户已知信息：
# - name: 张三
# - preference_food: 川菜
# - job: Java工程师

# 语义搜索相关记忆
results = memory.search_memory("年假", k=2)
for r in results:
    print(r.page_content)
```

### 4.5 向量库作为长期记忆

向量数据库是长期记忆的**核心基础设施**。它让 AI 能够：

1. **语义检索**：不用精确关键词，用意思找意思
2. **大规模存储**：存储数十万条记忆片段
3. **高效查询**：毫秒级返回相关记忆

```python
# ========== 向量库长期记忆完整实现 ==========
from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from datetime import datetime

class VectorLongTermMemory:
    """基于向量库的长期记忆系统"""
    
    def __init__(self, user_id: str, connection_string: str = "postgresql+psycopg://user:password@localhost:5432/ltm_db"):
        self.user_id = user_id
        self.embedding = OpenAIEmbeddings()
        self.vectorstore = PGVector(
            collection_name=f"ltm_{user_id}",
            embedding_function=self.embedding,
            connection_string=connection_string,
        )
    
    def add_memory(self, content: str, memory_type: str = "fact", 
                   importance: int = 3, tags: List[str] = None):
        """
        添加一条记忆
        
        Args:
            content: 记忆内容
            memory_type: 类型（fact/conversation/preference/event）
            importance: 重要程度（1-5，5 最重要）
            tags: 标签（用于过滤）
        """
        doc = Document(
            page_content=content,
            metadata={
                "user_id": self.user_id,
                "type": memory_type,
                "importance": importance,
                "tags": tags or [],
                "created_at": datetime.now().isoformat(),
                "access_count": 0,
            }
        )
        self.vectorstore.add_documents([doc])
    
    def recall(self, query: str, k: int = 5, 
               memory_type: str = None,
               min_importance: int = 1) -> List[Document]:
        """
        回忆相关记忆
        
        Args:
            query: 查询语义
            k: 返回数量
            memory_type: 按类型过滤
            min_importance: 最小重要度过滤
        """
        # 先语义检索
        results = self.vectorstore.similarity_search(query, k=k * 2)
        
        # 过滤
        filtered = []
        for doc in results:
            meta = doc.metadata
            
            # 类型过滤
            if memory_type and meta.get("type") != memory_type:
                continue
            
            # 重要度过滤
            if meta.get("importance", 1) < min_importance:
                continue
            
            filtered.append(doc)
            
            # 更新访问计数
            meta["access_count"] = meta.get("access_count", 0) + 1
            meta["last_accessed"] = datetime.now().isoformat()
        
        return filtered[:k]
    
    def forget(self, query: str, confirm: bool = False):
        """遗忘（删除）相关记忆"""
        results = self.vectorstore.similarity_search(query, k=10)
        
        for doc in results:
            # 实际删除（pgvector 的删除接口：通过 PostgreSQL SQL 或 similarity_search 定位后删除）
            # self.vectorstore._collection.delete(ids=[doc.id])
            pass
    
    def consolidate(self):
        """
        记忆巩固：定期整理记忆
        - 合并相似记忆
        - 删除低重要度且长期未访问的记忆
        - 升级高频访问记忆的重要度
        """
        # 获取所有记忆（pgvector 通过检索获取，这里用空查询获取大量结果）
        all_docs = self.vectorstore.similarity_search("", k=10000)
        
        # 找出长期未访问（30天）且低重要度的记忆
        to_forget = []
        for doc in all_docs:
            meta = doc.metadata
            last_accessed = meta.get("last_accessed")
            importance = meta.get("importance", 1)
            access_count = meta.get("access_count", 0)
            
            if importance <= 2 and access_count < 3:
                # 标记为可遗忘
                to_forget.append(doc_id)
        
        print(f"[Consolidate] 标记 {len(to_forget)} 条记忆为可遗忘")
        # 实际执行删除...


# ========== 使用示例 ==========
ltm = VectorLongTermMemory(user_id="user_001")

# 添加多种类型的记忆
ltm.add_memory("用户叫张三，30岁", "fact", importance=5, tags=["profile", "name"])
ltm.add_memory("用户喜欢川菜，不喜欢甜食", "preference", importance=4, tags=["food"])
ltm.add_memory("用户上周咨询了年假政策", "conversation", importance=2, tags=["年假"])
ltm.add_memory("用户今天生日", "event", importance=5, tags=["birthday"])

# 回忆
memories = ltm.recall("用户喜欢吃什么", k=3)
for m in memories:
    print(f"[{m.metadata['type']}] {m.page_content}")

# 按类型过滤
profile_memories = ltm.recall("用户信息", memory_type="fact", k=5)
```

---

## 五、会话窗口

### 5.1 会话窗口是什么？

会话窗口 = **界定"当前对话"范围的边界**。它决定了哪些消息属于"现在正在聊的"，哪些属于"历史"。

```
场景1: 单会话（浏览器标签页）
  用户打开页面 → 聊天 → 关闭页面 → 【会话结束】
  
场景2: 多会话并行
  用户同时打开 3 个标签页，和同一个 AI 聊不同话题
  会话A: 聊技术问题
  会话B: 聊请假流程
  会话C: 闲聊
  
场景3: 会话恢复
  用户昨天聊了一半，今天继续
  需要加载昨天的上下文
```

### 5.2 滑动窗口 vs 固定窗口

| 窗口类型 | 原理 | 优点 | 缺点 | 适用场景 |
|----------|------|------|------|----------|
| **固定窗口** | 一个会话固定保留 N 条消息，满了就截断 | 简单、内存可控 | 可能切断重要上下文 | 客服机器人 |
| **滑动窗口** | 始终保留最近 K 条，旧消息滑出 | 实时反映最新状态 | 历史信息丢失 | 实时对话 |
| **动态窗口** | 根据内容重要性动态调整窗口大小 | 重要信息不被截断 | 实现复杂 | 个人助手 |
| **会话级窗口** | 整个会话共享一个上下文，跨轮次累积 | 连贯性好 | 容易溢出 | 短对话 |

```python
class SessionWindowManager:
    """会话窗口管理器"""
    
    def __init__(self, window_type: str = "sliding", max_messages: int = 20):
        self.window_type = window_type
        self.max_messages = max_messages
        self.sessions: Dict[str, List[dict]] = {}  # session_id -> messages
    
    def create_session(self, session_id: str) -> str:
        """创建新会话"""
        self.sessions[session_id] = []
        return session_id
    
    def add_message(self, session_id: str, role: str, content: str):
        """添加消息到会话"""
        if session_id not in self.sessions:
            self.create_session(session_id)
        
        self.sessions[session_id].append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        
        # 应用窗口策略
        self._apply_window(session_id)
    
    def _apply_window(self, session_id: str):
        """根据窗口类型截断消息"""
        messages = self.sessions[session_id]
        
        if self.window_type == "fixed":
            # 固定窗口：只保留前 N 条（或后 N 条）
            if len(messages) > self.max_messages:
                # 保留系统消息 + 最近 N 条
                system_msgs = [m for m in messages if m["role"] == "system"]
                other_msgs = [m for m in messages if m["role"] != "system"]
                self.sessions[session_id] = system_msgs + other_msgs[-self.max_messages:]
        
        elif self.window_type == "sliding":
            # 滑动窗口：始终保留最近 N 条
            if len(messages) > self.max_messages:
                self.sessions[session_id] = messages[-self.max_messages:]
        
        elif self.window_type == "time_based":
            # 时间窗口：只保留最近 N 分钟的消息
            cutoff = time.time() - 3600  # 1小时
            self.sessions[session_id] = [
                m for m in messages if m["timestamp"] > cutoff
            ]
    
    def get_context(self, session_id: str) -> List[dict]:
        """获取会话当前上下文"""
        return self.sessions.get(session_id, [])
    
    def clear_session(self, session_id: str):
        """清空会话（用户点击"新建对话"）"""
        if session_id in self.sessions:
            # 可选：清空前先保存到长期记忆
            self._archive_session(session_id)
            self.sessions[session_id] = []
    
    def _archive_session(self, session_id: str):
        """归档会话到长期记忆"""
        messages = self.sessions.get(session_id, [])
        if not messages:
            return
        
        # 总结会话
        text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        # 存入向量库...（调用长期记忆模块）
        print(f"[Archive] Session {session_id} archived")
```

### 5.3 会话隔离与共享

**多用户场景**：每个用户应该有独立的会话和记忆。

```
用户A 的会话 ──► 用户A 的短期记忆 ──► 用户A 的长期记忆
     │                                    │
     │ 隔离！                             │ 隔离！
     │                                    │
用户B 的会话 ──► 用户B 的短期记忆 ──► 用户B 的长期记忆
```

```python
class MultiUserMemoryManager:
    """多用户记忆管理器（会话隔离）"""
    
    def __init__(self):
        # 每个用户独立的记忆存储
        self.user_memories: Dict[str, PersistentMemory] = {}
        self.user_sessions: Dict[str, SessionWindowManager] = {}
    
    def get_or_create_user_memory(self, user_id: str) -> PersistentMemory:
        """获取或创建用户记忆（隔离）"""
        if user_id not in self.user_memories:
            self.user_memories[user_id] = PersistentMemory(user_id)
        return self.user_memories[user_id]
    
    def get_or_create_session(self, user_id: str, session_id: str):
        """获取或创建用户会话"""
        key = f"{user_id}:{session_id}"
        if key not in self.user_sessions:
            self.user_sessions[key] = SessionWindowManager()
        return self.user_sessions[key]
    
    def chat(self, user_id: str, session_id: str, message: str) -> str:
        """处理用户消息（完整流程）"""
        # 1. 获取用户专属记忆
        user_memory = self.get_or_create_user_memory(user_id)
        session = self.get_or_create_session(user_id, session_id)
        
        # 2. 加载长期记忆作为上下文
        profile = user_memory.get_user_profile()
        relevant_memories = user_memory.search_memory(message, k=3)
        
        # 3. 构建 Prompt（长期记忆 + 短期会话）
        context_parts = []
        if profile:
            context_parts.append(f"[用户画像]\n{profile}")
        if relevant_memories:
            mem_text = "\n".join([m.page_content for m in relevant_memories])
            context_parts.append(f"[相关记忆]\n{mem_text}")
        
        session_context = session.get_context(session_id)
        context_parts.append("[当前对话]\n" + format_messages(session_context))
        
        prompt = "\n\n".join(context_parts)
        
        # 4. 调用 LLM 生成回复
        response = llm.invoke(prompt + f"\n\n用户: {message}")
        
        # 5. 更新会话
        session.add_message(session_id, "user", message)
        session.add_message(session_id, "assistant", response.content)
        
        # 6. 可选：提取信息到长期记忆
        if should_extract_memory(message, response.content):
            user_memory.store_conversation(f"用户: {message}\nAI: {response.content}")
        
        return response.content
```

---

## 六、缓存机制

### 6.1 为什么需要缓存？

AI 系统的很多操作是**重复且昂贵的**，缓存能大幅降本增效：

```
场景1: 相同问题反复问
  用户A: "公司年假政策是什么？" → LLM生成答案（耗时2秒，花费¥0.05）
  用户B: "公司年假政策是什么？" → 同样的问题！
  
  无缓存：再花 ¥0.05
  有缓存：直接返回缓存结果，花费 ¥0

场景2: Embedding重复计算
  文档A有1000个chunk，每个都要向量化
  无缓存：1000次API调用
  有缓存：第一次计算后存入缓存，后续直接读

场景3: 检索结果缓存
  高频问题"年假"的检索结果基本不变
  无缓存：每次都要向量检索
  有缓存：直接返回上次结果
```

### 6.2 缓存类型详解

#### 6.2.1 LLM 响应缓存

```python
from langchain.globals import set_llm_cache
from langchain_community.cache import (
    SQLiteCache,      # SQLite 本地缓存
    InMemoryCache,    # 内存缓存（进程重启丢失）
)
from langchain_redis import RedisCache

# 方式1: SQLite 本地缓存（推荐开发和测试）
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# 方式2: Redis 缓存（推荐生产环境，支持分布式）
# set_llm_cache(RedisCache(redis_url="redis://localhost:6379/0"))

# 启用后，所有 LangChain LLM 调用会自动缓存
llm = ChatOpenAI(model="gpt-4o-mini")

# 第一次调用（实际请求 API）
response1 = llm.invoke("你好")  # 花费 ¥0.001

# 第二次调用（相同输入，直接返回缓存）
response2 = llm.invoke("你好")  # 花费 ¥0！速度提升 10x+
```

**缓存命中率优化**：

```python
class SmartLLMCache:
    """智能 LLM 缓存（支持模糊匹配）"""
    
    def __init__(self, similarity_threshold: float = 0.95):
        self.cache = {}  # query_hash -> response
        self.embeddings = {}  # query_hash -> embedding
        self.embedding_model = OpenAIEmbeddings()
        self.threshold = similarity_threshold
    
    def get(self, query: str) -> str:
        """获取缓存，支持语义相似匹配"""
        query_emb = self.embedding_model.embed_query(query)
        
        # 查找语义相似的缓存
        best_match = None
        best_score = 0
        
        for cached_hash, cached_emb in self.embeddings.items():
            score = cosine_similarity(query_emb, cached_emb)
            if score > best_score:
                best_score = score
                best_match = cached_hash
        
        if best_match and best_score > self.threshold:
            print(f"[Cache Hit] Similarity={best_score:.3f}")
            return self.cache[best_match]
        
        return None
    
    def set(self, query: str, response: str):
        """存入缓存"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        self.cache[query_hash] = response
        self.embeddings[query_hash] = self.embedding_model.embed_query(query)
```

#### 6.2.2 Embedding 缓存

```python
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

# 创建本地文件存储（缓存 embedding）
store = LocalFileStore("./embedding_cache")

# 包装 embedding 模型，自动缓存
_cached_embedding = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=OpenAIEmbeddings(),
    document_embedding_cache=store,
    namespace="company_docs",  # 避免不同项目缓存冲突
)

# 使用方式和普通 embedding 一样，但会自动缓存
vectorstore = PGVector.from_documents(
    documents=docs,
    embedding=_cached_embedding,  # 使用带缓存的版本
    connection_string="postgresql+psycopg://user:password@localhost:5432/memory_db",
    collection_name="cached_embeddings",
)

# 第二次加载相同文档时，embedding 会从缓存读，速度提升 10x+
```

#### 6.2.3 检索结果缓存

```python
import functools
import hashlib
import time

class RetrievalCache:
    """检索结果缓存（带 TTL）"""
    
    def __init__(self, ttl_seconds: int = 300):
        self.cache = {}  # query_hash -> (results, timestamp)
        self.ttl = ttl_seconds
    
    def get(self, query: str) -> List[Document]:
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        if query_hash in self.cache:
            results, timestamp = self.cache[query_hash]
            if time.time() - timestamp < self.ttl:
                print(f"[Retrieval Cache Hit] '{query[:30]}...'")
                return results
            else:
                del self.cache[query_hash]  # TTL 过期
        
        return None
    
    def set(self, query: str, results: List[Document]):
        query_hash = hashlib.md5(query.encode()).hexdigest()
        self.cache[query_hash] = (results, time.time())
    
    def invalidate(self, pattern: str = None):
        """使缓存失效（文档更新后调用）"""
        if pattern:
            # 删除匹配的缓存
            to_delete = [k for k in self.cache if pattern in k]
            for k in to_delete:
                del self.cache[k]
        else:
            self.cache.clear()


# 使用
cache = RetrievalCache(ttl_seconds=600)  # 10分钟TTL

def search_with_cache(query: str) -> List[Document]:
    # 先查缓存
    cached = cache.get(query)
    if cached:
        return cached
    
    # 缓存未命中，执行检索
    results = vectorstore.similarity_search(query, k=5)
    
    # 存入缓存
    cache.set(query, results)
    return results
```

### 6.3 LangChain 缓存实现

```python
# ========== 完整的多层缓存架构 ==========
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from functools import lru_cache
import diskcache

class MultiLayerCache:
    """
    三层缓存架构：
    L1: 内存缓存（最快，进程内）
    L2: SQLite 本地缓存（持久化，单机）
    L3: Redis 分布式缓存（跨进程/跨机器）
    """
    
    def __init__(self):
        # L1: 内存（LRU）
        self.l1_memory = {}
        self.l1_max_size = 1000
        
        # L2: 磁盘
        self.l2_disk = diskcache.Cache("./cache_dir")
        
        # L3: Redis（可选）
        # self.l3_redis = redis.Redis()
    
    def get(self, key: str):
        """L1 → L2 → L3 逐级查找"""
        # L1
        if key in self.l1_memory:
            return self.l1_memory[key]
        
        # L2
        if key in self.l2_disk:
            value = self.l2_disk[key]
            # 回填 L1
            self._l1_set(key, value)
            return value
        
        # L3（如有）
        # value = self.l3_redis.get(key)
        
        return None
    
    def set(self, key: str, value, ttl: int = 3600):
        """写入所有层"""
        self._l1_set(key, value)
        self.l2_disk.set(key, value, expire=ttl)
        # self.l3_redis.setex(key, ttl, value)
    
    def _l1_set(self, key, value):
        """L1 写入（带简单的 LRU 淘汰）"""
        if len(self.l1_memory) >= self.l1_max_size:
            # 淘汰最早的
            oldest = next(iter(self.l1_memory))
            del self.l1_memory[oldest]
        self.l1_memory[key] = value


# ========== 在 LangGraph 中集成缓存节点 ==========
from langgraph.graph import StateGraph

def cached_llm_node(state):
    """带缓存的 LLM 调用节点"""
    cache = state.get("cache")
    query = state["messages"][-1].content
    
    # 构造缓存 key
    cache_key = f"llm:{hash(query)}"
    
    # 查缓存
    if cache:
        cached = cache.get(cache_key)
        if cached:
            return {"messages": [AIMessage(content=cached)]}
    
    # 调用 LLM
    response = llm.invoke(state["messages"])
    
    # 写缓存
    if cache:
        cache.set(cache_key, response.content)
    
    return {"messages": [response]}
```

---

## 七、参数调整

### 7.1 记忆相关参数汇总

```python
# 记忆系统配置模板
MEMORY_CONFIG = {
    # ========== 短期记忆 ==========
    "short_term": {
        "window_type": "sliding",           # sliding / fixed / dynamic
        "window_size": 10,                   # 保留消息数
        "token_budget": 4000,                # Token 预算上限
        "context_window_limit": 8000,        # 模型上下文上限
    },
    
    # ========== 压缩策略 ==========
    "compression": {
        "enabled": True,
        "strategy": "hierarchical",          # last_k / token_budget / summary / semantic / hierarchical
        "summary_trigger_tokens": 6000,      # 触发摘要的 Token 阈值
        "summary_model": "gpt-4o-mini",      # 摘要用模型
        "retain_recent_turns": 4,            # 摘要后保留的最近轮数
        "semantic_similarity_threshold": 0.85,  # 语义去重阈值
    },
    
    # ========== 长期记忆 ==========
    "long_term": {
        "enabled": True,
        "vector_store": "pgvector",          # pgvector / milvus / chroma
        "kv_store": "redis",                 # redis / sqlite
        "extraction_confidence": 0.7,        # 自动提取的置信度阈值
        "consolidation_interval_hours": 24,  # 记忆巩固周期
        "max_memories_per_query": 5,         # 单次检索返回的记忆数
        "importance_threshold": 2,           # 存储记忆的最低重要度
    },
    
    # ========== 缓存 ==========
    "cache": {
        "llm_cache_enabled": True,
        "llm_cache_ttl": 3600,               # LLM缓存TTL（秒）
        "embedding_cache_enabled": True,
        "retrieval_cache_enabled": True,
        "retrieval_cache_ttl": 600,          # 检索缓存TTL（秒）
    },
    
    # ========== 会话 ==========
    "session": {
        "session_timeout_minutes": 30,       # 会话超时时间
        "max_concurrent_sessions": 100,      # 最大并发会话数
        "archive_on_close": True,            # 关闭时归档到长期记忆
    }
}
```

### 7.2 各参数的影响和调优建议

| 参数 | 调大效果 | 调小效果 | 适用场景 |
|------|----------|----------|----------|
| `window_size` | 更多历史上下文，更连贯 | 省Token、省成本、更快 | 长对话场景调大；简单问答调小 |
| `token_budget` | 更少信息丢失 | 更省成本、更快响应 | 根据模型上下文和RAG需求调整 |
| `summary_trigger_tokens` | 摘要频率低，原始信息多 | 更早压缩，更省Token | 对精度要求高则调大；成本敏感则调小 |
| `retain_recent_turns` | 最近对话更完整 | 更省Token | 多轮推理场景保留4轮+；简单场景2轮 |
| `extraction_confidence` | 更严格，记忆质量高但数量少 | 更宽松，记忆多但可能噪声大 | 严谨场景（医疗/法律）调大 |
| `max_memories_per_query` | 更多上下文参考 | 更聚焦，更少噪声 | 复杂问题调大；简单问题调小 |
| `retrieval_cache_ttl` | 缓存更久，命中率高 | 数据更新更快被感知 | 知识更新频繁则调小；稳定知识调大 |
| `session_timeout` | 用户不用重复介绍自己 | 内存占用更大 | 个人助手调大；客服机器人调小 |

**企业调参流程**：

```
Step 1: 基线测试
  使用默认参数，跑测试集，记录指标
  
Step 2: 识别瓶颈
  Token 超了？→ 调小 window_size 或启用压缩
  历史丢失？→ 调大 window_size 或换分层记忆
  成本高？→ 启用缓存、调大 summary_trigger
  
Step 3: 单变量实验
  每次只调一个参数，对比效果
  
Step 4: A/B 测试上线
  小流量验证，确认无 regression
  
Step 5: 监控反馈
  上线后持续监控，根据实际反馈微调
```


---

## 八、主流产品记忆机制对比

不同 AI 产品在记忆机制上的设计差异很大，理解这些差异有助于你为自己的系统做选型。

### 8.1 Claude Code

**Claude Code** 是 Anthropic 推出的 CLI 编程助手，其记忆机制设计非常独特：

| 维度 | 设计特点 |
|------|----------|
| **上下文窗口** | 200K tokens（Claude 3.5 Sonnet），目前业界最大之一 |
| **工作区感知** | 自动读取项目中的 `.claude.md`、`CLAUDE.md` 等文件作为**持久化系统提示** |
| **文件系统记忆** | 直接操作本地文件系统，代码修改本身就是"记忆" |
| **会话记忆** | 单会话内全量保留，超长对话靠大窗口硬撑 |
| **跨会话** | 无内置跨会话用户记忆，每次启动都是新会话 |
| **显式记忆文件** | 用户可创建 `CLAUDE.md` 存放项目规范、偏好，Claude 每次自动读取 |

```
Claude Code 的记忆模式:

启动时 ──► 读取 CLAUDE.md（如果有）──► 加载到 System Prompt
              │
              ▼ 贯穿整个会话的"持久化偏好"
              
用户: "请用 TypeScript 写这个函数"
Claude: 已读取 CLAUDE.md 中的 "项目使用 TypeScript + strict mode"
        直接按规范生成代码

【会话结束】→ 记忆消失（除了修改过的文件）
【新会话】  → 重新读取 CLAUDE.md
```

**Claude Code 的启示**：
- **文件即记忆**：用项目内的 Markdown 文件作为"外化记忆"，比数据库更透明、更可版本控制
- **大窗口策略**：靠 200K 窗口减少压缩需求，简单但有效（成本较高）
- **显式优于隐式**：用户主动写 `CLAUDE.md`，而不是 AI 偷偷记住

---

### 8.2 ChatGPT

**ChatGPT**（OpenAI）的记忆机制经历了多次迭代，目前采用**混合记忆架构**：

| 维度 | 设计特点 |
|------|----------|
| **上下文窗口** | GPT-4o: 128K tokens；GPT-4o-mini: 128K |
| **短期记忆** | 单会话内全量保留，靠窗口硬撑 |
| **长期记忆（Memory）** | 2024 年推出的 explicit memory 功能 |
| **记忆触发** | 用户说"请记住..."或 AI 自动提取关键信息 |
| **记忆存储** | 存储在云端的用户级记忆池（跨会话共享） |
| **记忆控制** | 用户可在设置中查看、编辑、删除记忆 |
| **临时对话** | 支持"临时会话"模式，不保存任何记忆 |

```
ChatGPT 的记忆模式:

用户: "请记住，我叫张三，我是做 Java 的"
ChatGPT: "好的，我记住了。"
        │
        ▼ 存入用户级记忆池（跨会话持久化）
        
【一周后，新会话】
用户: "帮我写个代码"
ChatGPT: （自动读取记忆）"好的张三，基于你 Java 的背景，
         我用 Java 给你写..."

用户可在设置中看到：
  - 张三做 Java
  - 喜欢简洁的代码风格
  - 公司名称: ABC科技
  [编辑] [删除]
```

**ChatGPT 的记忆设计亮点**：
- **用户可控**：记忆对用户完全透明，可随时查看和删除（解决隐私焦虑）
- **自动+手动**：既支持用户显式说"请记住"，也支持 AI 自动提取
- **跨模型共享**：记忆在不同模型（GPT-4/GPT-4o）间共享
- **临时模式**：敏感对话可以不开记忆

---

### 8.3 Hermes

**Hermes** 通常指 Nous Research 发布的开源模型系列（如 Hermes 3），以及围绕它构建的**开源 Agent 记忆方案**：

| 维度 | 设计特点 |
|------|----------|
| **开源性质** | 模型和记忆方案都开源，可自托管 |
| **记忆架构** | 社区通常配合 MemGPT / Zep 等开源记忆框架使用 |
| **MemGPT 集成** | 虚拟上下文管理，把"大磁盘"当"小内存"用 |
| **分层存储** | 显存（快速）→ 主存（中速）→ 磁盘（慢速）三级 |
| **OS 类比** | 像操作系统的虚拟内存：把超长的上下文分页到磁盘 |

```
MemGPT / Hermes 的分层记忆模式（OS 类比）:

┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   显存      │───►│   主存      │───►│   磁盘      │
│ (4K-8K)     │    │ (32K-64K)   │    │ (无上限)    │
│             │    │             │    │             │
│ 当前对话    │    │ 近期摘要    │    │ 历史归档    │
│ 系统提示    │    │ 关键实体    │    │ 完整历史    │
└─────────────┘    └─────────────┘    └─────────────┘
       ▲                  ▲                  │
       │                  │                  │
       └──────────────────┴──────────────────┘
              分页调度（Page Fault 机制）
              需要时从磁盘加载到主存/显存

当用户提到"上周说的那个方案"：
  1. 显存中没有 → Page Fault
  2. 去主存查摘要 → 命中！
  3. 根据摘要定位磁盘中的详细记录
  4. 加载到显存，继续对话
```

**Hermes + MemGPT 的启示**：
- **虚拟内存思想**：把无限存储当有限内存用，分页加载
- **三级缓存**：显存(快/小) → 主存(中/中) → 磁盘(慢/大)
- **开源可定制**：所有代码可见，可深度定制记忆策略

---

### 8.4 开源记忆框架（以 OpenClaw 为例）

> OpenClaw 代表了一类**开源多 Agent 记忆协调框架**的设计思路。以下介绍基于开源记忆框架的通用模式。

| 维度 | 设计特点 |
|------|----------|
| **架构** | 分布式记忆存储，支持多 Agent 共享和隔离 |
| **存储后端** | 插件化：支持 Redis、PostgreSQL、pgvector、Milvus 等 |
| **记忆类型** | 程序性记忆（技能）、陈述性记忆（事实）、情节性记忆（事件） |
| **Agent 间共享** | 支持记忆在多个 Agent 间共享或隔离 |
| **记忆权限** | 可设置记忆的可见范围（全局/团队/个人） |

```
OpenClaw 类框架的记忆架构:

┌─────────────────────────────────────────────────────────────┐
│                     记忆协调层                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Agent A    │  │  Agent B    │  │  Agent C    │        │
│  │  (客服)      │  │  (技术)      │  │  (销售)      │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │               │
│         └────────────────┼────────────────┘               │
│                          │                                 │
│  ┌───────────────────────┴───────────────────────┐        │
│  │              共享记忆区（全局知识库）            │        │
│  │   - 产品手册、公司政策、常见问题               │        │
│  └───────────────────────┬───────────────────────┘        │
│                          │                                 │
│  ┌───────────────────────┴───────────────────────┐        │
│  │              团队记忆区（部门知识）              │        │
│  │   - 技术方案、销售话术、客服脚本               │        │
│  └───────────────────────┬───────────────────────┘        │
│                          │                                 │
│  ┌───────────────────────┴───────────────────────┐        │
│  │              个人记忆区（用户画像）              │        │
│  │   - 用户偏好、历史对话、个人事实               │        │
│  └───────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

**开源框架的核心组件**：

```python
# 概念代码：开源记忆框架的典型接口
class OpenMemoryFramework:
    """开源记忆框架核心接口"""
    
    def __init__(self, storage_backend="pgvector"):
        self.storage = self._init_storage(storage_backend)
        self.agents = {}
    
    def create_agent_memory(self, agent_id: str, scope: str = "personal"):
        """为 Agent 创建记忆空间"""
        self.agents[agent_id] = {
            "scope": scope,  # personal / team / global
            "memory_store": self.storage.create_collection(agent_id)
        }
    
    def write_memory(self, agent_id: str, content: str, 
                     memory_type: str = "episodic",
                     visibility: str = "private"):
        """写入记忆"""
        doc = Document(
            page_content=content,
            metadata={
                "agent_id": agent_id,
                "type": memory_type,      # episodic / semantic / procedural
                "visibility": visibility,  # private / shared / global
                "timestamp": time.time(),
            }
        )
        self.agents[agent_id]["memory_store"].add_documents([doc])
    
    def read_memory(self, agent_id: str, query: str, 
                    scope_filter: List[str] = None) -> List[Document]:
        """读取记忆（自动处理权限和范围）"""
        # 1. 查个人记忆
        results = self._search_agent_memory(agent_id, query)
        
        # 2. 查团队共享记忆（如果有权限）
        if scope_filter and "team" in scope_filter:
            team_results = self._search_team_memory(agent_id, query)
            results.extend(team_results)
        
        # 3. 查全局知识
        if scope_filter and "global" in scope_filter:
            global_results = self._search_global_memory(query)
            results.extend(global_results)
        
        return sorted(results, key=lambda x: x.metadata["score"], reverse=True)
    
    def share_memory(self, from_agent: str, to_agent: str, memory_id: str):
        """Agent 间共享记忆"""
        # 复制记忆到目标 Agent 的空间
        pass
```

---

### 8.5 四者对比总结

| 对比维度 | Claude Code | ChatGPT | Hermes (+MemGPT) | 开源框架 (OpenClaw) |
|----------|-------------|---------|------------------|---------------------|
| **定位** | 编程助手 CLI | 通用对话 | 开源模型+框架 | 多 Agent 协调框架 |
| **上下文窗口** | 200K | 128K | 依模型（4K~128K） | 依配置（可扩展） |
| **长期记忆** | ❌ 无（文件替代） | ✅ 云端记忆池 | ✅ MemGPT 虚拟内存 | ✅ 插件化存储 |
| **记忆触发** | 无（靠文件） | 自动+显式 | 系统级分页 | 程序化控制 |
| **用户可控性** | 完全（自己写文件） | 高（可查看删除） | 中（可配置） | 高（代码可控） |
| **跨会话** | ❌ | ✅ | ✅ | ✅ |
| **多 Agent** | ❌ | ❌ | 部分支持 | ✅ 核心特性 |
| **隐私** | 本地（不上云） | 云端存储 | 自托管 | 自托管 |
| **成本模式** | 高（大窗口API） | 订阅制 | 算力成本 | 基础设施成本 |
| **最佳场景** | 代码开发 | 个人助手 | 研究/定制 | 企业多Agent系统 |

**选型建议**：

- **个人开发者做代码助手** → 学 Claude Code，用 `CLAUDE.md` / `AGENTS.md` 模式
- **做 C 端 AI 助手产品** → 学 ChatGPT，透明可控的用户记忆
- **做深度定制研究** → 用 Hermes + MemGPT，完全掌控记忆机制
- **做多 Agent 企业系统** → 用开源框架模式，分层权限+共享隔离

---

## 九、Agent 自我进化

### 9.1 什么是自我进化？

**Agent 自我进化** = **AI 在使用过程中不断学习和改进自己**。

传统软件：程序员写代码 → 发布 → 用户使用 → 反馈给程序员 → 程序员改代码 → 再发布  
自进化 Agent：用户使用 → Agent 自己发现问题 → Agent 自己调整策略 → 立即生效

```
传统 RAG:
  用户: "这个问题答错了"
  开发者: （收到反馈，改代码，等下周发布）
  
自进化 Agent:
  用户: "这个问题答错了"
  Agent: （自动分析原因，调整检索策略，下次答对）
```

**自我进化的三个层次**：

| 层次 | 名称 | 能力 | 例子 |
|------|------|------|------|
| L1 | **参数调优** | 自动调整超参数 | 发现 Top-K=3 不够，自动调到 5 |
| L2 | **策略优化** | 自动改进决策策略 | 发现某类问题用 BM25 更好，自动切换 |
| L3 | **结构进化** | 自动修改自身结构 | 发现需要新工具，自动创建并集成 |

### 9.2 实现方案

#### L1: 参数自调优

```python
class SelfTuningRAG:
    """能自动调整参数的 RAG 系统"""
    
    def __init__(self):
        self.params = {
            "top_k": 5,
            "temperature": 0.1,
            "chunk_size": 500,
            "rerank_enabled": True,
        }
        self.performance_log = []
    
    def query(self, question: str) -> str:
        # 使用当前参数执行 RAG
        answer = self._rag_with_params(question, self.params)
        
        # 记录性能（需要用户反馈或自动评估）
        self.performance_log.append({
            "question": question,
            "params": self.params.copy(),
            "answer": answer,
            # 实际运行中这里会记录用户满意度或自动评估分数
        })
        
        return answer
    
    def tune_parameters(self, feedback_batch: List[dict]):
        """
        根据反馈批量调优
        feedback_batch: [{"question": "...", "rating": 1-5, "params": {...}}]
        """
        # 分析：什么参数组合效果最好？
        best_configs = self._analyze_best_configs(feedback_batch)
        
        # 应用最优参数
        if best_configs:
            self.params.update(best_configs)
            print(f"[Self-Tune] Updated params: {self.params}")
    
    def _analyze_best_configs(self, feedbacks: List[dict]) -> dict:
        """分析最佳参数配置"""
        from collections import defaultdict
        
        # 按参数组合分组，计算平均评分
        config_scores = defaultdict(list)
        for f in feedbacks:
            key = tuple(sorted(f["params"].items()))
            config_scores[key].append(f["rating"])
        
        # 找平均评分最高的配置
        best_config = max(config_scores.items(), 
                         key=lambda x: sum(x[1])/len(x[1]))
        
        return dict(best_config[0])
```

#### L2: 策略自优化

```python
class SelfImprovingAgent:
    """能自动改进策略的 Agent"""
    
    def __init__(self):
        self.strategies = {
            "default": DefaultStrategy(),
            "bm25_heavy": BM25HeavyStrategy(),
            "semantic_only": SemanticOnlyStrategy(),
            "hyde": HyDEStrategy(),
        }
        self.current_strategy = "default"
        self.strategy_performance = {name: [] for name in self.strategies}
    
    def query(self, question: str) -> str:
        # 选择当前策略
        strategy = self.strategies[self.current_strategy]
        answer = strategy.execute(question)
        return answer
    
    def analyze_and_improve(self, question_type: str, success: bool):
        """
        分析问题类型和策略效果的关系
        """
        # 记录哪种策略对哪种问题类型效果好
        if question_type not in self.strategy_effectiveness:
            self.strategy_effectiveness[question_type] = defaultdict(list)
        
        self.strategy_effectiveness[question_type][self.current_strategy].append(success)
        
        # 自适应选择：下次遇到同类问题，用成功率最高的策略
    
    def select_strategy(self, question: str) -> str:
        """根据问题特征自动选择最佳策略"""
        # 1. 判断问题类型
        q_type = self._classify_question(question)
        
        # 2. 查找该类型下表现最好的策略
        if q_type in self.strategy_effectiveness:
            strategy_scores = {}
            for strat, results in self.strategy_effectiveness[q_type].items():
                strategy_scores[strat] = sum(results) / len(results)
            
            best = max(strategy_scores, key=strategy_scores.get)
            if strategy_scores[best] > strategy_scores.get(self.current_strategy, 0) + 0.1:
                print(f"[Strategy Switch] {self.current_strategy} -> {best} for {q_type}")
                self.current_strategy = best
        
        return self.current_strategy
```

#### L3: 结构自进化

```python
class SelfEvolvingAgent:
    """
    能自动扩展自身能力的 Agent
    当发现现有工具/策略无法解决问题时，尝试创建新工具
    """
    
    def __init__(self):
        self.tools = {}  # 可用工具
        self.skill_library = []  # 技能库
    
    def execute(self, task: str) -> str:
        # 1. 尝试用现有工具解决
        for tool_name, tool in self.tools.items():
            if tool.can_handle(task):
                return tool.run(task)
        
        # 2. 无法解决 → 尝试学习新技能
        print(f"[Learn] No tool can handle: {task}")
        new_skill = self._learn_new_skill(task)
        
        if new_skill:
            self.tools[new_skill.name] = new_skill
            return new_skill.run(task)
        
        return "我无法完成这个任务，但我已记录以便未来学习。"
    
    def _learn_new_skill(self, task: str) -> Optional[Tool]:
        """
        学习新技能的过程：
        1. 分析任务需要什么能力
        2. 查看是否能组合现有技能
        3. 如果不能，生成新代码/提示词
        """
        # 分析需求
        analysis = self.llm.invoke(f"分析以下任务需要什么能力：{task}")
        
        # 尝试用代码生成新工具（需要沙箱执行环境）
        tool_code = self.llm.invoke(f"""
基于以下分析，生成一个 Python 函数来解决这类问题：
{analysis}

要求：
- 函数签名: def solve_{task_type}(input: str) -> str
- 包含输入验证
- 包含错误处理
- 添加文档字符串

只输出函数代码，不要解释：
""")
        
        # 在沙箱中验证新工具
        if self._validate_tool(tool_code):
            return self._create_tool_from_code(tool_code)
        
        return None
    
    def reflect_and_improve(self):
        """
        反思：定期回顾自己的表现，生成改进建议
        """
        recent_tasks = self.get_recent_tasks(n=50)
        
        reflection_prompt = f"""回顾以下近期任务执行情况，总结改进建议：

任务记录：
{format_tasks(recent_tasks)}

请回答：
1. 哪类任务经常失败？原因是什么？
2. 现有的哪个工具/策略需要优化？
3. 需要什么新能力？

输出改进计划（JSON格式）：
{{"improvements": [...], "new_skills_needed": [...]}}
"""
        
        improvement_plan = self.llm.invoke(reflection_prompt)
        self._apply_improvements(improvement_plan)
```

### 9.3 LangGraph 中的自我进化实现

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class EvolvingState(TypedDict):
    """支持自我进化的 State"""
    messages: List
    answer: str
    user_feedback: int  # 1-5 评分
    
    # 进化相关
    strategy_history: List[dict]  # 策略使用记录
    param_history: List[dict]     # 参数使用记录
    current_strategy: str
    current_params: dict
    
    # 反思
    reflection_notes: str
    improvement_plan: dict

# 节点1: 执行（使用当前最佳策略）
def execute(state: EvolvingState):
    strategy = state["current_strategy"]
    params = state["current_params"]
    
    # 执行 RAG
    answer = rag_with_strategy(state["messages"][-1].content, strategy, params)
    
    return {"answer": answer}

# 节点2: 收集反馈
def collect_feedback(state: EvolvingState):
    # 实际系统中从 UI 获取
    # 这里模拟
    return {"user_feedback": 4}  # 假设用户给了4分

# 节点3: 反思（每 N 次查询触发一次）
def reflect(state: EvolvingState):
    history = state["strategy_history"]
    
    if len(history) < 10:
        return {"reflection_notes": "数据不足，暂不反思"}
    
    # 分析哪种策略效果好
    prompt = f"""基于以下策略使用记录，给出优化建议：
{history}

当前策略: {state['current_strategy']}
当前参数: {state['current_params']}

建议："""
    
    reflection = llm.invoke(prompt).content
    
    # 解析改进计划
    improvement = parse_improvement(reflection)
    
    return {
        "reflection_notes": reflection,
        "improvement_plan": improvement,
        "current_strategy": improvement.get("recommended_strategy", state["current_strategy"]),
        "current_params": {**state["current_params"], **improvement.get("param_adjustments", {})}
    }

# 节点4: 执行改进
def apply_improvement(state: EvolvingState):
    plan = state["improvement_plan"]
    
    # 实际执行改进（如调整参数、切换策略）
    if "new_tool_needed" in plan:
        # 创建新工具
        pass
    
    return {}

# 构建进化循环
workflow = StateGraph(EvolvingState)
workflow.add_node("execute", execute)
workflow.add_node("collect_feedback", collect_feedback)
workflow.add_node("reflect", reflect)
workflow.add_node("apply_improvement", apply_improvement)

workflow.set_entry_point("execute")
workflow.add_edge("execute", "collect_feedback")
workflow.add_conditional_edges(
    "collect_feedback",
    lambda s: "reflect" if len(s["strategy_history"]) % 10 == 0 else END,
    {"reflect": "reflect", END: END}
)
workflow.add_edge("reflect", "apply_improvement")
workflow.add_edge("apply_improvement", END)

evolving_agent = workflow.compile()
```

---

## 十、完整实战代码

```python
# memory_system.py - 企业级记忆系统完整实现
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import time
import hashlib
import json
from collections import defaultdict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import PGVector
import numpy as np


@dataclass
class MemoryConfig:
    """记忆系统配置"""
    # 短期记忆
    window_type: str = "sliding"
    window_size: int = 10
    token_budget: int = 4000
    
    # 压缩
    compression_enabled: bool = True
    compression_strategy: str = "hierarchical"
    summary_trigger_tokens: int = 6000
    retain_recent_turns: int = 4
    
    # 长期记忆
    ltm_enabled: bool = True
    connection_string: str = "postgresql+psycopg://user:password@localhost:5432/memory_db"
    
    # 缓存
    cache_enabled: bool = True
    cache_ttl: int = 3600


class HierarchicalMemorySystem:
    """
    企业级分层记忆系统
    整合：短期记忆 + 分层压缩 + 长期记忆 + 缓存
    """
    
    def __init__(self, user_id: str, config: MemoryConfig = None):
        self.user_id = user_id
        self.config = config or MemoryConfig()
        
        # LLM
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.embedding = OpenAIEmbeddings()
        
        # Layer 1: 原始消息
        self.raw_messages = []
        self.raw_limit = self.config.retain_recent_turns * 2
        
        # Layer 2: 本轮摘要
        self.session_summary = ""
        
        # Layer 3: 主题实体
        self.topics = {}
        
        # Layer 4: 长期向量记忆
        if self.config.ltm_enabled:
            self.vector_store = PGVector(
                collection_name=f"ltm_{user_id}",
                embedding_function=self.embedding,
                connection_string=self.config.connection_string,
            )
        else:
            self.vector_store = None
        
        # 缓存
        self.response_cache = {}
        
        # 统计
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "summarizations": 0,
        }
    
    def chat(self, user_message: str) -> str:
        """
        主入口：处理用户消息并返回回复
        """
        self.stats["total_queries"] += 1
        
        # 1. 检查缓存
        if self.config.cache_enabled:
            cached = self._check_cache(user_message)
            if cached:
                self.stats["cache_hits"] += 1
                return cached
        
        # 2. 添加到原始消息
        self.raw_messages.append(("user", user_message))
        
        # 3. 检查是否需要压缩
        if self.config.compression_enabled:
            self._maybe_compress()
        
        # 4. 构建上下文
        context = self._build_context(user_message)
        
        # 5. 调用 LLM
        messages = [SystemMessage(content=context)]
        messages += [HumanMessage(content=m) if r == "user" else AIMessage(content=m)
                     for r, m in self.raw_messages]
        
        response = self.llm.invoke(messages)
        answer = response.content
        
        # 6. 添加 AI 回复到原始消息
        self.raw_messages.append(("assistant", answer))
        
        # 7. 更新长期记忆（异步或定期）
        if self.vector_store:
            self._update_ltm(user_message, answer)
        
        # 8. 写入缓存
        if self.config.cache_enabled:
            self._set_cache(user_message, answer)
        
        return answer
    
    def _build_context(self, current_query: str) -> str:
        """构建完整的上下文 Prompt"""
        parts = []
        
        # Layer 4: 长期记忆检索
        if self.vector_store:
            relevant = self.vector_store.similarity_search(current_query, k=3)
            if relevant:
                parts.append("[历史记忆]\n" + "\n".join([d.page_content for d in relevant]))
        
        # Layer 3: 主题实体
        if self.topics:
            parts.append(f"[用户画像] {json.dumps(self.topics, ensure_ascii=False)}")
        
        # Layer 2: 本轮摘要
        if self.session_summary:
            parts.append(f"[本轮摘要] {self.session_summary}")
        
        return "\n\n".join(parts)
    
    def _maybe_compress(self):
        """检查并执行压缩"""
        total_chars = sum(len(m) for _, m in self.raw_messages)
        
        if total_chars > self.config.summary_trigger_tokens:
            self._compress_memory()
    
    def _compress_memory(self):
        """执行分层压缩"""
        # 保留最近 N 轮
        recent = self.raw_messages[-self.raw_limit:]
        to_compress = self.raw_messages[:-self.raw_limit]
        
        if not to_compress:
            return
        
        # 生成摘要
        text = "\n".join([f"{r}: {m}" for r, m in to_compress])
        prompt = f"""将以下对话合并进现有摘要，保留关键事实：
现有摘要：{self.session_summary}
新对话：{text}
更新后摘要："""
        
        self.session_summary = self.llm.invoke(prompt).content.strip()
        self.raw_messages = recent
        self.stats["summarizations"] += 1
        
        # 提取实体
        self._extract_topics(to_compress)
    
    def _extract_topics(self, messages):
        """提取主题实体"""
        text = "\n".join([f"{r}: {m}" for r, m in messages])
        prompt = f"""提取关键信息（JSON格式）：
{{"姓名":"","职业":"","偏好":[],"重要事件":[]}}

对话：{text}"""
        
        try:
            extracted = json.loads(self.llm.invoke(prompt).content)
            self.topics.update(extracted)
        except:
            pass
    
    def _update_ltm(self, user_msg: str, ai_msg: str):
        """更新长期记忆"""
        # 存储对话摘要
        doc = Document(
            page_content=f"用户: {user_msg}\nAI: {ai_msg}",
            metadata={
                "user_id": self.user_id,
                "timestamp": datetime.now().isoformat(),
            }
        )
        self.vector_store.add_documents([doc])
    
    def _check_cache(self, query: str) -> Optional[str]:
        """检查缓存"""
        key = hashlib.md5(query.encode()).hexdigest()
        if key in self.response_cache:
            result, timestamp = self.response_cache[key]
            if time.time() - timestamp < self.config.cache_ttl:
                return result
        return None
    
    def _set_cache(self, query: str, answer: str):
        """写入缓存"""
        key = hashlib.md5(query.encode()).hexdigest()
        self.response_cache[key] = (answer, time.time())
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            **self.stats,
            "cache_hit_rate": self.stats["cache_hits"] / max(self.stats["total_queries"], 1),
            "raw_messages_count": len(self.raw_messages),
            "topics_count": len(self.topics),
            "summary_length": len(self.session_summary),
        }


# ========== 使用示例 ==========
if __name__ == "__main__":
    config = MemoryConfig(
        window_type="sliding",
        window_size=10,
        compression_enabled=True,
        ltm_enabled=True,
        cache_enabled=True,
    )
    
    memory_system = HierarchicalMemorySystem(user_id="user_001", config=config)
    
    # 模拟对话
    print(memory_system.chat("你好，我叫张三，做Java的"))
    print(memory_system.chat("我喜欢吃川菜"))
    print(memory_system.chat("记住我的生日是3月15日"))
    print(memory_system.chat("我叫什么名字？"))
    print(memory_system.chat("我喜欢吃什么？"))
    
    print("\n[Stats]", memory_system.get_stats())
```

---

## 附录：记忆机制全景图

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                        AI 记忆机制全景图                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  用户输入                                                                     ║
║      │                                                                       ║
║      ▼                                                                       ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                         缓存层（Cache）                              │    ║
║  │  ├── L1 内存缓存（相同 query 直接返回）                               │    ║
║  │  └── L2 磁盘缓存（TTL 过期后重新计算）                                │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║      │ 缓存未命中                                                            ║
║      ▼                                                                       ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                      短期记忆（Short-Term）                           │    ║
║  │                                                                     │    ║
║  │  原始消息层（Layer 1）                                               │    ║
║  │  ├── 保留最近 K 轮完整对话                                           │    ║
║  │  └── 溢出时触发压缩                                                  │    ║
║  │                                                                     │    ║
║  │  摘要层（Layer 2）                                                   │    ║
║  │  ├── 历史对话压缩成摘要                                              │    ║
║  │  └── 增量更新（不重复处理旧内容）                                     │    ║
║  │                                                                     │    ║
║  │  压缩策略选择：                                                       │    ║
║  │  ├── Sliding Window（滑动窗口）                                      │    ║
║  │  ├── Token Budget（预算截断）                                        │    ║
║  │  ├── Summarization（LLM摘要）                                        │    ║
║  │  ├── Semantic（语义去重）                                            │    ║
║  │  └── Hierarchical（分层组合）                                        │    ║
║  │                                                                     │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║      │                                                                       ║
║      ▼                                                                       ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                      长期记忆（Long-Term）                            │    ║
║  │                                                                     │    ║
║  │  主题实体层（Layer 3）                                               │    ║
║  │  ├── 用户画像（姓名、职业、偏好）                                     │    ║
║  │  ├── 提取方式：显式存储 + 隐式提取                                    │    ║
║  │  └── 触发条件：置信度阈值 / 用户确认                                  │    ║
║  │                                                                     │    ║
║  │  向量记忆层（Layer 4）                                               │    ║
║  │  ├── 向量数据库存储历史对话和事实                                     │    ║
║  │  ├── 语义检索召回相关记忆                                            │    ║
║  │  └── 定期巩固（合并/删除/升级）                                      │    ║
║  │                                                                     │    ║
║  │  存储后端：                                                           │    ║
║  │  ├── Redis（结构化事实，KV 存储）                                     │    ║
║  │  ├── pgvector/Milvus（语义记忆，向量存储）                            │    ║
║  │  └── PostgreSQL（复杂关系，关系存储）                                 │    ║
║  │                                                                     │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║      │                                                                       ║
║      ▼                                                                       ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                      Prompt 组装                                      │    ║
║  │                                                                     │    ║
║  │  系统指令 > 长期记忆(Top-K) > 主题实体 > 摘要 > 最近原始消息           │    ║
║  │                                                                     │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║      │                                                                       ║
║      ▼                                                                       ║
║  LLM 生成回复                                                                 ║
║      │                                                                       ║
║      ▼                                                                       ║
║  更新记忆（原始消息 → 检查压缩 → 更新长期记忆）                                ║
║      │                                                                       ║
║      ▼                                                                       ║
║  评估与监控                                                                    ║
║  ├── 离线：Recall@K / Faithfulness / RAGAS                                ║
║  └── 在线：延迟 / Token消耗 / 用户满意度                                   ║
║      │                                                                       ║
║      ▼                                                                       ║
║  自我进化（可选）                                                              ║
║  ├── L1 参数自调优                                                           ║
║  ├── L2 策略自优化                                                           ║
║  └── L3 结构自进化                                                           ║
║                                                                              ║
╚═════════════════════════════════════════════════════════════════════════════╝
```

---

> **写在最后**
>
> AI 的记忆机制不是"一个功能"，而是**一整套系统工程**。从纳秒级的 KV Cache，到毫秒级的对话窗口，再到秒级的长期向量存储，每一层都有它的位置和职责。
>
> 企业落地的核心原则：
> 1. **简单优先**：先用 Sliding Window + 全量保留，验证业务价值
> 2. **按需压缩**：窗口不够了再上摘要，摘要不够了再上分层
> 3. **用户可控**：长期记忆对用户透明，可查看、可删除、可导出
> 4. **监控必建**：没有评估的记忆系统就是黑盒，无法优化
> 5. **缓存必加**：相同的计算不要重复做，省钱又提速
>
> **记忆让 AI 从"工具"变成"伙伴"。** 没有记忆，每次对话都是初见；有了记忆，AI 才能"记得你的名字、懂你的偏好、接上你的话"。
