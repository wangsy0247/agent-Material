# RAG Agent 项目技术栈分析与面试准备

## 一、项目中用到的技术栈

### 基础框架与语言

| 技术 | 用途 |
|------|------|
| **Python 3.10+** | 全面使用类型注解、`str \| None` 新语法 |
| **FastAPI** | 主项目后端框架（`api_server.py`） |
| **asyncio** | 异步并发：Agent 流式推理、Embedding 批量请求、后台 Worker |

### LLM & AI 核心

| 技术 | 用途 |
|------|------|
| **LangChain** | 全栈：PromptTemplate、Document、BaseRetriever、EnsembleRetriever、ContextualCompressionRetriever、StrOutputParser、LCEL chain（`|` 管道语法） |
| **LangGraph** | ReAct Agent 状态机：StateGraph 显式节点编排、ToolNode、Checkpointer 持久化、条件边路由、HITL 人工审批（`interrupt` / `Command(resume=)`） |
| **ChatOpenAI** → 千问 qwen3.6-plus | Chat 模型，支持 `with_structured_output()` 实现查询路由的可靠分类 |
| **DashScope text-embedding-v4** | Embedding 模型，自研 `AsyncDashScopeEmbeddings` 异步批量封装（分组并发 + 速率限制 + 指数退避重试） |
| **MCP (Model Context Protocol)** | FastMCP 搭建 RAG Server 提供 8 个 Tool、langchain-mcp-adapters 客户端动态发现连接 |

### RAG 检索系统

| 技术 | 用途 |
|------|------|
| **PGVector** (PostgreSQL + pgvector + psycopg) | 向量存储，cosine distance 检索 |
| **BM25** (rank-bm25) | 稀疏关键词检索，内存索引 |
| **EnsembleRetriever (RRF)** | 混合检索融合（向量 0.5 + BM25 0.5） |
| **CrossEncoder Reranker** (Qwen3-Reranker-0.6B / HuggingFaceCrossEncoder) | 对 Top-K 候选做精排重排序 |
| **ContextualCompressionRetriever** | 重排序管道封装 |
| **TopKRetriever** | 截断包装器，减少进入 CrossEncoder 的候选数 |
| **VectorSimilarityThresholdRetriever** | 自定义 PGVector 检索器，显式做 cosine distance → similarity 转换，支持 HyDE embedding 注入 |
| **多粒度索引**（子 chunk 检索 → 父 chunk 上下文） | 两阶段：small collection 混合检索 → 收集 parent_id → large collection 取回父 chunk，提升长文档召回完整性 |
| **4 种查询改写策略** | expand（查询扩展）/ HyDE（假设文档生成）/ step_back（退一步抽象）/ decompose（问题分解），支持 `+` 流水线组合 |

### 数据工程

| 技术 | 用途 |
|------|------|
| **文档加载** | PDF / TXT / Markdown / HTML / DOCX / PPTX 多格式支持，含 MinerU PDF 解析器 |
| **HTML / Markdown Cleaner** | 文本清洗去噪 |
| **多策略切分** | RecursiveSplitter / SemanticChunker / MarkdownHeaderSplitter / HTMLHeaderSplitter / ParentChunkSplitter |
| **混合切分管道** (`extension_pipeline`) | 按文件扩展名配置切分流水线，链式执行多个 Splitter |
| **MetadataEnhancer** | chunk 元数据增强（语言检测 / 类型标注 / 长度统计） |

### 记忆系统（三层）

| 层 | 技术 | 职责 |
|----|------|------|
| **L1 短期记忆** | 内存 Dict（按 user_id + session_id 隔离） | Token 预算感知的上下文窗口 |
| **L2 中期记忆** | SQLite（WAL 模式，按 user_id 隔离） | 对话消息持久化、会话生命周期管理 |
| **L3 长期记忆** | PGVector 向量检索 + 文件存储（USER.md / MEMORY.md） | 跨会话知识沉淀、自动提取与召回 |

### Agent 架构（LangGraph 状态机）

```
entry → memory_node → should_summarize? ─┬→ summarization_node → agent_node
                                         │                            │
                                         └────────────────────────────┘
                                                                      │
                                                              should_continue?
                                          ┌───────────────────────────┼───────────────────┐
                                          │                           │                   │
                                   iteration > 10              有 tool_calls         无 tool_calls
                                          │                           │                   │
                                          ▼                           ▼                   ▼
                                      finalize              human_approval_node        finalize
                                                              ┌──────┴──────┐
                                                             批准           拒绝
                                                              │             │
                                                              ▼             ▼
                                                          tool_node ──→ agent_node
```

| 技术 | 用途 |
|------|------|
| **StateGraph** | 显式节点编排，7 个节点 + 条件边 |
| **Checkpointer**（PostgresSaver / MemorySaver 降级） | 对话状态持久化与恢复，thread_id = session_id |
| **HITL**（Human-in-the-Loop） | `interrupt()` 挂起 + `Command(resume=)` 恢复，敏感工具（如 `mp_download`）审批 |
| **ContextCompressor**（四阶段：Smart Collapse + 保护区 + LLM 摘要 + 清洗） | 长对话历史压缩，降低 Token 消耗 |
| **Tool Offload** | 工具返回结果超过 10 万字符自动落盘，防止上下文溢出 |
| **消息清洗**（`_sanitize_messages_for_llm`） | 移除孤立的 ToolMessage，防止损坏的 checkpoint 导致 API 400 错误 |

### 后台任务系统

| 技术 | 用途 |
|------|------|
| **FileTracker**（SQLite + MD5） | 增量文档追踪，只处理变更文件 |
| **TaskQueue**（SQLite） | 任务队列持久化（ingest / delete / reindex） |
| **BackgroundWorker**（daemon thread） | 轮询队列异步消费，支持失败重试 |

### 基础设施

| 技术 | 用途 |
|------|------|
| **PostgreSQL + pgvector** | 向量存储 + LangGraph Checkpointer 状态持久化 |
| **SQLite（WAL 模式）** | 中期记忆 + 文件追踪 + 任务队列 |
| **YAML 配置驱动** | 全配置化（agent / vector_store / prompt / queue / mcp） |
| **MCP 多传输协议** | stdio（子进程） / SSE（HTTP 流式）双模式 |

---

## 二、企业级项目中常见但缺少的技术点

### 1. 可观测性（Observability）

| 缺失项 | 严重程度 | 说明 |
|--------|----------|------|
| **OpenTelemetry / Jaeger 分布式链路追踪** | 🔴 高 | 目前仅 LangSmith 做 LLM 调用追踪，缺乏全链路 tracing。Agent 多轮推理 + MCP 跨进程调用 + RAG 多阶段检索的耗时无法逐段定位。当用户反馈"回答慢"时，无法快速定位是哪一环卡住 |
| **Prometheus + Grafana 指标监控** | 🔴 高 | 无 QPS、检索延迟 P50/P99、Embedding API 错误率、LLM Token 消耗趋势、PGVector 连接池状态、任务队列深度等关键指标 |
| **结构化日志聚合（ELK / Loki）** | 🟡 中 | 日志分散在各组件 stdout/stderr，故障排查需要跨多台机器查日志 |
| **告警规则** | 🔴 高 | 缺：LLM 调用失败率突增告警、PGVector 连接池耗尽告警、任务队列积压告警、Embedding API 限流告警 |
| **LangSmith 深度集成** | 🟡 中 | 已有基础的 `@get_run_config`，但未充分利用 dataset、experiment、annotation queue 等功能 |

### 2. 稳定性与容错

| 缺失项 | 严重程度 | 说明 |
|--------|----------|------|
| **LLM 调用熔断 / 降级（Circuit Breaker）** | 🔴 高 | 千问 API 超时或限流时，没有断路器 → 请求堆积导致线程池耗尽。应引入 `tenacity` + 熔断器模式 |
| **Embedding 降级方案** | 🟡 中 | DashScope 不可用时，缺少本地 ONNX / Sentence-Transformers 模型作为 fallback |
| **PGVector 连接池管理** | 🔴 高 | `PGVector(connection=conn_str)` 每次操作创建新连接，高并发下连接数爆炸。应该用 SQLAlchemy `create_engine` + `QueuePool` |
| **全局限流（Rate Limiting）** | 🟡 中 | MCP Server 层没有 Token Bucket / Sliding Window 限流，可能被高频调用打爆第三方 API 配额 |
| **健康检查端点** | 🟡 中 | 缺少 `/health` 端点检测：PGVector 连通性、Embedding API 可用性、DashScope 认证状态、任务队列积压深度 |
| **优雅关闭（Graceful Shutdown）** | 🟡 中 | Worker daemon 线程随进程直接终止，正在处理的任务丢失；关闭中的 HTTP 请求没有 drain 等待时间 |
| **重试风暴防护** | 🟡 中 | AsyncEmbeddings 有指数退避重试，但任务队列的 `retry_failed` 无最大重试间隔限制，可能在 API 恢复瞬间所有积压任务同时重试造成二次雪崩 |

### 3. 性能优化

| 缺失项 | 严重程度 | 说明 |
|--------|----------|------|
| **pgvector HNSW 索引** | 🟡 中 | pgvector 支持 HNSW 索引（`vector_l2_ops` / `vector_cosine_ops`），相比默认 IVFFlat 在十万级以上向量时有数量级差距 |
| **检索结果缓存（Redis / 内存 LRU）** | 🟡 中 | 相同 query 的高频检索可以走缓存，避免反复调用昂贵的 Embedding API + CrossEncoder 推理 |
| **BM25 索引持久化** | 🟡 中 | 每次重启从 PGVector 全量加载 → 重建 BM25 内存索引，文档量大时启动耗时很长。可以将 BM25 索引序列化到磁盘 |
| **Prompt Caching** | 🟡 中 | DeepSeek / Anthropic 的 prompt cache 未被使用。系统提示和长文档上下文每次都完整发送，Token 浪费严重 |
| **批量 CrossEncoder 推理** | 🟢 低 | CrossEncoder 目前逐对打分，`reranker_batch_size` 配置了但未充分利用，可以用 `model.predict(pairs, batch_size=n)` 批量推理 |
| **向量检索预热** | 🟢 低 | PGVector 冷启动时第一次查询的磁盘读取延迟很高，可以启动时做一次空查询预热 |

### 4. 评估与质量保障

| 缺失项 | 严重程度 | 说明 |
|--------|----------|------|
| **RAGAS 自动化评测** | 🔴 高 | 需要持续评测 Faithfulness（忠实度）、Answer Relevance（答案相关性）、Context Recall（上下文召回率）、Context Precision（上下文精确率） |
| **评测数据集 + CI 回归** | 🔴 高 | 每次代码改动后自动跑 RAG 评测（`pytest + RAGAS`），在 CI 中拦截检索质量回退 |
| **A/B 测试框架** | 🟡 中 | 不同改写策略组合、检索参数的效果对比缺乏量化手段 |
| **用户反馈闭环** | 🟡 中 | 无点赞/踩数据收集，无法用 RLHF 或 DPO 持续优化生成质量 |
| **Golden Dataset 管理** | 🟡 中 | 缺少标注的 QA Pair 数据集用于离线评测 |

### 5. 工程化与 DevOps

| 缺失项 | 严重程度 | 说明 |
|--------|----------|------|
| **Docker / Docker Compose 部署** | 🔴 高 | 无 Dockerfile、docker-compose.yml，生产环境无法标准化交付（PostgreSQL + pgvector + 应用 + MCP Server 的编排） |
| **CI/CD Pipeline** | 🔴 高 | 无 GitHub Actions / GitLab CI 的代码检查（lint / type check）、自动测试、构建部署 |
| **依赖管理** | 🔴 高 | `requirements.txt` 只有一行 `mcp>=1.0.0`，实际依赖散落在各处；无 `poetry.lock` 或完整的 `requirements.txt` |
| **单元测试覆盖率** | 🔴 高 | `tests/` 下仅有少量文件，核心逻辑缺乏单测——特别是检索链装配、改写策略分支、路由分类的边界情况 |
| **API 版本化管理** | 🟢 低 | 无 `/api/v1` vs `/api/v2` 的策略 |
| **配置热更新** | 🟢 低 | 修改 YAML 配置需要重启服务才能生效 |
| **Kubernetes Helm Chart** | 🟡 中 | 无 K8s 部署方案，无法水平扩展 MCP Server |

### 6. 安全

| 缺失项 | 严重程度 | 说明 |
|--------|----------|------|
| **Prompt Injection 防护** | 🟡 中 | 无对抗性输入检测（如 "Ignore previous instructions" 类攻击），文档内容中可能包含注入语句 |
| **API Key 管理** | 🟡 中 | `.env` 裸写，缺少 Vault / Cloud Secrets Manager 集成；日志中可能泄露 Key |
| **多租户数据严格隔离** | 🟡 中 | 向量库按 collection 隔离但无访问控制校验；user_id 隔离仅在记忆层生效，RAG 检索无租户感知 |
| **内容安全审核** | 🟡 中 | 无 LLM 输入/输出的敏感内容过滤（涉政、涉黄、涉暴等合规审核） |
| **Tool 调用沙箱** | 🟡 中 | MCP Tool（如文件下载）直接执行，无权限校验和执行环境隔离 |

---

## 三、面试可能的难点与高分回答思路

### 3.1 检索链路的深度追问

#### Q1: PGVector cosine distance 为什么要手动转 similarity？

**考点**：对底层算法的理解深度。

**问题背景**：PGVector 的 `<=>` 运算符返回 cosine distance（范围 [0, 2]），而 LangChain 内置的 `similarity_score_threshold` 检索器在不同版本中对 score 的处理存在不一致——有的版本直接把 distance 当作 similarity（0.5 的 distance 会被当作 0.5 的 similarity），导致阈值过滤完全失效（所有文档都能通过 0.3 的阈值）。

**你的代码做法**：自定义 `VectorSimilarityThresholdRetriever`，显式做 `similarity = 1 - distance / 2`，再与 `score_threshold` 比较。同时加了 fallback：如果过滤后为空，至少保留 Top-1。

**延伸追问**：为什么是 `1 - distance / 2`？——因为 cosine distance = 1 - cosine similarity，所以 similarity = 1 - distance；但 PGVector 的 cosine distance 定义域是 [0, 2]（归一化向量后），所以需要除以 2 归一化到 [0, 1]。严格来说对于 L2 归一化的向量，`cosine_distance = 1 - cos(θ)`，`similarity = 1 - distance`。

#### Q2: RRF 融合公式是什么？权重怎么调的？

**RRF 公式**：`score(d) = Σ (1 / (k + rank_i(d)))`，其中 `rank_i(d)` 是文档 d 在第 i 个检索器中的排名，k 是平滑常数（默认 60）。

**你的代码做法**：用 `EnsembleRetriever` 的 weights 参数（0.5 / 0.5），这实际上是加权 RRF。

**延伸追问**：为什么不是 0.7/0.3？——需要根据自己的数据集做实验，向量检索在语义相似的长尾问题上更强，BM25 在精确术语匹配上更强。权重取决于知识库的文本特性（术语密集度、同义词丰富度）。

#### Q3: TopKRetriever 截断到 8 / 10，会不会丢失重要候选？

**这是典型的 Recall vs Latency tradeoff**。CrossEncoder（Qwen3-Reranker-0.6B）对每个 (query, doc) pair 做推理，时间复杂度 O(n)。截断的理由是：

1. 混合检索的 RRF 已经做了第一轮粗排，Top-10 通常包含了所有真正相关的文档
2. CrossEncoder 的增益主要体现在 Top-10 内部的精排，对第 50 名和第 5 名的区分意义不大（用户通常不会翻到第 5 页）
3. 延迟预算：0.6B 模型推理 10 个 pair 约 50ms，推理 50 个就需要 250ms+

**但必须坦诚**：如果 RRF 阶段把一个高相关文档排到了第 11 名（刚好被截断），那确实会丢失。可以通过离线评测调整截断阈值。

---

### 3.2 多粒度索引的设计决策

#### Q: 子 chunk 500 + 父 chunk 2000？为什么是这个比例？

- **子 chunk 500**：足够承载一个完整的知识点（通常 3-5 句话），又足够小让 embedding 模型（1024 维）聚焦于单一语义。太小（如 200）会丢失上下文，太大（如 1000）会稀释 embedding 的语义精度
- **父 chunk 2000**：约等于一页 A4 纸的内容，提供完整上下文给 LLM 生成。同时 2000 字符 ≈ 1500 token，在 8192 token 的上下文窗口中可以容纳约 5 个父 chunk
- **比例 4:1**：每个父 chunk 平均包含 4 个子 chunk，保证每个知识点都有精确的向量表达

#### Q: 父 chunk 的排序信息丢失怎么处理？

**这就是你代码中一个值得讨论的 tradeoff**。`_fetch_parents` 流程：

```python
# 1. 小 collection 混合检索 → 返回排序好的子 chunk
child_docs = self._hybrid_search(query)  # 按相似度排序

# 2. 收集唯一的 parent_id（去重，但丢失了排序信息！）
parent_ids = list(dict.fromkeys(doc.metadata.get("parent_id") for doc in child_docs))

# 3. 批量取回父 chunk（SQL 返回顺序不保证与 parent_ids 一致）
parent_docs = self._fetch_parents(parent_ids)
```

**问题**：`_fetch_parents` 使用 `ANY(%s)` 批量查询，PG 返回顺序是任意的，父 chunk 的最终顺序没有保留子 chunk 的检索得分排序。

**可讨论的解决方案**：
- 后排序：取回父 chunk 后，按 `parent_ids` 的原始顺序重排
- 保留得分：在子 chunk 检索时携带 similarity score，按最高得分子 chunk 对应的父 chunk 排序
- 直接不排序：因为交给 CrossEncoder 后会精排，所以原始顺序丢失影响可控

---

### 3.3 查询改写策略的组合设计

#### Q: "step_back + expand" 的流水线执行顺序为什么是这个？换顺序会怎样？

**当前顺序**：step_back 先抽象 → expand 再对抽象后的查询做扩展。

**反过来（expand → step_back）**：
- expand 先生成 3 个 query 变体 → step_back 对每个变体（共 6 个 query）都生成抽象版本 → 最终 12 个 query → 12 路检索 → **计算成本 ×4，但召回增益边际递减**

**设计原则**：先缩小范围再发散（step_back 是收敛的，expand 是发散的），用收敛操作减少后续发散操作的基数。

#### Q: HyDE 如果生成质量差（幻觉），对检索是正向还是负向？

**双面性**：
- 正向场景：查询过于简短（如 "MOF 材料"），HyDE 生成一段专业性强的假设文本，embedding 更接近真实文档 → 召回率提升
- 负向场景：HyDE 生成了错误领域的假设（如把 "MOF" 理解成了游戏术语），embedding 偏离知识库分布 → 检索结果完全跑偏

**你的代码做法**：`enable_hyde: false` 默认关闭，需要用户显式开启。同时 `set_hyde_embedding()` 有 try-finally 保护，确保结束后恢复。

**延伸追问**：如何评估 HyDE 是正向还是负向？——对比实验：同一批 query，with/without HyDE 的 Recall@5，统计显著性检验。

---

### 3.4 MCP 协议的理解

#### Q: MCP 的 stdio transport vs SSE transport 有什么区别？

| | stdio | SSE |
|----|-------|-----|
| 通信方式 | 子进程 stdin/stdout JSON-RPC | HTTP + Server-Sent Events |
| 生命周期 | 与父进程绑定 | 独立进程，可单独扩缩 |
| 并发 | 单连接，请求串行化 | 多连接，天然支持并发 |
| 适用场景 | 本地工具、IDE 插件 | 远程服务、微服务架构 |
| 你的项目 | RAG MCP Server 支持两种，默认 stdio | SSE 模式用于 Agent 通过网络调用 RAG |

**关键追问**：为什么你的 MCP 连接是 per-invocation 的？——注释说 "避免 stale connection 复杂性"，每次 tool call 创建连接 → 用完销毁，牺牲性能换稳定性，避免长连接断线重连时的状态同步问题。

---

### 3.5 LangGraph 状态机设计

#### Q: thread_id 固定为 session_id 有什么隐含问题？

**你的设计**：`configurable={"thread_id": thread_id}`，thread_id = session_id，LangGraph checkpointer 自动管理所有轮次的消息。

**潜在问题**：
1. **并发冲突**：同一 session 同时发两个请求 → 两个 `astream` 操作同一个 thread_id → checkpointer 写入冲突（PostgresSaver 有行级锁，MemorySaver 无保护会丢消息）
2. **HITL 恢复的竞态**：用户点了批准，又快速发了新消息 → 两个请求操作同一个状态
3. **状态无限增长**：长对话的消息列表会无限增加（虽然有 summarization 缓解，但 checkpointer 需要在每次写入时全量序列化 State）

**可讨论的优化**：
- 加分布式锁（Redis Redlock）
- 使用带乐观锁的 checkpointer（version 号比较）
- session_id + sequence_number 作为 thread_id

#### Q: RemoveMessage 和 add_messages 的 reducer 怎么协作？

LangGraph 的 `Annotated[list[AnyMessage], add_messages]` 定义了 messages 状态的 reducer。`add_messages` 函数对 `RemoveMessage` 有特殊处理：根据 `RemoveMessage.id` 查找并删除对应消息。所以：

1. memory_node 返回 `[RemoveMessage(id=old_sys_id), SystemMessage(content=new)]` → reducer 先删除旧 system message，再追加新的
2. summarization_node 返回 `[RemoveMessage(...), RemoveMessage(...), AIMessage(summary)]` → reducer 批量删除旧消息 + 追加压缩后的摘要

**潜在问题**：如果一条消息的 id 在 checkpoint 中不存在（比如已经在上一个 checkpoint 中被删除了），RemoveMessage 会静默忽略还是报错？——取决于 LangGraph 版本，需要测试边界行为。

---

### 3.6 记忆系统的设计

#### Q: 三层记忆的边界是什么？为什么 SQLite 而不是 Redis？

| 层 | 存储 | 延迟 | 容量 | 生命周期 |
|----|------|------|------|----------|
| L1 短期 | 内存 Dict | ~μs | Token 预算（~8K） | 单次请求上下文窗口 |
| L2 中期 | SQLite WAL | ~ms | 无限（磁盘） | 单次会话 |
| L3 长期 | PGVector + 文件 | ~10ms | 无限 | 跨会话持久化 |

**为什么 SQLite 而不是 Redis**：对于单机部署的对话系统，SQLite 在以下方面优于 Redis：
- 零运维：不需要额外进程，WAL 模式并发读已经足够
- 持久化简单：直接写入磁盘文件，不需要配置 RDB/AOF
- SQL 查询灵活：按 user_id 过滤、按时间排序、统计 message_count

**当然 Redis 的优势**：纯内存读写延迟更低（~100μs vs ~1ms）、天然支持 TTL 过期、更好的分布式支持。如果用户量到 1000+ 并发，SQLite 会成为瓶颈。

---

### 3.7 Agent 可靠性与评估

#### Q: Agent 死循环怎么防？除了 iteration_count 还有什么方案？

你的代码用了 `iteration_count > 10` 强制终止。其他方案可讨论：

1. **Token 预算控制**：设置总 Token 消耗上限，超限强制终止（你的 STM 有此机制但 Agent 循环未接入）
2. **语义重复检测**：检测连续两轮的 tool_calls 是否完全一致（Agent 陷入调用→失败→再调用的循环）
3. **时间超时**：单次 Agent 执行超过 120 秒强制终止
4. **Tool 调用前置去重**：缓存最近 N 次 tool call args，相同参数拒绝执行
5. **LLM 级别的 early stopping**：某些模型支持 `stop_reason` 信号

#### Q: 如何评估一个 Agent 的好坏？什么指标最重要？

**离线指标**（自动化）：
- **Task Success Rate**：给定标准任务集，Agent 能否正确完成（需要标注 ground truth）
- **Tool Selection Accuracy**：Agent 选择的工具是否与预期一致
- **Avg. Steps to Completion**：完成任务的平均推理轮次（越少越好）
- **Hallucination Rate**：生成内容中虚构信息的比例
- **RAGAS 四项**：Faithfulness / Answer Relevance / Context Recall / Context Precision

**在线指标**（用户反馈）：
- 用户点赞/踩比例
- 对话完成率（用户没有中途放弃）
- 人工重写率（用户重新组织问题二次提问的比例）
- 平均对话轮次和时长

---

## 四、面试中如何"扬长避短"

### 主动展示的亮点

1. **工程化意识强**：全配置驱动（YAML）、懒加载模式、Lazy-init 模式的全局 Singleton、显式的错误处理和日志
2. **对 LangChain 源码级理解**：自定义 `VectorSimilarityThresholdRetriever`、`object.__setattr__` 绕过 Pydantic 校验补 `.predict()` 方法 → 说明你读过源码，知道它的问题并能修复
3. **多策略组合设计**：查询改写 4 种策略 + 流水线组合，不是简单的"调个 API"
4. **完整的记忆系统**：三层记忆 + ContextCompressor 四阶段压缩，超出大部分应届生项目
5. **MCP 协议的实践落地**：MCP 是 2024-2025 年 AI 工程化的关键趋势

### 坦诚承认的不足（附带改进思路）

1. **评估体系缺失** → "我计划引入 RAGAS 做自动化评测，建立 Golden QA Dataset，在 CI 中回归"
2. **可观测性不足** → "我了解 OpenTelemetry 的 Span/Context Propagation 概念，接下来会接入"
3. **容错机制不完整** → "我了解 Circuit Breaker 和 Rate Limiter 的设计模式，可以用 tenacity 库快速实现"
4. **部署方案欠缺** → "我有 Docker 基础，计划写 Docker Compose 把 PostgreSQL + 应用 + MCP Server 容器化"

### 一个"杀伤力"强的主动展示

如果你能在面试中主动画出下面这个完整架构图并解释每层的设计决策，会非常有说服力：

```
┌─────────────────────────────────────────────────────────────────┐
│                        MCP Client (Agent)                        │
│  ReactAgent → LangGraph StateGraph → Checkpointer (Postgres)     │
│     │                                                             │
│     ├── tools: [native_tools, MCP tools (RAG Server 8 tools)]    │
│     ├── memory: L1(内存) → L2(SQLite) → L3(PGVector + 文件)      │
│     └── HITL: interrupt() / Command(resume=)                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │ SSE / stdio
┌──────────────────────────▼──────────────────────────────────────┐
│                     RAG MCP Server (FastMCP)                      │
│                                                                    │
│  ┌──────────────┐  ┌─────────────────┐  ┌──────────────────┐     │
│  │  QueryRouter │  │  QueryRewriter  │  │   RagService     │     │
│  │  structured  │  │  expand / hyde  │  │   route → search │     │
│  │  output 分类 │  │  step_back /    │  │   → generate     │     │
│  │              │  │  decompose      │  │                  │     │
│  └──────────────┘  └─────────────────┘  └──────┬───────────┘     │
│                                                  │                 │
│                    ┌─────────────────────────────▼──────────────┐ │
│                    │           VectorStore.get_retriever()       │ │
│                    │                                              │ │
│                    │  ┌──────────────────┐  ┌──────────────┐    │ │
│                    │  │ PGVector (cosine)│  │  BM25 (内存)  │    │ │
│                    │  │ distance→sim转换 │  │  关键词检索   │    │ │
│                    │  └────────┬─────────┘  └──────┬───────┘    │ │
│                    │           └──────┬────────────┘            │ │
│                    │                  ▼                          │ │
│                    │     EnsembleRetriever (RRF 融合)            │ │
│                    │                  │                          │ │
│                    │          TopKRetriever (截断)               │ │
│                    │                  │                          │ │
│                    │     CrossEncoder Reranker (精排)            │ │
│                    │                  │                          │ │
│                    │  ┌───────────────▼────────────────────┐    │ │
│                    │  │  MultiGranularityRetriever (可选)  │    │ │
│                    │  │  子chunk检索 → parent_id → 父chunk │    │ │
│                    │  └────────────────────────────────────┘    │ │
│                    └─────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
```
