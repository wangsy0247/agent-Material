# 材料科学 LLM Agent — 面试导向升级计划表

## 项目背景

当前项目是一个基于 FastAPI + LangChain + ChromaDB 的材料科学领域 LLM Agent。架构为单体应用，依赖 SQLite（会话）、ChromaDB（向量存储）、外部 API（大模型调用）。目标是通过 7 大技术点升级为企业级生产架构，以应对 2025-2026 年 AI 应用开发岗位面试要求。

---

## 总体策略

> **目标**：7 周内将项目从"原型级"提升到"面试杀手级"，每个技术点都对应明确的面试考点和话术。

| 阶段 | 周期 | 主题 | 面试回报 |
|------|------|------|---------|
| Phase 1 | Week 1 | MCP 协议 + LangGraph 重构 | 🔥🔥🔥🔥🔥 |
| Phase 2 | Week 2 | 认证授权 + 多租户隔离 | 🔥🔥🔥🔥🔥 |
| Phase 3 | Week 3 | Docker-Compose + Redis 缓存 | 🔥🔥🔥🔥 |
| Phase 4 | Week 4 | Milvus 向量数据库升级 | 🔥🔥🔥🔥 |
| Phase 5 | Week 5 | RAGAS 评估 + 护栏技术 | 🔥🔥🔥🔥 |
| Phase 6 | Week 6 | Celery 异步队列 | 🔥🔥🔥 |
| Phase 7 | Week 7 | vLLM + LoRA 微调（加分项） | 🔥🔥🔥 |

---

## Phase 1：MCP 协议 + LangGraph 重构（Week 1）

> **为什么放第一**：2026 年面试最高频加分项，MCP 被称为"AI 时代的 USB-C"，不会 MCP = 技术前沿敏感度不足。

### 1.1 MCP 协议集成（P0）

| 任务 | 具体内容 | 产出文件 |
|------|---------|---------|
| MCP Server 基类 | 实现 `stdio` / `sse` 两种传输模式 | `mcp_servers/server.py` |
| MCP Client | 连接外部 MCP Server | `mcp_servers/client.py` |
| 现有工具 MCP 化 | 将 `rag_summary`、`mp_download_tool` 等封装为 MCP Tool | `mcp/tools/rag_server.py`、`mcp/tools/mp_mcp_server.py` |
| LangChain 桥接 | `MCPToolAdapter`：MCP Tool ↔ LangChain `BaseTool` 双向转换 | `mcp/adapters/langchain_adapter.py` |
| 配置驱动 | `config/mcp_config.yaml` 管理所有 MCP Server | `config/mcp_config.yaml` |
| 接入第三方 Server | 接入文件系统 MCP Server (`@modelcontextprotocol/server-filesystem`) | 验证可用性 |

**面试话术**：
> "我把现有工具改造成了 MCP Server，实现了标准化接口。MCP 和 Function Calling 最大的区别是：MCP 是协议层（跨模型、跨语言），Function Calling 是模型能力。比如我现在把 RAG 工具暴露为 MCP Server，既可以被 LangChain Agent 调用，也可以被 Claude Desktop 或其他 MCP Client 调用，完全解耦。"

### 1.2 LangGraph 状态机重构（P0）

| 任务 | 具体内容 | 产出文件 |
|------|---------|---------|
| StateGraph 定义 | 定义 `AgentState`（messages、memory、tool_calls、iteration_count） | `agent/graph/agent_graph.py` |
| 节点实现 | `agent_node`（推理）、`tool_node`（工具执行）、`memory_node`（记忆注入） | 同上 |
| 条件边 | `should_continue`：判断是继续推理还是返回结果 | 同上 |
| HITL 支持 | `human_approval` 节点：敏感操作前挂起等待人工确认 | 同上 |
| 循环限制 | `iteration_count > 10` 时强制退出，防止死循环 | 同上 |
| 替换 create_agent | `ReactAgent` 改为编译 LangGraph，支持 stream / invoke 双模式 | 改造 `agent/React_agent.py` |

**面试话术**：
> "我不用 LangChain 的 `create_agent` 黑盒，而是用 LangGraph 显式构建了状态机。每个节点（推理、工具执行、记忆注入、人工审批）都是独立的，通过条件边连接。这样做的好处是：可以精确控制执行流程、支持断点调试、实现 Human-in-the-Loop，而且图结构可视化后非常清晰。"

### 面试考点

| 考点 | 核心问题 |
|------|---------|
| MCP 的核心价值 | 标准化工具调用接口，解耦 LLM 与工具实现，实现工具生态的可插拔 |
| MCP vs Function Calling | MCP 是协议层（跨模型、跨语言），Function Calling 是模型能力（OpenAI/Claude 专有格式），MCP 可桥接不同模型的 function calling |
| stdio vs SSE 传输 | stdio（子进程通信，适合本地工具）、SSE（HTTP 流式，适合远程服务）、WebSocket（双向，适合复杂交互） |
| MCP 的安全模型 | 工具执行沙箱化、权限白名单、用户确认机制（human-in-the-loop）、敏感操作二次确认 |
| LangGraph 的优势 | 显式状态机、条件边路由、持久化状态、支持 HITL 和断点调试 |
| LangGraph vs create_agent | create_agent 是黑盒封装，LangGraph 是白盒编排，后者更适合复杂流程控制 |

---

## Phase 2：认证授权 + 多租户隔离（Week 2）

> **为什么放第二**：生产系统基础能力，没有认证 = 项目不可上线，面试官会直接质疑工程素养。

### 2.1 JWT 认证体系（P0）

| 任务 | 具体内容 | 产出文件 |
|------|---------|---------|
| JWT 工具 | `create_access_token` / `verify_token` / `get_current_user` | `api/auth/jwt_utils.py` |
| 登录接口 | `POST /api/v1/auth/login` 返回 access_token | `api/routes/auth.py` |
| 用户模型 | `User` 表/模型：id, username, hashed_password, role | `api/schemas.py` |
| 密码安全 | bcrypt 哈希，不存明文 | 同上 |
| 路由保护 | 所有业务路由加 `Depends(get_current_user)` | 改造各路由文件 |
| Token 刷新 | `POST /api/v1/auth/refresh` | `api/routes/auth.py` |

### 2.2 多租户隔离（P0）

| 任务 | 具体内容 | 产出文件 |
|------|---------|---------|
| user_id 注入 | `get_current_user` 返回 user 对象，注入到所有服务 | `api/dependencies.py` |
| 会话隔离 | `SessionStore` 按 `user_id` 过滤，用户只能看到自己的会话 | 改造 `memory/session_store.py` |
| 长期记忆隔离 | `LongTermMemory` 的 `USER.md` 按用户分文件 | 改造 `memory/long_term.py` |
| 向量隔离 | Milvus/Chroma 的 collection 或 metadata 中加 `user_id` 过滤 | 改造 `rag/vector_store.py` |
| 角色权限 | `role: admin / user`，admin 可查看所有会话 | `api/auth/permissions.py` |

**面试话术**：
> "我的系统支持多租户隔离。每个用户的会话、长期记忆、向量检索都是隔离的。实现上，JWT Token 解析出 user_id，传递到记忆层和向量层，数据库查询都带 `WHERE user_id = ?` 条件。这不仅仅是安全性问题，也是数据隐私合规的基础。"

### 面试考点

| 考点 | 核心问题 |
|------|---------|
| JWT 结构 | Header.Payload.Signature，分别包含什么，如何防止篡改 |
| JWT vs Session | 无状态 vs 有状态、扩展性、Token 刷新策略、注销处理（黑名单） |
| 密码存储 | 为什么不能用 MD5/SHA1、bcrypt/scrypt/Argon2 的选择、盐值的作用 |
| 多租户隔离方案 | 数据库级隔离（独立库/Schema）vs 行级隔离（user_id 字段）的权衡 |
| RBAC 权限模型 | Role-Based Access Control 的设计，权限粒度控制 |

---

## Phase 3：Docker-Compose + Redis 缓存（Week 3）

> **为什么放第三**：Redis 缓存是企业控制 LLM 调用成本的核心手段，面试必问"你的系统怎么做成本控制"。

### 3.1 Docker 容器化

| 任务 | 具体内容 | 产出文件 |
|------|---------|---------|
| Dockerfile | 多阶段构建，Python 3.11 slim 基础镜像 | `Dockerfile` |
| Docker Compose | `api`、`redis`、`postgres/milvus` 服务编排 | `docker-compose.yml` |
| 环境变量 | `.env` 管理所有配置，不同环境加载不同文件 | `.env.example` |
| 健康检查 | `HEALTHCHECK` 指令 | `Dockerfile` |

### 3.2 Redis 语义缓存（P1）

| 任务 | 具体内容 | 产出文件 |
|------|---------|---------|
| 语义缓存类 | `SemanticCache`：Embedding 余弦相似度判断命中 | `cache/semantic_cache.py` |
| Key 设计 | `cache:emb:{md5}` + `cache:ans:{md5}` | 同上 |
| TTL 策略 | RAG 问答 1h、MP 属性 24h、通用对话 10min | 同上 |
| RAG 接入 | `rag_service.py` 查询前检查缓存 | 改造 `rag/rag_service.py` |
| Agent 接入 | 常见材料问答直接命中缓存 | 改造 `agent/tools/agent_tools.py` |

### 3.3 Redis 分布式会话（P1）

| 任务 | 具体内容 | 产出文件 |
|------|---------|---------|
| Redis Session Store | `session:{id}:messages`（List）、`session:{id}:meta`（Hash） | `memory/redis_session_store.py` |
| 配置切换 | 通过 `config.yaml` 切换 SQLite / Redis | 改造 `memory/manager.py` |
| TTL 过期 | 24h 自动清理过期会话 | 同上 |

### 3.4 API 速率限制（P1）

| 任务 | 具体内容 | 产出文件 |
|------|---------|---------|
| 滑动窗口限流 | Redis Sorted Set 实现 | `api/middleware/rate_limit.py` |
| 分级限流 | 普通用户 10/min，VIP 100/min | 同上 |
| 依赖注入 | `@router.post(..., dependencies=[Depends(rate_limit_check)])` | 改造路由文件 |

**面试话术**：
> "我用 Redis 做了三层防护：语义缓存（命中相似查询，降低 60%+ LLM 调用成本）、分布式会话（多 Worker 共享状态，替代 SQLite 单文件锁）、速率限制（滑动窗口防止 API 费用爆掉）。缓存设计考虑了穿透（布隆过滤器）、击穿（互斥锁）、雪崩（随机 TTL）三种场景。"

### 面试考点

| 考点 | 核心问题 |
|------|---------|
| 缓存穿透/击穿/雪崩 | 布隆过滤器防穿透、互斥锁防击穿、随机 TTL 防雪崩 |
| Redis 持久化 | RDB 快照 vs AOF 日志的权衡，混合持久化配置 |
| 分布式锁 | Redlock 算法原理、SET NX EX 原子操作、锁续约机制 |
| 一致性哈希 | Cluster 模式下 Key 分布、Slot 迁移、MOVED/ASK 重定向 |
| 语义缓存设计 | Embedding 相似度阈值选择、缓存命中率优化、Embedding 模型版本管理 |
| 多 Worker 会话共享 | 为什么 SQLite WAL 不够（单文件锁）、Redis List 的并发安全 |

---

## Phase 4：Milvus 向量数据库升级（Week 4）

> **为什么放第四**：Chroma → Milvus 是工程深度的体现，能讲清 HNSW vs IVFFlat = 向量检索基础扎实。

### 4.1 Milvus 部署与接入

| 任务 | 具体内容 | 产出文件 |
|------|---------|---------|
| Docker Compose | `etcd` + `minio` + `milvus-standalone` | `docker-compose.yml` |
| 向量存储抽象 | `VectorStoreBase` ABC → `ChromaVectorStore` / `MilvusVectorStore` | `db/vector_stores/base.py`、`db/vector_stores/milvus_store.py` |
| Schema 设计 | `id`, `chunk_text`, `embedding`(1024d), `doc_source`, `doc_type`, `user_id`, `md5_hash`, `created_at` | 同上 |
| 索引选择 | HNSW（数据量大）或 IVF_FLAT（默认） | 配置驱动 |

### 4.2 混合检索迁移

| 任务 | 具体内容 | 产出文件 |
|------|---------|---------|
| 向量检索 | `collection.search()` cosine 相似度 | 同上 |
| BM25 保留 | Python 层 `rank-bm25` 做关键词检索 | 复用现有 |
| RRF 融合 | 复用现有融合逻辑 | 复用现有 |
| CrossEncoder 重排序 | 复用现有重排序器 | 复用现有 |
| 标量过滤 | `expr="user_id == 'xxx' AND doc_source LIKE '%.pdf'"` | 同上 |

### 4.3 数据迁移

| 任务 | 具体内容 | 产出文件 |
|------|---------|---------|
| 迁移脚本 | ChromaDB 导出 → Milvus 批量导入（batch=100） | `scripts/migrate_chroma_to_milvus.py` |
| MD5 去重 | 迁移时保持去重逻辑 | 同上 |

**面试话术**：
> "我从 Chroma 迁移到了 Milvus，因为 Chroma 是单机嵌入式数据库，不适合生产环境。Milvus 支持十亿级向量、GPU 加速、分布式查询。Schema 设计上，我除了向量字段，还加了 `user_id`（多租户隔离）和 `md5_hash`（去重）标量字段，利用 Milvus 的标量过滤做 pre-filtering，查询性能比 post-filter 提升很多。索引方面，小数据量用 IVF_FLAT，大数据量切换到 HNSW。"

### 方案对比

| 维度 | Milvus（主推） | PostgreSQL + pgvector（备选） |
|------|---------------|------------------------------|
| 向量规模 | 十亿级 | 百万级 |
| 检索速度 | GPU 加速、分布式查询 | 单节点，索引构建慢 |
| 混合查询 | 标量过滤 + 向量检索 | SQL JOIN + 向量检索（强大） |
| 运维复杂度 | 中（Etcd/MinIO/Pulsar） | 低（单容器） |
| 适用场景 | 海量文档、高并发检索 | 中小规模、强事务需求 |

### 面试考点

| 考点 | 核心问题 |
|------|---------|
| HNSW vs IVFFlat | HNSW 图索引（m/ef_construction/ef_search）vs IVFFlat 倒排（nlist/nprobe）构建代价、查询速度、召回率对比 |
| Milvus 架构 | standalone vs cluster、write node / query node / data node 分离、Etcd 元数据、MinIO 对象存储 |
| 标量过滤前置 vs 后置 | Milvus 的 pre-filter（先过滤再检索）与 post-filter 的性能差异、数据分布影响 |
| 向量维度与量化 | PQ（乘积量化）、SQ（标量量化）如何降低内存占用、对精度的影响 |
| pgvector 的局限性 | 没有 pre-filtering、高维向量召回率下降、索引构建阻塞写入 |
| RRF 融合公式 | `score = Σ 1/(k + rank)`，不同检索器的权重分配策略 |
| Embedding 版本管理 | 模型升级后向量空间变化、重新嵌入策略、蓝绿切换方案 |

---

## Phase 5：RAGAS 评估 + 护栏技术（Week 5）

> **为什么放第五**：RAGAS 是生成质量评估的行业标准，护栏是生产安全必备，两者都是"你有没有生产意识"的试金石。

### 5.1 RAGAS 评估体系（P1）

| 任务 | 具体内容 | 产出文件 |
|------|---------|---------|
| 测试集构建 | 50-100 条用例，覆盖事实问答、比较分析、多文档综合、无答案边界 | `eval/datasets/rag_testset.yaml` |
| RAGAS 指标 | `Context Precision`、`Context Recall`、`Faithfulness`、`Answer Relevancy` | `eval/ragas_evaluator.py` |
| 评判模型 | 支持本地 vLLM 或 API（DeepSeek/Claude）作为 Judge | 同上 |
| Agent 评估 | `Tool Selection Accuracy`、`Task Completion Rate` | `eval/agent_evaluator.py` |
| 评估流水线 | 加载测试集 → 运行 → 计算指标 → 生成 HTML 报告 | `eval/run_evaluation.py` |
| CI/CD 集成 | GitHub Actions 自动运行，指标下降阻断合并 | `.github/workflows/eval.yml` |

### 5.2 护栏技术（Guardrails）（P2）

| 任务 | 具体内容 | 产出文件 |
|------|---------|---------|
| 输入护栏 | 敏感词过滤、Prompt 注入检测（正则 + 轻量模型） | `guardrails/input_guard.py` |
| 输出护栏 | 事实一致性校验（与检索上下文对比）、敏感信息脱敏 | `guardrails/output_guard.py` |
| 主题限制 | 限制 Agent 只能回答材料科学相关问题 | 同上 |
| 失败处理 | 触发护栏时返回固定话术，不暴露内部错误 | 同上 |

**面试话术**：
> "我的 RAG 系统不是只有检索评估，还有生成质量评估。用 RAGAS 的 Faithfulness 指标判断答案是否忠于检索到的文档，Answer Relevancy 判断答案与问题的相关度。同时我加了护栏系统：输入层检测 Prompt 注入和敏感词，输出层校验事实一致性，防止模型幻觉导致错误信息流出。这在企业场景里是合规要求。"

### 面试考点

| 考点 | 核心问题 |
|------|---------|
| RAGAS 的局限 | 依赖 LLM 评判（judge）的偏差、对中文支持、计算成本高、无法评估实时性 |
| Faithfulness 的原理 | 答案 claim 抽取 → 上下文支撑判断 → 统计比例，与事实准确性的区别 |
| LLM-as-a-Judge 的 bias | 位置偏见（先出现的答案给高分）、长度偏见（长的给高分）、自我偏好（同模型给高分）、缓解策略 |
| 评估指标选择 | 什么时候看重 Context Recall（知识库问答）vs Answer Relevancy（开放域对话） |
| 自动化评估 vs 人工评估 | 自动评估的快速反馈 vs 人工评估的准确性，如何建立 Human-in-the-loop 流程 |
| A/B 测试的统计基础 | 样本量计算（power analysis）、显著性水平 α=0.05、效应量 Cohen's d |
| RAG 的 Ground Truth 构建 | 如何构建高质量测试集（人工标注 vs LLM 生成 vs 用户反馈），数据泄漏防范 |

---

## Phase 6：Celery 异步任务队列（Week 6）

> **为什么放第六**：异步队列是工程能力的重要体现，但面试优先级略低于前面的"看得见"的技术点。

### 6.1 Celery 基础设施

| 任务 | 具体内容 | 产出文件 |
|------|---------|---------|
| Celery App | Redis 作为 broker 和 backend | `tasks/celery_app.py` |
| Worker 服务 | Docker Compose 添加 `worker` 服务 | `docker-compose.yml` |
| Flower 监控 | 可选添加 Flower 面板 | `docker-compose.yml` |

### 6.2 异步任务定义

| 任务 | 具体内容 | 产出文件 |
|------|---------|---------|
| 文档入库任务 | `load_document_task`：PDF 解析 → 切分 → Embedding → Milvus 入库 | `tasks/document_tasks.py` |
| 批量下载任务 | `batch_download_task`：MP 材料批量异步下载 | `tasks/mp_tasks.py` |
| 评估任务 | `run_rag_evaluation_task`：异步执行 RAGAS | `tasks/eval_tasks.py` |
| 进度推送 | Redis Pub/Sub 或 WebSocket 推送前端 | 同上 |

### 6.3 API 改造

| 任务 | 具体内容 | 产出文件 |
|------|---------|---------|
| 异步提交接口 | `POST /rag/load-document` 返回 `task_id` | 改造 `api/routes/rag.py` |
| 进度查询接口 | `GET /rag/task-status/{task_id}` | 同上 |
| 定时任务 | Celery Beat：每日清理过期会话、每周全量评估 | 同上 |

**面试话术**：
> "PDF 解析和向量入库是 CPU 密集型操作，不能阻塞 API 主线程。我用 Celery 将这些操作异步化，前端提交任务后拿到 task_id，通过轮询或 WebSocket 获取进度。任务支持幂等性（MD5 去重）、优先级分级、失败重试（指数退避）。Worker 和 API 服务独立部署，可以按需扩缩容。"

### 面试考点

| 考点 | 核心问题 |
|------|---------|
| Celery 的架构 | Producer → Broker(Redis/RabbitMQ) → Worker → Backend(Redis)，消息路由、序列化、ACK 机制 |
| ACK 与可见性超时 | Worker 处理完才 ACK vs 预取即 ACK，任务超时导致的重复执行问题、幂等性设计 |
| 任务重试策略 | max_retries、countdown（固定间隔）、exponential backoff（指数退避）、死信队列 |
| 优先级队列 | Redis 的 list 不支持优先级，需用 RabbitMQ 或 Redis 的 sorted set 模拟（`priority` 参数） |
| 定时任务设计 | Celery Beat（单点调度，需高可用方案如多 Beat + 锁）vs APScheduler vs Linux crontab |
| 任务结果存储 | Redis 结果过期策略、结果过大时的处理（存文件系统/对象存储，Redis 存引用） |
| 并发控制 | Worker 的 `prefetch_multiplier`、并发模型（prefork/gevent/eventlet）、CPU 密集 vs IO 密集任务的 Worker 分离 |
| 分布式锁 | Celery 任务幂等性如何通过 Redis SETNX 实现、防止同一文档重复解析 |
| 与 FastAPI 的集成 | 后台任务（BackgroundTasks）vs Celery（独立进程），什么场景用哪个 |

---

## Phase 7：vLLM + LoRA 微调（Week 7）

> **为什么是加分项**：私有化部署和微调是高级岗位的区分度，但基础岗位不一定问到。如果时间紧，可以只了解原理，不全面实现。

### 7.1 vLLM 推理服务（P2）

| 任务 | 具体内容 | 产出文件 |
|------|---------|---------|
| vLLM 部署 | Docker Compose 添加 `vllm` 服务，OpenAI-compatible API | `docker-compose.yml` |
| 模型工厂改造 | `ChatModelFactory` 支持 vLLM 后端（`base_url="http://vllm:8000/v1"`） | 改造 `model/model_factory.py` |
| Embedding 本地化 | `BAAI/bge-large-zh-v1.5` 本地加载，替代 DashScope | 改造 `model/model_factory.py` |

### 7.2 LoRA 微调（P2，可选）

| 任务 | 具体内容 | 产出文件 |
|------|---------|---------|
| 数据准备 | 材料科学论文摘要 → 问答对（ShareGPT 格式） | `training/data/` |
| LoRA 配置 | `r=64, alpha=128, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]` | `training/lora_finetune.py` |
| 训练 | `peft` + `transformers` + `trl`，bf16、deepspeed ZeRO-2 | 同上 |
| vLLM + LoRA | 动态加载 LoRA 适配器，`model="material-lora"` 切换 | 配置 |

**面试话术**（即使没完全实现，也要能讲）：
> "我了解 vLLM 的核心优化：PagedAttention 把 KV Cache 分页管理（类比操作系统虚拟内存），避免内存碎片；Continuous Batching 让新请求随时加入当前 batch，最大化 GPU 利用率。LoRA 微调方面，我只训练低秩矩阵（参数量减少 99%+），在材料科学语料上做了领域适配。推理时 vLLM 支持多 LoRA 适配器动态切换，不需要为每个适配器加载完整模型。"

### 面试考点

| 考点 | 核心问题 |
|------|---------|
| vLLM 的核心优化 | PagedAttention（KV Cache 分页管理）、Continuous Batching（动态批处理）、Tensor Parallelism（张量并行） |
| PagedAttention 原理 | 类比操作系统虚拟内存，将 KV Cache 划分为固定大小的 block，避免内存碎片和重复分配 |
| Continuous Batching | 与 Static Batching 对比，新请求随时加入当前 batch，GPU 利用率最大化，推理延迟变化 |
| LoRA vs Full Fine-tuning | 只训练低秩矩阵（A∈R^{r×d}, B∈R^{d×r}），参数量减少 99%+，显存占用降低，推理时合并权重或动态加载 |
| LoRA 超参选择 | r（秩，通常 8-64）、alpha（缩放系数，通常 2r）、target_modules（哪些层加 LoRA）、dropout |
| QLoRA 量化 | 4-bit NormalFloat + Double Quantization + Paged Optimizer，单卡微调 65B 模型的原理 |
| vLLM 的调度策略 | FCFS（先来先服务）+ preemptive scheduling（抢占低优先级请求），TTFT（首 token 延迟）与 TPOT（每 token 延迟）的权衡 |
| 推理服务高可用 | 多实例负载均衡、健康检查、自动扩缩容（K8s HPA）、请求队列（vLLM 自带 vs 外部队列） |
| 模型量化对 LoRA 的影响 | GPTQ/AWQ 量化后的模型加载 LoRA 适配器的兼容性、精度损失评估 |

---

## 综合产出物清单

| 文件/目录 | 所属阶段 | 面试价值 |
|-----------|---------|---------|
| `mcp_servers/` | Phase 1 | 🔥🔥🔥🔥🔥 2026 最热技术 |
| `agent/graph/`（LangGraph） | Phase 1 | 🔥🔥🔥🔥🔥 架构深度 |
| `api/auth/`（JWT） | Phase 2 | 🔥🔥🔥🔥🔥 工程基础 |
| `docker-compose.yml` | Phase 3 | 🔥🔥🔥🔥 部署能力 |
| `cache/semantic_cache.py` | Phase 3 | 🔥🔥🔥🔥 成本控制 |
| `memory/redis_session_store.py` | Phase 3 | 🔥🔥🔥🔥 分布式意识 |
| `db/vector_stores/milvus_store.py` | Phase 4 | 🔥🔥🔥🔥 生产级数据库 |
| `eval/` | Phase 5 | 🔥🔥🔥🔥 质量意识 |
| `guardrails/` | Phase 5 | 🔥🔥🔥 安全意识 |
| `tasks/` | Phase 6 | 🔥🔥🔥 工程能力 |
| `training/` | Phase 7 | 🔥🔥🔥 加分项 |

---

## 风险与简化策略

| 风险 | 简化方案 |
|------|---------|
| Milvus 资源占用太高 | 先用 PostgreSQL + pgvector 替代，面试时能讲清 Milvus 架构即可 |
| 没有 GPU 跑 vLLM | 用 Ollama / llama.cpp 跑小模型（Qwen2.5-7B），或只了解原理 |
| MCP 生态不成熟 | 先接入官方示例 Server（filesystem），证明理解协议即可 |
| 时间不够 | **优先保证 Phase 1+2+3**，这三项覆盖 80% 面试考点 |

---

## 最终面试话术框架

当被问到"你的项目有什么亮点"时，用这段结构回答：

> "我的项目是一个面向材料科学的 LLM Agent，核心亮点有五个：
> 
> 1. **检索质量**：四阶段混合检索（向量 + BM25 + RRF + CrossEncoder 重排序），用 RAGAS 量化评估 Faithfulness 和 Answer Relevancy；
> 2. **记忆系统**：三层记忆架构（短期 Token 预算 + 中期 SQLite/Redis + 长期文件+向量），自动触发 LLM 上下文压缩；
> 3. **协议标准化**：工具层接入了 MCP 协议，实现跨模型、跨语言的标准化工具调用；
> 4. **工程化**：Docker-Compose 全栈编排、Redis 语义缓存（降低 LLM 调用成本）、JWT 多租户隔离；
> 5. **生产意识**：加了护栏系统防止 Prompt 注入和幻觉输出，Celery 异步化耗时操作，API 分级速率限制。
> 
> 整个系统从原型级升级到了可上线的生产架构。"
