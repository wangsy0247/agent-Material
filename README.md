# Agent Material — 材料科学领域 LLM Agent

基于 FastAPI + LangGraph + PGVector 的材料科学智能体系统，集 RAG 文献检索、MCP 工具调用（Materials Project 数据库）、多轮记忆与流式对话于一体，前端为无构建依赖的原生 JS 单页应用。

## 功能特性

- **Agent 对话**：LangGraph 驱动的 ReAct Agent，支持工具调用、HITL（人工审批）中断与恢复
- **流式协议**：SSE 六通道事件流（token / status / tool_call / tool_result / hitl / done），带心跳、取消事件与序号校验
- **断流续传**：服务端 stream buffer 缓存运行事件，浏览器刷新/断网后自动续流，不丢输出
- **RAG 知识库**：PDF 文献解析（MinerU）、父块优先多粒度切分、PGVector（PostgreSQL + pgvector）父子双 collection、向量 + BM25 多路混合检索（weighted RRF 融合）、DashScope 全局 rerank、文档版本化索引（staging→active 幂等激活）、ragas 离线评估
- **MCP 工具**：内置 Materials Project 材料数据库查询/下载服务（`mp_material_server/`）
- **记忆系统**：会话记忆 + 长期记忆 + 上下文压缩，会话可删除、首问自动生成标题
- **认证鉴权**：JWT 注册登录，接口级归属校验
- **可观测性**：LangSmith / Langfuse / OpenTelemetry + Prometheus 指标（可选启用）

## 架构

```
frontend/            原生 JS 前端（聊天页 index.html + RAG 管理页 rag.html）
backend/
  app/gateway/       FastAPI 网关：路由、SSE 协议、stream buffer、鉴权
  packages/
    agent_material/  核心包：agents(LangGraph)、rag_engine、memory、models、observability
mp_material_server/  Materials Project MCP 服务（stdio）
config/              YAML 配置（agent / rag / memory / 向量库等）
evaluation_data/     ragas 评估数据集与评分
scripts/             启动与运维脚本
```

## 快速开始

环境要求：Python ≥ 3.10（开发环境为 Anaconda `langchain` 环境）。

```bash
# 一键初始化：安装依赖、从 .env.example 创建 .env、生成 AUTH_SECRET_KEY、
# 校验 config/ 下 8 个 YAML 配置、创建运行时目录
make setup

# 编辑 .env 填入 DASHSCOPE_API_KEY 等密钥后启动（默认热重载）
make run
```

也可以手动执行等效步骤：

```bash
pip install -e backend/packages/agent_material
pip install -e .
cp .env.example .env   # 填入 DASHSCOPE_API_KEY 等
bash scripts/start_server.sh        # 开发模式（热重载）
bash scripts/start_server.sh prod   # 生产模式
```

启动后访问：

- 聊天页：`http://localhost:8000/`
- RAG 管理页：`http://localhost:8000/rag`
- API 文档：`http://localhost:8000/docs`

> WSL/Anaconda 注意：若 psycopg 相关库报 libffi 冲突，先执行
> `export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7`（启动脚本已内置）。

也可使用 `docker-compose.yaml` 一键起服务（含 PostgreSQL checkpointer）。

## 测试

```bash
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
python -m pytest backend/tests
```

## 数据目录说明

以下目录为运行时生成或本地数据，已在 `.gitignore` 中排除：

- `data/`：知识库 PDF 文献（自行放入，经 RAG 管理页索引）
- `datas/`：本地文档镜像与清洗测试数据
- `docs/`：本地设计与排障笔记（个人文档，不入库）
- `chroma_db/`：历史残留的 ChromaDB 目录（当前实现已迁移至 PGVector，此目录不再使用）
- `postgresql/`：PGVector（PostgreSQL + pgvector）Docker 数据目录
- `logs/`、`workspace/`、`downloads/`：日志、工作区、MCP 下载产物

## RAG 索引与检索说明

- **PostgreSQL 控制面**：任务队列（`FOR UPDATE SKIP LOCKED` 原子领取 + 租约/心跳）、文件状态、文档版本状态、知识库 revision 统一存于 PostgreSQL，与 PGVector 数据同库，消除跨库一致性窗口（SQLite 旧路径保留为 `pg_control_plane: false` 回滚）。
- **版本化索引**：每个文档按内容 hash 生成 `document_id`/`version_id`，新版本先写 staging（`is_active=false`），校验通过后单事务激活（旧版本退役 + FileTracker 更新 + revision 递增同事务）；重复入库幂等，Worker 崩溃后租约过期自动恢复。
- **在线检索**：单次检索（sources 与生成上下文同源冻结）、多查询跨查询 weighted RRF 融合、原始 query 全局 rerank（每请求一次）、原位有序父块回溯、请求级 HyDE（无共享状态）、BM25 按 revision 检测原子刷新 + RRF 前 active 二次校验。
- **Feature flags**：`config/vector_store_config.yaml` 的 `feature_flags` 段可逐项关闭新能力回滚到旧路径（pg_control_plane / versioned_ingest / parent_first_split / active_version_filter / atomic_bm25_refresh / cross_query_rrf / global_rerank / request_scoped_hyde / single_retrieval_sse）。
- **存量数据**：启用 `active_version_filter` 前，对已有索引执行一次 `python scripts/backfill_active_metadata.py` 补 `is_active` 标记（幂等）；从 SQLite 控制面迁移执行 `python scripts/migrate_control_plane_to_pg.py`（支持 `--dry-run`）。
