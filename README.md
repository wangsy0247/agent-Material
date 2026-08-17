# Agent Material — 材料科学 Agentic RAG

Agent Material 是一个面向材料科学文献问答与工具调用的开源 Agent 项目。
系统由 FastAPI、LangGraph、PostgreSQL/PGVector、BM25 和原生 JavaScript
前端组成，覆盖 PDF 离线索引、混合检索、全局重排、流式生成、MCP 工具
调用和多轮记忆。

> 项目状态：`v0.1.0-alpha`，面向学习、研究与作品演示。目前按可信的
> 单租户、本地单实例部署设计，不应直接作为公网生产 SaaS 使用。

## 核心能力

- LangGraph ReAct Agent：工具调用、HITL 人工审批、中断与恢复。
- SSE 流式协议：token、状态、工具事件、终态事件、心跳和断流续传。
- Agentic RAG：查询路由、查询改写、多路召回、weighted RRF、全局 rerank。
- 版本化离线索引：PostgreSQL 任务队列、租约、fencing、staging/active 原子切换。
- 多粒度索引：父子文档切分，PGVector 父块/子块 collection，父块回溯。
- PDF 质量门禁：原生、扫描和混合 PDF 识别；扫描件路由到 MinerU OCR/视觉解析。
- Materials Project MCP：材料查询与受 HITL 保护的文件下载。
- 可选可观测性：LangSmith、Langfuse、OpenTelemetry 和 Prometheus。

## 系统边界

- 知识库为共享知识库，上传、删除、重建、任务重试和一致性检查仅管理员可用。
- 普通登录用户可以检索、问答和查看允许公开的来源信息。
- MCP 配置只允许管理员读取；通过 HTTP 在线修改默认关闭。
- SSE 续传缓冲区位于进程内，因此当前服务固定单 worker。
- SQLite 只用于认证和会话等单机状态；RAG 控制面与向量数据使用 PostgreSQL。
- `interview_prep/`、`evaluation_data/` 和 `data/` 均为本地私有目录，
  分别存放个人面试资料、私有评测数据和知识库文献，不随仓库发布。

更详细的边界见 [架构说明](docs/architecture.md)、[数据与隐私](docs/privacy.md)
和 [部署说明](docs/deployment.md)。

## 快速开始

要求：Python 3.10–3.12、[uv](https://docs.astral.sh/uv/) 和 Docker。

```bash
# 1. 启动本地 PostgreSQL + pgvector（仅绑定 127.0.0.1:5433）
docker compose up -d

# 2. 安装锁定依赖、创建 .env、生成随机 JWT 密钥和运行目录
make setup

# 3. 编辑 .env，至少配置 DASHSCOPE_API_KEY

# 4. 交互式创建管理员；仓库不提供默认账号或默认密码
uv run python scripts/create_admin.py your_admin --email you@example.com

# 5. 启动服务，默认绑定 127.0.0.1:8000 且不开热重载
make run
```

启动后访问：

- 聊天页：`http://127.0.0.1:8000/`
- RAG 管理页：`http://127.0.0.1:8000/rag`
- OpenAPI：`http://127.0.0.1:8000/docs`

如需开发热重载，在 `.env` 中显式设置 `AGENT_API_RELOAD=true`。WSL/Conda
出现 `psycopg` 与 `libffi` 冲突时，再参考
[部署排障](docs/deployment.md#wslconda-libffi)，不要在所有环境中全局设置
`LD_PRELOAD`。

## 配置原则

`.env.example` 只包含非敏感示例值。必须设置至少 32 字符的
`AUTH_SECRET_KEY`；`make setup` 会为本机 `.env` 自动生成随机值。真实
`.env`、API Key 和数据库密码禁止提交。

本地 Compose 的数据库账号密码仅用于 loopback 开发环境。远程部署必须覆盖
`POSTGRES_PASSWORD`、`POSTGRES_CHECKPOINTER_URL` 和 `PGVECTOR_URL`，并通过
网络策略限制数据库访问。

遥测默认关闭。启用 LangSmith、Langfuse、OpenTelemetry 或外部 PDF 解析前，
请先阅读 [数据与隐私](docs/privacy.md)。

Materials Project MCP 默认关闭，避免未配置 API Key 或可选依赖时启动外部子进程。
如需启用，先以 `--extra materials-project` 安装依赖、配置 `MP_API_KEY`，再将
`extensions_config.json` 中 `material-project.enabled` 改为 `true`。`{python}`
占位符会解析为启动当前服务的 Python 解释器，避免跨虚拟环境调用。

## RAG 离线索引流程

1. 管理员上传 PDF，网关以分块流式读取执行 50MB 限制和 `%PDF-` 签名校验。
2. 文件写入同目录临时文件，校验完成后原子替换；任务创建失败会恢复旧文件。
3. PostgreSQL 控制面创建 `ingest_file` 任务，Worker 通过
   `FOR UPDATE SKIP LOCKED` 领取并维护租约与 `lease_generation` fencing。
4. PDF 先分类为 native、scanned、mixed 或 unknown。扫描/混合/未知文档必须
   使用 MinerU OCR/视觉路径；解析失败时关闭本次版本，不污染 active 索引。
5. 文本依次经过标题感知切分、递归切分、语义合并和父子块构建。
6. Embedding 写入 PGVector staging 版本；父子数量、身份和任务租约校验通过后，
   单事务激活新版本、退役旧版本并递增知识库 revision。
7. BM25 观察 revision 变化，按“捕获 revision → 加载 → 构建 → 原子替换 → 提交”
   刷新词法索引。服务重启后会继续 pending 或租约过期任务。

直接写入 chunks 会绕过父子和版本契约，因此 `/rag/add-chunks` 固定返回 410。

## RAG 在线检索流程

1. 查询路由识别闲聊、无关问题和需要 RAG 的问题。
2. 按配置执行 step-back、expand、HyDE 或 decomposition，并限制请求级查询数量。
3. 每个查询并行执行 PGVector 向量召回与 BM25 词法召回。
4. weighted RRF 在版本身份复核后融合候选，并以原始问题执行一次全局 rerank。
5. 子块按 `(parent_id, document_id, version_id)` 回溯 active 父块，去重后按相关性
   和字符预算冻结上下文。
6. 来源列表和生成上下文来自同一份冻结快照，随后通过 SSE 输出答案与引用。

详细的不变量、降级路径和存储边界见 [架构说明](docs/architecture.md)。

## 测试与质量检查

```bash
uv sync --locked --all-packages --extra pdf --extra materials-project
uv run ruff check backend mp_material_server scripts
uv run pytest backend/tests
uv run pip-audit
```

默认使用 DashScope reranker，不安装体积较大的本地 CrossEncoder。需要离线
reranker 时显式增加 `--extra local-reranker`。

CI 在 Python 3.10、3.11、3.12 和 PostgreSQL/pgvector 环境运行测试；CodeQL
按提交、PR 和每周计划执行。当前锁定依赖环境的完整基线为 `422 passed`。

## 仓库目录

```text
frontend/                       原生 JS 聊天页与 RAG 管理页
backend/app/gateway/            FastAPI、认证、SSE、REST 路由
backend/packages/agent_material Agent、RAG、记忆、MCP 客户端、可观测性
backend/tests/                  单元与集成测试
mp_material_server/             Materials Project stdio MCP server
config/                         非敏感 YAML 默认配置
docs/                           可公开的架构、隐私与部署文档
scripts/                        初始化、管理员创建、评测和运维脚本
```

`interview_prep/`、`evaluation_data/` 和 `data/` 整目录不公开：开源仓库只提供
代码、配置模板和测试夹具，不提供个人面试内容、实际评测集/结果或知识库文献。
`datas/`、`logs/`、`workspace/`、`downloads/`、数据库和向量索引同样属于本地数据，
均已被 Git 排除。首次运行时会按需创建所需目录，使用者需自行准备有权处理的文献。

## 安全、贡献与许可证

- 安全问题请按 [SECURITY.md](SECURITY.md) 私下报告。
- 贡献流程见 [CONTRIBUTING.md](CONTRIBUTING.md) 和
  [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。
- 项目代码采用 [Apache License 2.0](LICENSE)。
- MinerU、Materials Project 和 vendored 前端依赖有独立条款，见
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 致谢

项目的分层 Agent 架构设计参考了 DeerFlow 等开源 Agent 项目的公开思路。
这不代表上游项目对本项目提供背书。若后续引入或改编上游代码，必须在对应
文件和第三方声明中保留其版权与许可证。
