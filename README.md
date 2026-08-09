# Agent Material — 材料科学领域 LLM Agent

基于 FastAPI + LangGraph + ChromaDB 的材料科学智能体系统，集 RAG 文献检索、MCP 工具调用（Materials Project 数据库）、多轮记忆与流式对话于一体，前端为无构建依赖的原生 JS 单页应用。

## 功能特性

- **Agent 对话**：LangGraph 驱动的 ReAct Agent，支持工具调用、HITL（人工审批）中断与恢复
- **流式协议**：SSE 六通道事件流（token / status / tool_call / tool_result / hitl / done），带心跳、取消事件与序号校验
- **断流续传**：服务端 stream buffer 缓存运行事件，浏览器刷新/断网后自动续流，不丢输出
- **RAG 知识库**：PDF 文献解析（MinerU）、多粒度切分、ChromaDB 向量检索、DashScope rerank、ragas 离线评估
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
docs/                设计与排障文档
```

## 快速开始

环境要求：Python ≥ 3.10（开发环境为 Anaconda `langchain` 环境）。

```bash
# 1. 安装依赖（核心包以 editable 方式安装）
pip install -e backend/packages/agent_material
pip install -e .

# 2. 配置环境变量
cp .env.example .env   # 填入 DASHSCOPE_API_KEY 等

# 3. 启动服务（脚本会自动加载 .env）
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
- `chroma_db/`：向量库（索引后自动重建）
- `logs/`、`workspace/`、`downloads/`：日志、工作区、MCP 下载产物

## 文档

`docs/` 下有各子系统的设计与排障记录：SSE 流式协议、MCP 并发调用修复、记忆压缩、可观测性分析、升级路线图（`docs/upgrade-roadmap.md`）等。
