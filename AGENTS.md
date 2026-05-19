# AGENTS.md — 材料科学 LLM Agent 项目

> 本文档面向 AI 编程助手。如果你正在阅读本文件，说明你对本项目一无所知——以下信息将帮助你快速理解项目结构、技术栈、构建方式和开发约定。

---

## 1. 项目概览

本项目是面向**材料科学领域**的大语言模型（LLM）Agent 系统，核心能力包括：

- **ReAct 智能体对话**：基于 LangChain 的 ReAct Agent，支持工具调用（RAG 检索、Materials Project 查询/下载）。
- **RAG 知识库**：混合检索（向量相似度 + BM25 关键词）+ CrossEncoder 重排序，覆盖材料科学 8 大领域文献。
- **三层记忆系统**：短期记忆（上下文窗口管理）、中期记忆（SQLite 会话持久化）、长期记忆（文件 + 向量检索）。
- **MCP 工具生态**：支持通过 Model Context Protocol 接入外部工具服务（stdio / SSE 双模式）。
- **SSE 流式 API**：Agent 对话和 RAG 查询均支持 Server-Sent Events 实时推送。
- **Web 前端**：原生 HTML/CSS/JS 三栏式聊天界面（左栏会话、中栏聊天、右栏参考资料）。

项目名：`agent-material`（`pyproject.toml` 中定义）。

---

## 2. 技术栈

| 层级 | 技术/库 | 说明 |
|------|---------|------|
| **Web 框架** | FastAPI + Uvicorn | REST API 服务，SSE 流式输出 |
| **LLM 框架** | LangChain + LangGraph | ReAct Agent、工具链、Prompt 模板 |
| **大模型** | ChatAnthropic (DeepSeek-V4-Flash) | 通过 Anthropic API 协议调用 DeepSeek |
| **Embedding** | DashScopeEmbeddings (text-embedding-v4) | 阿里云灵积 Embedding API |
| **向量数据库** | ChromaDB | 本地持久化向量存储 |
| **检索增强** | BM25 + CrossEncoder | `rank-bm25` + `sentence-transformers` |
| **材料计算** | pymatgen + mp-api | Materials Project 数据查询与结构分析 |
| **数据模型** | Pydantic V2 | 所有 API 入参/出参校验 |
| **配置管理** | PyYAML | 全部配置集中放在 `config/*.yaml` |
| **测试** | pytest + pytest-asyncio | 支持异步测试，conftest 中有完善的 mock 降级 |
| **部署** | Uvicorn / Gunicorn / Docker | `scripts/start_server.sh` 支持 dev/prod/docker 三种模式 |

---

## 3. 项目结构

```
├── api_server.py              # FastAPI 主入口（ lifespan、中间件、路由注册、静态文件挂载）
├── pyproject.toml             # 项目元数据、依赖、pytest 配置
├── .env                       # 环境变量（API Key、端口等），不会被提交
├── nginx.conf                 # Nginx 反向代理配置（生产环境可选）
│
├── agent/                     # Agent 核心
│   ├── React_agent.py         # ReAct Agent 封装（LangGraph StateGraph、记忆集成、流式/非流式执行）
│   ├── graph/
│   │   ├── __init__.py
│   │   └── agent_graph.py     # LangGraph StateGraph 定义（节点、条件边、HITL、防死循环）
│   └── tools/
│       ├── agent_tools.py     # 原生 LangChain Tool 定义 + MCP 工具聚合
│       ├── mcp_tool_manager.py # MCP 工具发现、转换、调用管理
│       ├── middleware.py      # Agent 节点级日志（替代原 LangChain 中间件）
│       └── mp_downloader.py   # Materials Project 下载/查询/历史记录
│
├── api/                       # FastAPI 路由层
│   ├── schemas.py             # Pydantic V2 请求/响应模型
│   ├── dependencies.py        # 依赖注入（懒加载单例：Agent、RAG、MemoryManager）
│   └── routes/
│       ├── agent.py           # /api/v1/agent/chat（SSE 流式对话）
│       ├── rag.py             # /api/v1/rag/query + /api/v1/rag/retrieve
│       ├── system.py          # /api/v1/system/health + /api/v1/system/rag/stats
│       └── memory.py          # /api/v1/memory/sessions + /api/v1/memory/recall
│
├── rag/                       # RAG 知识库
│   ├── rag_service.py         # RAG 服务封装（检索 + 生成链）
│   ├── vector_store.py        # ChromaDB 向量库 + 混合检索器 + 重排序器
│   └── text_spliter.py        # 语义文本分割器（基于 Embedding 相似度合并）
│
├── memory/                    # 三层记忆系统
│   ├── manager.py             # MemoryManager：会话生命周期 + 上下文组装
│   ├── short_term.py          # 短期记忆：Token 预算 + 智能压缩（保护头尾消息）
│   ├── session_store.py       # 中期记忆：SQLite + WAL 模式会话持久化
│   ├── long_term.py           # 长期记忆：USER.md / MEMORY.md + ChromaDB 向量记忆
│   ├── context_compressor.py  # 上下文压缩器（LLM 驱动的对话摘要）
│   ├── models.py              # 数据模型（MessageRecord、SessionInfo、MemoryEntry 等）
│   └── utils.py               # Token 估算、时间格式化等工具函数
│
├── mcp_servers/               # MCP（Model Context Protocol）基础设施
│   ├── client.py              # MCPClient：stdio / SSE 双模式连接、工具调用
│   ├── server.py              # BaseMCPServer：FastMCP 封装，支持 stdio/sse/嵌入 FastAPI
│   ├── adapters/              # LangChain 适配器
│   └── tools/
│       ├── rag_mcp_server.py  # RAG MCP Server（独立进程或远程服务）
│       └── mp_mcp_server.py   # Materials Project MCP Server
│
├── model/                     # 模型工厂
│   └── model_factory.py       # ChatModelFactory + EmbeddingModelFactory，程序启动时单例初始化
│
├── utils/                     # 通用工具
│   ├── config_handler.py      # 读取所有 YAML 配置（模块导入时即加载为全局变量）
│   ├── logger.py              # 统一日志（控制台 + 文件，按日期分文件）
│   ├── file_handler.py        # 多格式文件加载器（PDF/TXT/MD/DOCX/PPTX）
│   ├── path_tools.py          # 路径解析（兼容相对/绝对路径）
│   ├── prompt_loader.py       # Prompt 文件加载
│   └── langsmith_setup.py     # LangSmith 可观测性配置
│
├── config/                    # YAML 配置目录（全部被 config_handler.py 加载）
│   ├── agent_config.yaml      # 大模型名称、max_tokens、外部数据路径
│   ├── rag_config.yaml        # Embedding 模型名称
│   ├── chroma_config.yaml     # 向量库参数（切分策略、检索阈值、重排序模型路径等）
│   ├── memory_config.yaml     # 三层记忆系统的全部参数
│   ├── mcp_config.yaml        # MCP Server 接入配置（stdio/sse、enabled、安全策略）
│   ├── mp_config.yaml         # Materials Project 下载目录、历史文件路径
│   ├── prompts_config.yaml    # Prompt 文件路径映射
│   └── sc_config.yaml         # （预留）
│
├── prompt/                    # Prompt 文本文件
│   ├── system_prompt.txt      # Agent 系统提示（含工具说明、思考准则、输出规范）
│   └── rag_prompt.txt         # RAG 问答提示模板
│
├── frontend/                  # Web 前端（纯静态）
│   ├── index.html             # 三栏式聊天界面
│   └── static/
│       ├── css/style.css
│       └── js/app.js
│
├── data/                      # 知识库文档目录（PDF/TXT/MD/DOCX/PPTX）
│   └── ...
│
├── workspace/                 # 运行时工作区
│   ├── memory/                # 长期记忆文件（USER.md、MEMORY.md、daily/）
│   └── mp_downloads/          # Materials Project 下载的材料结构文件
│
├── logs/                      # 日志与 SQLite 数据库
│   ├── *.log                  # 按名称分文件日志
│   └── memory_sessions.db     # 会话持久化 SQLite 数据库
│
├── chroma_db/                 # ChromaDB 本地持久化数据
│
├── scripts/
│   ├── start_server.sh        # 一键启动脚本（dev/prod/docker）
│   └── wsl_network.sh         # WSL 网络配置辅助脚本
│
└── tests/                     # 测试目录
    ├── conftest.py            # pytest 共享 fixtures（含缺失依赖的 mock 降级）
    ├── test_rag_retrieval.py  # RAG 检索多阶段召回率评估（Vector/BM25/Hybrid/Rerank）
    ├── test_mp_downloader.py  # Materials Project 下载器测试
    ├── test_utils.py          # 通用工具函数测试
    ├── rag_diagnose.py        # RAG 诊断脚本
    └── rag_eval_report.json   # 评估报告输出
```

---

## 4. 构建与运行命令

### 4.1 安装依赖

项目使用 `setuptools` 构建，推荐在虚拟环境中安装：

```bash
pip install -e .
```

或手动安装核心依赖（`pyproject.toml` 中 `dependencies` 已列出全部所需包）。

### 4.2 环境变量

复制 `.env` 并填入真实值（文件已存在于项目根目录，但**不得将真实 API Key 提交到版本控制**）：

```bash
# 关键环境变量
ANTHROPIC_API_KEY=xxx           # DeepSeek 通过 Anthropic 协议调用
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
DASHSCOPE_API_KEY=xxx           # Embedding API
MP_API_KEY=xxx                  # Materials Project
LANGCHAIN_API_KEY=xxx           # LangSmith 可观测性（可选）
AGENT_API_PORT=8000             # 服务端口
AGENT_API_HOST=0.0.0.0          # 绑定地址
```

### 4.3 启动服务

**开发模式（热重载）：**

```bash
# 方式 1：脚本
bash scripts/start_server.sh dev

# 方式 2：直接 uvicorn
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

# 方式 3：Python 入口
python api_server.py
```

**生产模式：**

```bash
bash scripts/start_server.sh prod
# 内部执行：uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 4
```

**Docker 模式：**

```bash
bash scripts/start_server.sh docker
```

启动后访问：
- API 文档（Swagger）：`http://localhost:8000/docs`
- 前端界面：`http://localhost:8000/`
- ReDoc：`http://localhost:8000/redoc`

### 4.4 测试

```bash
# 运行全部测试
pytest -v

# 运行 RAG 检索评估（较耗时，会加载模型）
pytest tests/test_rag_retrieval.py -v -s

# 运行特定测试
pytest tests/test_mp_downloader.py -v
```

pytest 配置已写入 `pyproject.toml`：
- 测试目录：`tests/`
- 文件匹配：`test_*.py`
- 类匹配：`Test*`
- 函数匹配：`test_*`
- 默认参数：`-v --tb=short`
- 忽略 DeprecationWarning

---

## 5. 配置系统

项目采用 **YAML 集中配置 + 环境变量敏感信息** 的双层策略：

- **非敏感配置** → `config/*.yaml`，通过 `utils.config_handler` 在模块导入时加载为全局变量。
- **敏感配置** → `.env` 环境变量，通过 `os.environ` 读取。

**重要约定：**
- 新增配置项时，优先在对应 `config/*.yaml` 中添加，并在 `utils/config_handler.py` 中暴露读取函数。
- 不要在代码中硬编码路径或参数，应通过 `config_handler` 读取。
- `config_handler.py` 在导入时会读取所有 YAML，因此新增配置文件后必须在此注册。

---

## 6. 代码风格与开发约定

### 6.1 语言与注释

- **所有文档字符串、注释、日志输出使用中文。**
- 类型提示（Type Hints）为强制要求，使用 Python 3.10+ 语法：`str | None`、`list[Document]`。
- Pydantic 模型使用 V2 API（`model_dump()`、`model_dump_json()`、`ConfigDict` 等）。

### 6.2 懒加载模式（Lazy Loading）

项目中大量重量级对象（VectorStore、重排序模型、RAG Service、Agent 等）采用**懒加载**以避免启动时 segfault 或长时间阻塞：

```python
# 推荐模式：模块级全局变量 + 首次调用时初始化
_rag_service = None

def _get_rag_service():
    global _rag_service
    if _rag_service is None:
        from rag.rag_service import RagService
        _rag_service = RagService()
    return _rag_service
```

`api/dependencies.py` 中使用 `@lru_cache(maxsize=1)` 实现单例懒加载。

### 6.3 错误处理与日志

- 统一使用 `utils.logger.get_logger(name)` 获取 logger，不要直接使用 `print`。
- 日志格式包含文件名和行号，便于定位问题。
- 对外部依赖（Embedding API、Materials Project、MCP Server）的调用必须有 try-except 包裹，失败时降级或返回友好提示。

### 6.4 路径处理

- 所有相对路径必须通过 `utils.path_tools.get_abs_path()` 解析为绝对路径，确保兼容多种启动方式（直接 `python`、uvicorn、`pytest` 等）。

### 6.5 Agent 工具开发

- **原生工具**：使用 `@tool` 装饰器定义，放在 `agent/tools/agent_tools.py` 中。
- **MCP 工具**：在 `config/mcp_config.yaml` 中配置 Server 信息，由 `mcp_tool_manager.py` 自动发现并转换为 LangChain Tool。
- 工具描述（description）必须清晰说明入参、出参、使用场景和调用规则，这直接影响 LLM 的工具选择能力。

---

## 7. 核心架构细节

### 7.1 RAG 检索流水线（4 级检索）

1. **向量检索**：ChromaDB cosine similarity，带自定义阈值过滤（`ChromaSimilarityThresholdRetriever` 显式处理 `distance → similarity` 转换）。
2. **BM25 关键词检索**：基于内存中的文档列表构建稀疏索引。
3. **混合检索**：`EnsembleRetriever` 以 RRF 融合向量 + BM25 结果（权重可配置，默认各 0.5）。
4. **CrossEncoder 重排序**：`ContextualCompressionRetriever` + `HuggingFaceCrossEncoder` 精排，默认取 Top-3。

**注意**：重排序模型路径在 `config/chroma_config.yaml` 中配置（当前为本地路径 `/mnt/d/models/Qwen3-Reranker-0.6B`），如果模型不存在会自动跳过重排序。

### 7.2 三层记忆系统

| 层级 | 组件 | 存储介质 | 职责 |
|------|------|----------|------|
| L1 短期记忆 | `ShortTermMemory` | 内存 | 当前对话上下文窗口，Token 超限触发智能压缩 |
| L2 中期记忆 | `SessionStore` | SQLite (`logs/memory_sessions.db`) | 会话元数据 + 完整消息历史，WAL 模式 |
| L3 长期记忆 | `LongTermMemory` | 文件 (`workspace/memory/`) + ChromaDB 向量 | 用户画像、项目知识、向量记忆条目、自动提取 |

**记忆注入流程**：`MemoryManager.build_context()` 按以下顺序组装消息：
1. 系统提示（原始提示 + 长期记忆上下文：用户画像 + 项目知识 + 相关记忆召回）
2. 对话历史（短期记忆中的消息，已做压缩保护）

### 7.3 LangGraph Agent 状态机

Agent 核心已从 `langchain.agents.create_agent` 迁移为显式 `LangGraph StateGraph`：

```
entry -> memory_node -> agent_node -> should_continue
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                         │
        iteration>10               has tool_calls?           无 tool_calls
              │                         │                         │
              ▼                         ▼                         ▼
         finalize                human_approval_node          finalize
                                         │
                                  ┌──────┴──────┐
                                 批准           拒绝
                                  │             │
                                  ▼             ▼
                               tool_node  -> agent_node
                                  │
                                  └────────-> agent_node
```

**节点说明**：

| 节点 | 职责 |
|------|------|
| `memory_node` | 从 `MemoryManager` 组装完整上下文（系统提示 + 长期记忆 + 对话历史） |
| `agent_node` | 调用 `chat_model.bind_tools(tools).ainvoke()`，迭代计数 +1 |
| `human_approval_node` | 检测敏感操作（`mp_download` 等），需要时调用 `interrupt()` 挂起 |
| `tool_node` | 使用 `langgraph.prebuilt.ToolNode` 并行执行工具，内置日志监控 |
| `finalize_node` | 提取最终答案，超过 10 轮时附加截断提示 |

**关键机制**：
- **防死循环**：`iteration_count > 10` 时 `should_continue` 强制路由到 `finalize_node`
- **HITL 恢复**：`human_approval_node` 拒绝时生成拒绝 `ToolMessage`，直接回到 `agent_node` 重新推理；批准时继续到 `tool_node` 执行
- **Checkpointer**：使用 `MemorySaver` 保存 graph 状态，支持 `Command(resume=...)` 恢复中断执行
- **Thread 隔离**：每次新调用使用唯一 `thread_id`（`{session_id}_{random}`），避免 checkpointer 消息累积导致 system message 重复

### 7.4 MCP 架构

- **Client**：`mcp_servers/client.py` 封装 `stdio` 和 `sse` 两种传输模式，支持上下文管理器自动管理连接生命周期。
- **Server**：`mcp_servers/server.py` 提供 `BaseMCPServer` 基类，可快速注册 Tool/Resource/Prompt，支持独立运行（stdio/sse）或嵌入 FastAPI。
- **Tool 转换**：`mcp_tool_manager.py` 将 MCP JSON Schema 动态转为 Pydantic 模型，再包装为 LangChain `StructuredTool`。
- **调用策略**：发现时建立临时连接获取元数据；Agent 每次调用工具时新建临时连接（简单可靠，避免长连接管理复杂性）。

---

## 8. 测试策略

### 8.1 测试结构

- `tests/conftest.py` 是核心基础设施，提供以下能力：
  - 自动将项目根目录加入 `sys.path`。
  - **缺失依赖的 mock 降级**：如果环境中未安装 `langchain`、`pymatgen`、`mp_api` 等库，会自动注入 `MagicMock`，确保测试文件可以正常导入而不触发 `ImportError`。
  - 为 `langchain_core.tools.tool` 提供模拟装饰器，支持有参/无参调用。

### 8.2 主要测试文件

- `test_rag_retrieval.py`：RAG 系统的量化评估测试，包含 13 条材料科学领域测试文档和 8 条查询用例，评估指标包括 Recall@K、Precision@K、MRR、NDCG@K。支持 Embedding API 不可用时自动降级为本地 TF-IDF。
- `test_mp_downloader.py`：Materials Project 下载器的单元测试。
- `test_utils.py`：通用工具函数的单元测试。
- `rag_diagnose.py`：RAG 诊断脚本（非 pytest 标准测试，用于现场问题排查）。

### 8.3 测试运行注意事项

- RAG 检索测试会实际加载重排序模型（如果本地路径存在），首次运行可能耗时较长。
- 评估报告会自动输出到 `tests/rag_eval_report.json`。
- 若 DashScope API 不可用，`conftest.py` 和 `test_rag_retrieval.py` 中均有 TF-IDF 降级逻辑，确保测试不依赖外部网络。

---

## 9. 安全注意事项

### 9.1 API Key 管理

- **`.env` 文件包含真实 API Key，绝不要将其提交到版本控制。**
- `pyproject.toml` 未将 `.env` 列入打包文件，但仍需在 `.gitignore` 中明确排除（如未排除请补充）。
- Materials Project API Key 优先读取环境变量 `MP_API_KEY`，其次回退到 `config/mp_config.yaml` 中的 `mp_api_key`（通常为空字符串，由环境变量覆盖）。

### 9.2 CORS 配置

- `api_server.py` 中当前配置 `allow_origins=["*"]`，这是**开发环境设置**。
- 生产环境部署时，必须将 `allow_origins` 限制为具体的前端域名。

### 9.3 MCP 安全

- `config/mcp_config.yaml` 中定义了敏感操作关键词列表（`download`、`write`、`delete`、`exec`）。
- Agent 层已实现 HITL（Human-in-the-loop）敏感操作审批：`human_approval_node` 会在执行敏感工具（如 `mp_download`）前调用 `interrupt()` 挂起执行，前端收到 `approval_required` SSE 事件后需调用 `/api/v1/agent/approve` 恢复。
- 文件系统 MCP Server (`filesystem`) 默认 `enabled: false`，启用前请确认挂载路径范围。

### 9.4 输入安全

- API 层使用 Pydantic 模型校验所有入参：`question` 限制长度 1-10000，`top_k` 限制 1-20。
- Agent 工具调用时，工具入参由 LLM 生成，需确保工具函数内部对参数做二次校验（如 `mp_downloader` 中的化学式校验）。

---

## 10. 常见问题与排查

| 问题现象 | 可能原因 | 解决方式 |
|----------|----------|----------|
| 启动时 Embedding 模型初始化失败 | `DASHSCOPE_API_KEY` 未设置或欠费 | 检查 `.env`，或观察降级为 TF-IDF 的警告日志 |
| ChromaDB 检索无结果 | `score_threshold` 过高或文档未加载 | 检查 `config/chroma_config.yaml` 中的 `score_threshold`，运行 `vector_store.load_document()` |
| MCP 工具未加载 | MCP Server 进程启动失败或配置 `enabled: false` | 检查 `config/mcp_config.yaml`，查看 lifespan 启动日志 |
| 重排序器初始化失败 | 本地模型路径不存在 | 检查 `chroma_config.yaml` 中的 `reranker_model` 路径，或改为 HuggingFace Hub 模型名 |
| SQLite database is locked | 并发写冲突 | SessionStore 已内置随机抖动重试，若仍失败请检查是否有其他进程持有数据库连接 |

---

## 11. 扩展指南

### 11.1 添加新的 Agent 工具

1. 在 `agent/tools/` 下新建工具模块（或直接在 `agent_tools.py` 中添加）。
2. 使用 `@tool` 装饰器定义函数，编写清晰的 `description`。
3. 在 `agent_tools.py` 的 `get_native_tools()` 中注册。
4. 更新 `prompt/system_prompt.txt`，在「可使用工具及能力边界」章节添加新工具说明。

### 11.2 添加新的 MCP Server

1. 在 `mcp_servers/tools/` 下新建 Server 模块，继承/使用 `BaseMCPServer`。
2. 在 `config/mcp_config.yaml` 的 `servers` 列表中添加配置（`name`、`command`、`transport`、`enabled`）。
3. 重启 API 服务，lifespan 启动时会自动发现并加载。

### 11.3 添加新的 API 路由

1. 在 `api/routes/` 下新建路由模块，使用 `APIRouter`。
2. 在 `api_server.py` 中通过 `app.include_router()` 注册，前缀为 `/api/v1`。
3. 在 `api/schemas.py` 中定义对应的 Pydantic 模型。
4. 在 `api/dependencies.py` 中添加需要的懒加载依赖（如有）。

---

*本文件基于项目实际代码和配置生成，最后更新日期：2026-05-17。*
