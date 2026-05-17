# 材料科学 LLM Agent

> 面向材料科学领域的大语言模型 Agent 系统，支持 ReAct 智能体对话、RAG 知识库检索、Materials Project 数据查询与下载。

---

## 技术栈

| 层级 | 技术 |
|------|------|
| Web 框架 | FastAPI + Uvicorn |
| LLM 框架 | LangChain + LangGraph |
| 大模型 | ChatAnthropic (DeepSeek-V4-Flash) |
| Embedding | DashScopeEmbeddings (text-embedding-v4) |
| 向量数据库 | ChromaDB |
| 检索增强 | BM25 + CrossEncoder 重排序 |
| 材料计算 | pymatgen + mp-api |
| 记忆系统 | 三层记忆（短期/中期/长期） |

---

## 目录结构

```
├── agent/              # ReAct Agent 核心
├── api/                # FastAPI 路由层
├── rag/                # RAG 知识库
├── memory/             # 三层记忆系统
├── mcp_servers/        # MCP 工具基础设施
├── model/              # 模型工厂
├── config/             # YAML 配置文件
├── frontend/           # Web 前端（纯静态）
├── data/               # 知识库文档（PDF/TXT/MD）
├── tests/              # 测试目录
├── api_server.py       # FastAPI 主入口
└── pyproject.toml      # 项目依赖
```

---

## 环境准备

### 1. Python 版本要求

```bash
python --version
# 要求 >= 3.10
```

### 2. 安装项目依赖

```bash
cd /path/to/agent-material
pip install -e .
```

---

## Git 仓库初始化与 SSH 配置

### 1. 初始化本地仓库

```bash
cd /path/to/agent-material
git init
git add .
git commit -m "Initial commit"
```

### 2. 生成 SSH 密钥（免密推送）

如果你还没有配置 SSH 密钥，执行以下命令生成：

```bash
ssh-keygen -t ed25519 -C "your-email@example.com" -f ~/.ssh/id_ed25519 -N ""
```

> `-t ed25519` 使用更安全的 Ed25519 算法，`-N ""` 表示不设密码（方便自动化推送）。

启动 ssh-agent 并添加密钥：

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

查看公钥内容：

```bash
cat ~/.ssh/id_ed25519.pub
```

输出示例（你的公钥会不同）：

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPPl8SWl0+Z8fzgPyrfzKC4qsH9pVt4TnecyXtU2xcyI agent-material@wsy
```

### 3. 添加公钥到 GitHub

1. 打开 https://github.com/settings/keys
2. 点击 **New SSH key**
3. Title 填写任意名称，例如 `WSL-Local`
4. Key type 保持 `Authentication Key`
5. 将上一步 `cat ~/.ssh/id_ed25519.pub` 输出的内容完整粘贴到 Key 文本框
6. 点击 **Add SSH key**

### 4. 测试 SSH 连接

```bash
ssh -T git@github.com
```

预期输出：

```
Hi xxx! You've successfully authenticated, but GitHub does not provide shell access.
```

### 5. 在 GitHub 创建私有仓库并推送

1. 打开 https://github.com/new
2. Repository name 填写 `agent-material`
3. Visibility 选择 **Private**
4. 不要勾选 README 和 .gitignore（本地已有）
5. 点击 Create repository

回到终端，关联远程仓库并推送：

```bash
# 将 YOUR_USERNAME 替换为你的 GitHub 用户名
git remote add origin git@github.com:YOUR_USERNAME/agent-material.git
git branch -M main
git push -u origin main
```

> 使用 SSH 地址 `git@github.com:...` 推送时**不需要输入 Personal Access Token**。

---

## 项目配置

在项目根目录创建 `.env` 文件，填入你的 API Key：

```bash
# 大模型（DeepSeek 通过 Anthropic 协议调用）
ANTHROPIC_API_KEY=your-api-key
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic

# Embedding API（阿里云 DashScope）
DASHSCOPE_API_KEY=your-api-key

# Materials Project
MP_API_KEY=your-api-key

# LangSmith 可观测性（可选）
LANGCHAIN_API_KEY=your-api-key

# 服务端口
AGENT_API_PORT=8000
AGENT_API_HOST=0.0.0.0
```

> `.env` 文件已被 `.gitignore` 排除，不会提交到 GitHub。

---

## 启动服务

### 方式一：启动脚本（推荐）

```bash
# 开发模式（热重载）
bash scripts/start_server.sh dev

# 生产模式
bash scripts/start_server.sh prod
```

### 方式二：直接启动

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

### 方式三：Python 入口

```bash
python api_server.py
```

---

## 访问前端与 API

服务启动后，浏览器访问：

| 地址 | 说明 |
|------|------|
| http://localhost:8000/ | **前端聊天界面**（三栏式 UI） |
| http://localhost:8000/docs | API 文档（Swagger UI） |
| http://localhost:8000/redoc | API 文档（ReDoc） |

> 前端为纯静态页面，所有依赖（Markdown 渲染、代码高亮）均已本地化，**无需翻墙**即可使用。

---

## 核心 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/agent/chat` | Agent 对话（SSE 流式） |
| POST | `/api/v1/rag/query` | RAG 知识库问答 |
| POST | `/api/v1/rag/retrieve` | RAG 检索（仅返回文献） |
| GET | `/api/v1/memory/sessions` | 列出会话 |
| GET | `/api/v1/memory/recall` | 长期记忆召回 |
| GET | `/api/v1/system/health` | 健康检查 |
| GET | `/api/v1/system/rag/stats` | RAG 统计信息 |

---

## 日常 Git Workflow

配置好 SSH 后，日常开发只需三步：

```bash
# 1. 查看修改
git status

# 2. 添加并提交
git add .
git commit -m "feat: 新增 xxx 功能"

# 3. 推送到 GitHub
git push
```

### 常用提交类型前缀

| 前缀 | 含义 |
|------|------|
| `feat:` | 新功能 |
| `fix:` | 修复 bug |
| `docs:` | 文档更新 |
| `refactor:` | 重构 |
| `test:` | 测试相关 |
| `chore:` | 构建/工具改动 |

---

## 测试

```bash
# 运行全部测试
pytest -v

# 运行特定测试
pytest tests/test_rag_retrieval.py -v -s
pytest tests/test_mp_downloader.py -v
```

---

## 常见问题

| 问题 | 解决方式 |
|------|---------|
| 启动时 Embedding 失败 | 检查 `.env` 中 `DASHSCOPE_API_KEY` |
| ChromaDB 检索无结果 | 检查 `config/chroma_config.yaml` 中的 `score_threshold` |
| SQLite database is locked | 检查是否有其他进程占用数据库 |
| 前端页面样式异常 | 确认已执行 `pip install -e .` 安装全部依赖 |
| 推送时提示 Permission denied | 确认 SSH 公钥已添加到 GitHub，且远程地址为 SSH 格式 |

---

*最后更新：2026-05-17*
