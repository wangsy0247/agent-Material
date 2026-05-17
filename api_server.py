"""
FastAPI 主应用入口
材料科学 LLM Agent REST API 服务

功能：
- Agent 对话（SSE 流式）
- RAG 知识库查询（SSE 流式 + 非流式检索）
- 健康检查、统计信息

启动方式：
    python api_server.py              # 开发模式
    uvicorn api_server:app --reload   # 热重载开发
    uvicorn api_server:app --host 0.0.0.0 --port 8000  # 生产模式

API 文档：
    http://localhost:8000/docs        # Swagger UI
    http://localhost:8000/redoc       # ReDoc
"""

import os
import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中（兼容多种启动方式）
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 从环境变量读取端口，默认 8000（WSL 下可自由切换避免冲突）
API_PORT = int(os.environ.get("AGENT_API_PORT", 8000))
API_HOST = os.environ.get("AGENT_API_HOST", "0.0.0.0")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

# 离线 Swagger UI（不依赖外网 CDN）
try:
    from fastapi_offline import FastAPIOffline
    FASTAPI_OFFLINE_AVAILABLE = True
except ImportError:
    FASTAPI_OFFLINE_AVAILABLE = False

from api.routes import agent, rag, system, memory
from utils.logger import logger


# ==================== 生命周期管理 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期事件

    startup: 服务启动时执行（如加载模型、预热连接、初始化 MCP 工具）
    shutdown: 服务关闭时执行（如释放资源、关闭连接）
    """
    # ===== Startup =====
    logger.info("=" * 50)
    logger.info("🚀 材料科学 LLM Agent API 服务启动中...")
    logger.info(f"📁 项目根目录: {PROJECT_ROOT}")

    # 初始化 MCP 工具（连接配置中启用的 MCP Server）
    try:
        from agent.tools.mcp_tool_manager import init_mcp_tools
        mcp_tools = await init_mcp_tools()
        logger.info(f"✅ MCP 工具初始化完成 | 共 {len(mcp_tools)} 个")
    except Exception as e:
        logger.warning(f"⚠️ MCP 工具初始化失败: {e} | Agent 将仅使用原生工具")

    # 延迟加载关键组件（首次请求时才会真正初始化）
    logger.info("⏳ Agent 和 RAG 组件将在首次请求时懒加载")

    logger.info("=" * 50)
    yield

    # ===== Shutdown =====
    logger.info("🛑 服务关闭中...")

    # 关闭 MCP 长连接（连接池释放）
    try:
        from agent.tools.mcp_tool_manager import close_mcp_connections
        await close_mcp_connections()
        logger.info("✅ MCP 连接已关闭")
    except Exception as e:
        logger.warning(f"⚠️ MCP 连接关闭失败: {e}")

    logger.info("=" * 50)


# ==================== FastAPI 应用实例 ====================

_FastAPIClass = FastAPIOffline if FASTAPI_OFFLINE_AVAILABLE else FastAPI

app = _FastAPIClass(
    title="材料科学 LLM Agent API",
    description="""
    材料科学领域的大模型 Agent 和 RAG 知识库 REST API 服务。
    
    ## 核心功能
    - **Agent 对话**：ReAct 智能体，支持工具调用（Materials Project 查询/下载、RAG 检索）
    - **RAG 知识库**：混合检索（向量 + BM25）+ CrossEncoder 重排序
    - **SSE 流式输出**：大模型生成过程实时推送
    
    ## 技术栈
    - FastAPI + Uvicorn
    - LangChain + LangGraph
    - ChromaDB + BM25 + CrossEncoder
    """,
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ==================== 中间件配置 ====================

# CORS：允许前端跨域调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip 压缩：减少响应体积（对非 SSE 请求有效）
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ==================== 路由注册 ====================

# API v1 版本前缀
API_PREFIX = "/api/v1"

app.include_router(agent.router, prefix=API_PREFIX)
app.include_router(rag.router, prefix=API_PREFIX)
app.include_router(system.router, prefix=API_PREFIX)
app.include_router(memory.router, prefix=API_PREFIX)


# ==================== 静态文件 & 前端页面 ====================

# 挂载静态资源（CSS/JS）
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

@app.get("/", summary="前端页面", tags=["根路径"])
async def root():
    """返回前端聊天界面"""
    return FileResponse("frontend/index.html")


# ==================== API 信息路由 ====================


@app.get("/api/v1", summary="API v1 信息", tags=["根路径"])
async def api_info():
    """返回 API 版本信息"""
    return {
        "version": "v1",
        "endpoints": {
            "agent": "/api/v1/agent/chat",
            "rag_query": "/api/v1/rag/query",
            "rag_retrieve": "/api/v1/rag/retrieve",
            "memory_sessions": "/api/v1/memory/sessions",
            "memory_recall": "/api/v1/memory/recall",
            "health": "/api/v1/system/health",
            "stats": "/api/v1/system/rag/stats",
        },
    }


# ==================== 入口 ====================

if __name__ == "__main__":
    import uvicorn
    
    # 开发模式启动（生产环境建议使用 gunicorn + uvicorn.workers.UvicornWorker）
    uvicorn.run(
        "api_server:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,  # 开发模式热重载
        log_level="info",
    )
