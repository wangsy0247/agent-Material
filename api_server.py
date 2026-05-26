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

import asyncio
import os
import sys

import numpy  # noqa: F401

from utils.path_tools import get_project_path
from dotenv import load_dotenv

load_dotenv() 

# 确保项目根目录在 Python 路径中（兼容多种启动方式）
PROJECT_ROOT = get_project_path()

# 从环境变量读取端口（WSL 下可自由切换避免冲突）
API_PORT = int(os.environ.get("AGENT_API_PORT"))
API_HOST = os.environ.get("AGENT_API_HOST")

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

from api.routes import agent, rag, system, memory, auth
from utils.logger import logger


# ==================== MCP 健康检查后台任务 ====================

async def _mcp_health_monitor() -> None:
    """定期检查 MCP 连接健康，不健康则触发重连。"""
    await asyncio.sleep(5)  # 启动后稍等片刻再开始检查

    while True:
        try:
            from agent.tools.mcp_adapter import _adapters
            from utils.config_handler import agent_config

            servers_cfg = {
                cfg.get("name", ""): cfg
                for cfg in agent_config.get("mcp_servers", [])
            }

            for name, conn in list(_adapters.items()):
                # 跳过已禁用的 server
                cfg = servers_cfg.get(name)
                if cfg is not None and not cfg.get("enabled", True):
                    continue

                healthy = await conn._check_health()
                if not healthy and cfg is not None:
                    success = await conn.reconnect(cfg)
                    if not success and conn._reconnect_attempts >= conn.max_reconnect_attempts:
                        logger.warning(
                            f"[MCPHealth] '{name}' 已达最大重连次数，"
                            f"已标记为离线 | 最后错误: {conn._last_error}"
                        )

            # 默认每 30 秒检查一次（可从配置读取）
            interval = 30
            try:
                from utils.config_handler import agent_config
                interval = agent_config.get("mcp_global", {}).get("health_check_interval", 30)
            except Exception:
                pass
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("[MCPHealth] 健康检查任务已取消")
            raise
        except Exception as e:
            logger.error(f"[MCPHealth] 健康检查循环异常: {e}")
            await asyncio.sleep(30)


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

    from utils.config_handler import agent_config

    # MCP 模式：连接多个远程 MCP Server
    mp_tool_mode = agent_config.get("mp_tool_mode", "local")
    _mcp_health_task: asyncio.Task | None = None

    if mp_tool_mode == "mcp":
        servers = agent_config.get("mcp_servers", [])
        if servers:
            logger.info(f"🔗 初始化 {len(servers)} 个 MCP Server 连接...")
            try:
                from agent.tools.mcp_adapter import init_mcp_tools
                tools = await init_mcp_tools(servers)
                logger.info(f"✅ MCP 连接完成 | 远程工具={[t.name for t in tools]}")

                # 启动 MCP 健康检查后台任务
                _mcp_health_task = asyncio.create_task(
                    _mcp_health_monitor(), name="mcp-health-monitor"
                )
                logger.info("🩺 MCP 健康检查后台任务已启动")
            except Exception as e:
                logger.warning(f"⚠️ MCP 连接失败: {e} | Agent 将回退到本地模式")
        else:
            logger.warning("⚠️ MCP 模式已启用但未配置 mcp_servers，回退到本地模式")
    else:
        logger.info("🔧 MP 工具模式: 本地直连")

    # 预热 RAG 组件（主线程加载，确保在请求到来前完成）
    logger.info("⏳ 预热 RAG 组件（reranker / BM25）...")
    try:
        from agent.tools.agent_tools import _get_rag_service
        _get_rag_service()
        logger.info("✅ RAG 组件预热完成")
    except Exception as e:
        logger.warning(f"⚠️ RAG 组件预热失败: {e} | 将在首次请求时懒加载")

    logger.info("=" * 50)
    yield

    # ===== Shutdown =====
    logger.info("🛑 服务关闭中...")
    if _mcp_health_task is not None and not _mcp_health_task.done():
        _mcp_health_task.cancel()
        try:
            await _mcp_health_task
        except asyncio.CancelledError:
            pass
        logger.info("✅ MCP 健康检查任务已取消")

    if mp_tool_mode == "mcp":
        try:
            from agent.tools.mcp_adapter import close_mcp
            await close_mcp()
            logger.info("✅ MCP 连接已关闭")
        except Exception as e:
            logger.warning(f"⚠️ MCP 关闭失败: {e}")
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
app.include_router(auth.router, prefix=API_PREFIX)


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
            "auth_login": "/api/v1/auth/login",
            "auth_register": "/api/v1/auth/register",
        },
    }


# ==================== 入口 ====================

if __name__ == "__main__":
    import uvicorn

    # 热重载开关（默认关闭）
    # 注意：WSL + PyTorch/ONNX/ChromaDB 环境下，reload=True 可能导致多进程 fork 时的
    # Segmentation Fault。如需热重载，请设置环境变量 AGENT_API_RELOAD=true
    RELOAD = os.environ.get("AGENT_API_RELOAD", "false").lower() == "true"

    uvicorn.run(
        "api_server:app",
        host=API_HOST,
        port=API_PORT,
        reload=RELOAD,
        log_level="info",
    )
