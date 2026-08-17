"""
FastAPI 主应用入口
材料科学 LLM Agent REST API 服务

功能：
- Agent 对话（SSE 流式）
- RAG 知识库查询（SSE 流式 + 非流式检索）
- 健康检查、统计信息

启动方式：
    uvicorn backend.app.gateway.app:app --reload   # 热重载开发
    uvicorn backend.app.gateway.app:app --host 0.0.0.0 --port 8000  # 生产模式
    python -m backend.app.gateway.app               # 模块方式运行

API 文档：
    http://localhost:8000/docs        # Swagger UI
    http://localhost:8000/redoc       # ReDoc
"""

import asyncio
import os
import sys

import numpy  # noqa: F401
from pathlib import Path

# 确保 backend/packages 在 Python 路径中（用于导入 agent_material）
_backend_dir = Path(__file__).resolve().parent.parent.parent
_packages_dir = _backend_dir / "packages"
if str(_packages_dir) not in sys.path:
    sys.path.insert(0, str(_packages_dir))

from agent_material.utils.path_tools import get_project_path
from dotenv import load_dotenv

load_dotenv()

# 确保项目根目录在 Python 路径中（兼容多种启动方式）
PROJECT_ROOT = get_project_path()

# 从环境变量读取端口（WSL 下可自由切换避免冲突）
API_PORT = int(os.environ.get("AGENT_API_PORT", "8000"))
API_HOST = os.environ.get("AGENT_API_HOST", "127.0.0.1")

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

from .routers import agent, rag, system, memory, auth, mcp as mcp_router
from agent_material.observability.structured_logger import get_logger

logger = get_logger("gateway.app")


# ==================== MCP 健康检查后台任务 ====================

async def _mcp_health_monitor() -> None:
    """定期检查 MCP 连接健康，不健康则触发重连。

    deer-flow 风格：通过检测配置文件 mtime 变化来触发缓存失效，
    由 session_pool 和 MultiServerMCPClient 内部管理连接健康。
    """
    await asyncio.sleep(5)

    while True:
        try:
            from agent_material.mcp.cache import _is_cache_stale, reset_mcp_tools_cache

            if _is_cache_stale():
                logger.info("[MCPHealth] 检测到配置文件变更，重置 MCP 缓存...")
                reset_mcp_tools_cache()

            await asyncio.sleep(30)
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
    logger.info("=" * 50, extra={"event": "startup_begin"})

    # 认证配置属于启动硬依赖：禁止带默认 JWT 密钥运行。
    from .auth.auth import validate_auth_configuration

    validate_auth_configuration()

    # ── 可观测性初始化 ──
    try:
        from agent_material.observability import init_all
        status = init_all(service_name="agent_material")
        logger.info("observability_init", extra=status)
    except Exception:
        logger.warning("可观测性初始化失败，使用默认配置", exc_info=True)

    from agent_material.utils.config_handler import agent_config

    # MP 工具模式: local (RAG 进程内直连 + MCP 其他工具) / mcp (全部 MCP)
    mp_tool_mode = agent_config.get("mp_tool_mode", "local")
    _mcp_health_task: asyncio.Task | None = None

    if mp_tool_mode == "local":
        logger.info("🔧 MP 工具模式: 本地直连（RAG 工具进程内调用 agent_material.rag_engine）")

    # MCP 工具初始化（material-project 等非 RAG 工具仍通过 MCP 加载）
    try:
        from agent_material.mcp.cache import initialize_mcp_tools
        tools = await initialize_mcp_tools()
        logger.info(f"✅ MCP 连接完成 | 远程工具={[t.name for t in tools]}")

        _mcp_health_task = asyncio.create_task(
            _mcp_health_monitor(), name="mcp-health-monitor"
        )
        logger.info("🩺 MCP 健康检查后台任务已启动")
    except Exception as e:
        logger.warning(f"⚠️ MCP 连接失败: {e} | Agent 将仅使用本地工具")

    # ── RAG 运行时初始化（WRK-001）──
    # 初始化进程内统一 VectorStore 单例（REST/RagService/Agent Tool 共享），
    # 并启动 PG Worker：启动后立即领取 pending/租约过期任务，不依赖新写请求。
    # 失败只记日志降级（RAG 不可用不应拖垮整个网关）。
    try:
        from agent_material.rag_engine.vector_store import get_vector_store

        rag_store = get_vector_store()
        rag_store.start_worker()
        logger.info("✅ RAG 运行时初始化完成，PG Worker 已启动（自动恢复待处理任务）")
    except Exception as e:
        logger.warning(f"⚠️ RAG 运行时初始化失败（RAG 功能降级，不影响网关其余功能）: {e}")

    logger.info("=" * 50)
    yield

    # ===== Shutdown =====
    logger.info("🛑 服务关闭中...")

    # ── 停止 RAG 后台 Worker（WRK-001）──
    # 停止领取新任务，等待当前任务到安全点退出；peek 避免未初始化时触发构造。
    try:
        from agent_material.rag_engine.vector_store import peek_vector_store

        store = peek_vector_store()
        if store is not None:
            store.stop_worker(timeout=10.0)
            logger.info("✅ RAG Worker 已停止")
    except Exception as e:
        logger.warning(f"⚠️ RAG Worker 停止失败: {e}")
    if _mcp_health_task is not None and not _mcp_health_task.done():
        _mcp_health_task.cancel()
        try:
            await _mcp_health_task
        except asyncio.CancelledError:
            pass
        logger.info("✅ MCP 健康检查任务已取消")

    try:
        from agent_material.mcp.cache import reset_mcp_tools_cache
        reset_mcp_tools_cache()
        logger.info("✅ MCP 缓存已重置")
    except Exception as e:
        logger.warning(f"⚠️ MCP 缓存重置失败: {e}")

    # 关闭 Checkpointer 持久化资源
    try:
        from agent_material.agents.graph.checkpointer_factory import close_checkpointer_resources
        await close_checkpointer_resources()
    except Exception as e:
        logger.warning(f"⚠️ Checkpointer 资源关闭失败: {e}")

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
    - PGVector + BM25 + CrossEncoder
    """,
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ==================== 中间件配置 ====================

# CORS 默认仅允许本机页面。部署到其他域名时必须显式提供白名单。
_default_cors_origins = "http://localhost:8000,http://127.0.0.1:8000"
_cors_origins = [
    origin.strip()
    for origin in os.environ.get(
        "AGENT_CORS_ORIGINS", _default_cors_origins
    ).split(",")
    if origin.strip()
]
_cors_allow_credentials = "*" not in _cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip 压缩：减少响应体积（对非 SSE 请求有效）
app.add_middleware(GZipMiddleware, minimum_size=1000)

# OpenTelemetry HTTP 中间件：自动为每个 HTTP 请求创建 Span
try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    FastAPIInstrumentor.instrument_app(
        app,
        excluded_urls="/metrics,/health,/docs,/redoc,/openapi.json",
    )
    logger.info("otel_fastapi_instrumentation_enabled")
except ImportError:
    logger.info("opentelemetry-instrumentation-fastapi 未安装，跳过 HTTP 自动追踪")

# ==================== 路由注册 ====================

# API v1 版本前缀
API_PREFIX = "/api/v1"

app.include_router(agent.router, prefix=API_PREFIX)
app.include_router(rag.router, prefix=API_PREFIX)
app.include_router(system.router, prefix=API_PREFIX)
app.include_router(memory.router, prefix=API_PREFIX)
app.include_router(auth.router, prefix=API_PREFIX)
# deer-flow 风格：MCP 路由使用 /api 前缀（非 /api/v1）
app.include_router(mcp_router.router)


# ==================== 静态文件 & 前端页面 ====================

# 挂载静态资源（CSS/JS）
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

@app.get("/", summary="前端页面", tags=["根路径"])
async def root():
    """返回前端聊天界面"""
    return FileResponse("frontend/index.html")


@app.get("/rag", summary="RAG 管理页面", tags=["根路径"])
async def rag_page():
    """返回 RAG 知识库管理页面"""
    return FileResponse("frontend/rag.html")


# ==================== API 信息路由 ====================


@app.get("/api/v1", summary="API v1 信息", tags=["根路径"])
async def api_info():
    """返回 API 版本信息"""
    return {
        "version": "v1",
        "endpoints": {
            "agent": "/api/v1/agent/chat",
            "memory_sessions": "/api/v1/memory/sessions",
            "memory_recall": "/api/v1/memory/recall",
            "health": "/api/v1/system/health",
            "stats": "/api/v1/system/rag/stats",
            "metrics": "/metrics",
            "auth_login": "/api/v1/auth/login",
            "auth_register": "/api/v1/auth/register",
            "mcp_config": "/api/mcp/config",
            "mcp_status": "/api/mcp/status",
        },
    }


# ==================== Prometheus /metrics ====================


@app.get("/metrics", summary="Prometheus 指标", include_in_schema=False)
async def metrics():
    """Prometheus 格式的指标端点（可选，需自行部署 Prometheus 抓取）。"""
    from fastapi.responses import Response
    try:
        from agent_material.observability.metrics import get_metrics
        return Response(content=get_metrics(), media_type="text/plain; charset=utf-8")
    except ImportError:
        return Response(content="# prometheus_client 未安装\n", media_type="text/plain")


# ==================== 入口 ====================

if __name__ == "__main__":
    import uvicorn

    # 热重载开关（默认关闭）
    # 注意：WSL + PyTorch/ONNX/PGVector 环境下，reload=True 可能导致多进程 fork 时的
    # Segmentation Fault。如需热重载，请设置环境变量 AGENT_API_RELOAD=true
    RELOAD = os.environ.get("AGENT_API_RELOAD", "false").lower() == "true"

    uvicorn.run(
        "backend.app.gateway.app:app",
        host=API_HOST,
        port=API_PORT,
        reload=RELOAD,
        log_level="info",
    )
