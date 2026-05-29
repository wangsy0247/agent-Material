"""
LangGraph Checkpointer 工厂

支持多种持久化后端：
- postgres: PostgreSQL 持久化（企业推荐，需 PostgreSQL 服务）
- sqlite: SQLite 持久化（立即可用，无需额外服务，推荐单机部署）
- memory: 内存存储（开发和测试环境，默认降级）

配置来源：config/agent_config.yaml -> checkpointer
"""

from __future__ import annotations

import asyncio
import inspect
import os
import sqlite3
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from agent_material.utils.config_handler import agent_config
from agent_material.utils.logger import logger

# ---- 配置读取 ----
_CP_CFG: dict[str, Any] = agent_config.get("checkpointer", {})
_CP_TYPE: str = _CP_CFG.get("type", "memory")

# 全局资源引用（用于 lifespan shutdown 时释放）
_pool_instance: Any = None
_sqlite_conn: sqlite3.Connection | None = None


def create_checkpointer() -> Any:
    """根据配置创建并返回 checkpointer 实例。

    优先级：
    1. 按配置类型尝试创建（postgres -> sqlite -> memory）
    2. 当指定类型创建失败时，自动降级到下一级
    """
    cp_type = _CP_TYPE

    if cp_type == "postgres":
        cp = _create_postgres_checkpointer()
        if cp is not None:
            return cp
        logger.warning("[CheckpointerFactory] PostgreSQL 不可用，降级到 SQLite")
        cp_type = "sqlite"

    if cp_type == "sqlite":
        cp = _create_sqlite_checkpointer()
        if cp is not None:
            return cp
        logger.warning("[CheckpointerFactory] SQLite 不可用，降级到 MemorySaver")

    logger.info("[CheckpointerFactory] 使用 MemorySaver（内存模式）")
    return MemorySaver()


def _create_sqlite_checkpointer() -> Any | None:
    """创建 SQLite checkpointer，失败时返回 None。"""
    global _sqlite_conn
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        logger.warning(
            "[CheckpointerFactory] langgraph-checkpoint-sqlite 未安装，"
            "如需 SQLite 持久化，请执行: pip install langgraph-checkpoint-sqlite"
        )
        return None

    db_path: str = _CP_CFG.get("sqlite", {}).get("db_path", "logs/langgraph_checkpoints.db")
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    try:
        _sqlite_conn = sqlite3.connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn=_sqlite_conn)
        checkpointer.setup()
        logger.info(f"[CheckpointerFactory] 使用 SqliteSaver | path={db_path}")
        return checkpointer
    except Exception as e:
        logger.warning(f"[CheckpointerFactory] SQLite 初始化失败: {e}")
        if _sqlite_conn is not None:
            try:
                _sqlite_conn.close()
            except Exception:
                pass
            _sqlite_conn = None
        return None


def _create_postgres_checkpointer() -> Any | None:
    """创建 PostgreSQL checkpointer，失败时返回 None。

    注意：LangGraph 的 async astream 需要 checkpointer 实现 aget_tuple 等异步方法。
    langgraph-checkpoint-postgres 3.x 将同步/异步实现拆分：
    - PostgresSaver（同步）
    - AsyncPostgresSaver（异步，在 aio 子模块中）
    因此必须使用 AsyncPostgresSaver 才能配合 async Agent 使用。

    兼容策略：
    - 如果在异步上下文（有 running loop）中调用，直接创建并后台调度 setup
    - 如果在同步上下文（无 running loop）中调用，使用 asyncio.run() 完成初始化
    """
    global _pool_instance
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg_pool import AsyncConnectionPool
    except ImportError:
        logger.warning(
            "[CheckpointerFactory] langgraph-checkpoint-postgres 未安装，"
            "如需 PostgreSQL 持久化，请执行: pip install langgraph-checkpoint-postgres"
        )
        return None

    conn_str = _get_postgres_conn_string()
    if not conn_str:
        logger.warning(
            "[CheckpointerFactory] PostgreSQL 连接字符串未配置，"
            "请在 config/agent_config.yaml 中设置 checkpointer.postgres.connection_string"
        )
        return None

    # 尝试在有 running loop 时直接创建（FastAPI async dependency 路径）
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        # 异步上下文：直接创建对象，后台调度 open + setup
        try:
            _pool_instance = AsyncConnectionPool(
                conn_str,
                min_size=1,
                max_size=5,
                open=False,
                kwargs={"autocommit": True},
            )
            checkpointer = AsyncPostgresSaver(conn=_pool_instance)

            async def _async_init() -> None:
                await _pool_instance.open()
                await checkpointer.setup()

            loop.create_task(_async_init())
            logger.info(
                f"[CheckpointerFactory] 使用 AsyncPostgresSaver | "
                f"host={conn_str.split('@')[-1].split('/')[0]}"
            )
            return checkpointer
        except Exception as e:
            logger.warning(f"[CheckpointerFactory] PostgreSQL 初始化失败: {e}")
            _close_pool(_pool_instance)
            _pool_instance = None
            return None

    # 无 running loop：使用 asyncio.run() 完整初始化（线程池路径）
    async def _async_init():
        pool = AsyncConnectionPool(
            conn_str,
            min_size=1,
            max_size=5,
            open=False,
            kwargs={"autocommit": True},
        )
        cp = AsyncPostgresSaver(conn=pool)
        await pool.open()
        await cp.setup()
        return cp, pool

    try:
        checkpointer, _pool_instance = asyncio.run(_async_init())
        logger.info(
            f"[CheckpointerFactory] 使用 AsyncPostgresSaver | "
            f"host={conn_str.split('@')[-1].split('/')[0]}"
        )
        return checkpointer
    except Exception as e:
        logger.warning(f"[CheckpointerFactory] PostgreSQL 连接失败: {e}")
        _close_pool(_pool_instance)
        _pool_instance = None
        return None


def _get_postgres_conn_string() -> str:
    """获取 PostgreSQL 连接字符串。"""
    pg_cfg = _CP_CFG.get("postgres", {})
    conn_str: str = pg_cfg.get("connection_string", "")

    # 支持 ${ENV_VAR:-default} 语法
    if conn_str.startswith("${") and conn_str.endswith("}"):
        inner = conn_str[2:-1]
        env_var = inner.split(":-")[0]
        default_val = inner.split(":-", 1)[1] if ":-" in inner else ""
        conn_str = os.getenv(env_var, default_val)
    elif not conn_str:
        conn_str = os.getenv("POSTGRES_CHECKPOINTER_URL", "")

    return conn_str


def _close_pool(pool: Any) -> None:
    """关闭连接池（兼容同步/异步）。"""
    if pool is None:
        return
    try:
        if inspect.iscoroutinefunction(pool.close):
            try:
                asyncio.get_running_loop().create_task(pool.close())
            except RuntimeError:
                pass
        else:
            pool.close()
    except Exception:
        pass


async def close_checkpointer_resources() -> None:
    """关闭 checkpointer 占用的资源（在应用 shutdown 时调用）。"""
    global _pool_instance, _sqlite_conn

    if _pool_instance is not None:
        try:
            if inspect.iscoroutinefunction(_pool_instance.close):
                await _pool_instance.close()
            else:
                _pool_instance.close()
            logger.info("[CheckpointerFactory] PostgreSQL 连接池已关闭")
        except Exception as e:
            logger.warning(f"[CheckpointerFactory] 关闭 PostgreSQL 连接池失败: {e}")
        _pool_instance = None

    if _sqlite_conn is not None:
        try:
            _sqlite_conn.close()
            logger.info("[CheckpointerFactory] SQLite 连接已关闭")
        except Exception as e:
            logger.warning(f"[CheckpointerFactory] 关闭 SQLite 连接失败: {e}")
        _sqlite_conn = None
