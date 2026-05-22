"""
LangGraph Checkpointer 工厂

支持多种持久化后端：
- postgres: PostgreSQL 持久化（企业推荐，需安装 langgraph-checkpoint-postgres）
- memory: 内存存储（开发和测试环境，默认降级）

配置来源：config/agent_config.yaml -> checkpointer
"""

from __future__ import annotations

import os
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from utils.config_handler import agent_config
from utils.logger import logger

# ---- 配置读取 ----
_CP_CFG: dict[str, Any] = agent_config.get("checkpointer", {})
_CP_TYPE: str = _CP_CFG.get("type", "memory")
_PG_CFG: dict[str, Any] = _CP_CFG.get("postgres", {})


def create_checkpointer() -> Any:
    """根据配置创建并返回 checkpointer 实例。

    优先级：
    1. postgres: 使用 PostgresSaver（持久化，需 PostgreSQL + langgraph-checkpoint-postgres）
    2. memory: 使用 MemorySaver（内存，默认降级）

    当 postgres 依赖未安装或连接失败时，自动降级到 MemorySaver。
    """
    if _CP_TYPE == "postgres":
        return _create_postgres_checkpointer()

    # 默认：内存 checkpointer
    logger.info("[CheckpointerFactory] 使用 MemorySaver（内存模式）")
    return MemorySaver()


def _create_postgres_checkpointer() -> Any:
    """创建 PostgreSQL checkpointer，失败时降级到 MemorySaver。"""
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError:
        logger.warning(
            "[CheckpointerFactory] langgraph-checkpoint-postgres 未安装，"
            "自动降级到 MemorySaver。如需 PostgreSQL 持久化，请执行:\n"
            "  pip install langgraph-checkpoint-postgres"
        )
        return MemorySaver()

    conn_str: str = _PG_CFG.get("connection_string", "")
    # 支持从环境变量读取
    if conn_str.startswith("${") and conn_str.endswith("}"):
        env_var = conn_str[2:-1].split(":-")[0]
        default_val = conn_str[2:-1].split(":-", 1)[1].rstrip("}") if ":-" in conn_str[2:-1] else ""
        conn_str = os.getenv(env_var, default_val)
    elif not conn_str:
        conn_str = os.getenv("POSTGRES_CHECKPOINTER_URL", "")

    if not conn_str:
        logger.warning(
            "[CheckpointerFactory] PostgreSQL 连接字符串未配置，"
            "自动降级到 MemorySaver。请在 config/agent_config.yaml 中设置\n"
            "  checkpointer.postgres.connection_string"
        )
        return MemorySaver()

    try:
        # 异步 PostgresSaver（适配 async graph.astream/ainvoke）
        checkpointer = AsyncPostgresSaver.from_conn_string(conn_str)
        logger.info(
            f"[CheckpointerFactory] 使用 AsyncPostgresSaver | "
            f"conn={conn_str.split('@')[-1].split('/')[0]}"
        )
        return checkpointer
    except Exception as e:
        logger.warning(
            f"[CheckpointerFactory] PostgreSQL 连接失败 ({e})，"
            f"自动降级到 MemorySaver"
        )
        return MemorySaver()
