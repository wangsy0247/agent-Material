"""PGVector 向量索引管理工具。

在 PGVector collection 的 embedding 列上创建 HNSW / IVFFlat 索引，
加速向量相似度检索（替代全表顺序扫描）。

使用方式：
    from agent_material.rag_engine.vector_index import create_vector_index
    create_vector_index(conn_str, index_config)

所有 collection 共享 langchain_pg_embedding 表，
索引只创建一次（IF NOT EXISTS 幂等安全）。
"""

from __future__ import annotations

import re

import psycopg

from agent_material.utils.logger import get_logger

logger = get_logger("vector_index")

# ── 常量 ──────────────────────────────────────────────────────────
_TABLE_NAME = "langchain_pg_embedding"
_EMBEDDING_COLUMN = "embedding"
HNSW_INDEX_NAME = "idx_langchain_pg_embedding_embedding_hnsw"
IVFFLAT_INDEX_NAME = "idx_langchain_pg_embedding_embedding_ivfflat"

# 索引名称白名单（仅允许字母、数字、下划线）
_VALID_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_identifier(name: str) -> None:
    """校验 PostgreSQL 标识符安全性，防止 SQL 注入。"""
    if not _VALID_NAME_RE.match(name):
        raise ValueError(f"非法 PostgreSQL 标识符: {name!r}")


def create_vector_index(
    conn_str: str,
    index_config: dict | None = None,
) -> bool:
    """在 langchain_pg_embedding.embedding 列上创建向量索引（幂等安全）。

    自动处理 embedding 列无维度声明的情况（ALTER COLUMN TYPE）。

    Args:
        conn_str: PostgreSQL 连接字符串（psycopg 格式，不含 +psycopg 前缀）
        index_config: 索引配置字典，取自 vector_store_config.vector_index
            格式: {
                "enabled": True,
                "method": "hnsw",
                "hnsw": {"m": 16, "ef_construction": 64},
            }

    Returns:
        True 表示索引创建成功或已存在，False 表示被配置禁用或创建失败
    """
    cfg = index_config or {}

    if not cfg.get("enabled", True):
        logger.info("[VectorIndex] 向量索引已禁用（enabled=false），使用精确暴力搜索")
        return False

    # 确保 embedding 列有显式维度（HNSW/IVFFlat 索引要求）
    if not _ensure_column_dimensions(conn_str):
        return False

    method = cfg.get("method", "hnsw").lower()

    if method == "hnsw":
        return _create_hnsw_index(conn_str, cfg.get("hnsw", {}))
    elif method == "ivfflat":
        return _create_ivfflat_index(conn_str, cfg.get("ivfflat", {}))
    else:
        logger.warning(f"[VectorIndex] 未知的索引方法: {method}，跳过")
        return False


def _ensure_column_dimensions(conn_str: str) -> bool:
    """确保 embedding 列声明了显式维度（pgvector 索引要求）。

    当 langchain-postgres 未传入 embedding_length 时，列类型为 vector 而非 vector(N)，
    无法直接建 HNSW/IVFFlat 索引。此函数自动从已有数据推断维度并 ALTER COLUMN。

    Returns:
        True 表示列已有维度或修复成功，False 表示失败
    """
    conn = psycopg.connect(conn_str, autocommit=True)
    try:
        # 检查是否已有维度声明（atttypmod != -1 表示有限定长度）
        row = conn.execute(
            """
            SELECT atttypmod FROM pg_attribute
            WHERE attrelid = %s::regclass AND attname = %s
            """,
            (_TABLE_NAME, _EMBEDDING_COLUMN),
        ).fetchone()

        if row is None:
            logger.warning(f"[VectorIndex] 列 {_TABLE_NAME}.{_EMBEDDING_COLUMN} 不存在")
            return False

        atttypmod = row[0]
        if atttypmod != -1:
            # 已有显式维度（如 vector(1024)），无需修改
            return True

        # 从已有数据推断向量维度
        dim_row = conn.execute(
            f"SELECT vector_dims({_EMBEDDING_COLUMN}) FROM {_TABLE_NAME} LIMIT 1"
        ).fetchone()

        if dim_row is None:
            logger.warning(
                "[VectorIndex] embedding 列无数据且无维度声明，"
                "跳过索引创建（入库后自动创建）"
            )
            return False

        dims = dim_row[0]
        logger.info(
            f"[VectorIndex] embedding 列无显式维度，"
            f"根据已有数据推断为 vector({dims})，执行 ALTER COLUMN..."
        )

        conn.execute(
            f"ALTER TABLE {_TABLE_NAME} "
            f"ALTER COLUMN {_EMBEDDING_COLUMN} TYPE vector({int(dims)})"
        )
        logger.info(f"[VectorIndex] embedding 列已更新为 vector({dims})")
        return True

    except Exception as e:
        logger.warning(f"[VectorIndex] 确保列维度失败: {e}")
        return False
    finally:
        conn.close()


def _create_hnsw_index(conn_str: str, hnsw_cfg: dict) -> bool:
    """创建 HNSW 索引。"""
    index_name = HNSW_INDEX_NAME
    m = hnsw_cfg.get("m", 16)
    ef_construction = hnsw_cfg.get("ef_construction", 64)

    _validate_identifier(index_name)

    sql = (
        f"CREATE INDEX IF NOT EXISTS {index_name} "
        f"ON {_TABLE_NAME} "
        f"USING hnsw ({_EMBEDDING_COLUMN} vector_cosine_ops) "
        f"WITH (m = {int(m)}, ef_construction = {int(ef_construction)})"
    )

    try:
        _execute_ddl(conn_str, sql)
        logger.info(
            f"[VectorIndex] HNSW 索引就绪 | "
            f"index={index_name} | m={m} | ef_construction={ef_construction}"
        )
        return True
    except Exception as e:
        logger.warning(
            f"[VectorIndex] HNSW 索引创建失败，检索将回退到顺序扫描: {e}"
        )
        return False


def _create_ivfflat_index(conn_str: str, ivfflat_cfg: dict) -> bool:
    """创建 IVFFlat 索引。"""
    index_name = IVFFLAT_INDEX_NAME
    lists = ivfflat_cfg.get("lists", 100)

    _validate_identifier(index_name)

    sql = (
        f"CREATE INDEX IF NOT EXISTS {index_name} "
        f"ON {_TABLE_NAME} "
        f"USING ivfflat ({_EMBEDDING_COLUMN} vector_cosine_ops) "
        f"WITH (lists = {int(lists)})"
    )

    try:
        _execute_ddl(conn_str, sql)
        logger.info(
            f"[VectorIndex] IVFFlat 索引就绪 | "
            f"index={index_name} | lists={lists}"
        )
        return True
    except Exception as e:
        logger.warning(
            f"[VectorIndex] IVFFlat 索引创建失败，检索将回退到顺序扫描: {e}"
        )
        return False


def _execute_ddl(conn_str: str, sql: str) -> None:
    """执行 DDL 语句（建索引）。"""
    conn = psycopg.connect(conn_str, autocommit=True)
    try:
        conn.execute(sql)
    finally:
        conn.close()


def build_psycopg_conn_str(conn_str: str) -> str:
    """将 SQLAlchemy 连接字符串转换为 psycopg 原生格式。

    "postgresql+psycopg://..." → "postgresql://..."
    """
    return conn_str.replace("postgresql+psycopg://", "postgresql://")
