"""存量数据 backfill：为历史 chunk 补 is_active=true 标记（OFF-002 / 7.2）。

用途：
    版本化检索引入 active_version_filter 后，在线检索只读取
    is_active=true（或字段缺失）的数据。本脚本对 agent_small / agent_large /
    agent 三个 collection 中尚未携带 is_active 字段的存量记录显式补上
    is_active=true，使开关生效后旧数据仍然可见。

特性：
    - 幂等：只更新不包含 is_active 键的行，重复执行无副作用。
    - 只读配置：连接串取自 config/vector_store_config.yaml。
    - 不自动运行：手动执行 `python scripts/backfill_active_metadata.py`。
"""

import sys
from pathlib import Path

import psycopg

# 允许直接以脚本方式运行（python scripts/backfill_active_metadata.py）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend" / "packages"))

from agent_material.utils.config_handler import vector_store_config

# 待 backfill 的 collection（多粒度父子 + 基础单粒度），名称随配置推导
_base = vector_store_config.get("character_name", "agent")
_mg = vector_store_config.get("multi_granularity", {})
COLLECTIONS = (
    _base + _mg.get("small_collection_suffix", "_small"),
    _base + _mg.get("large_collection_suffix", "_large"),
    _base,
)


def backfill() -> dict[str, int]:
    """执行 backfill，返回各 collection 更新行数。"""
    conn_str = vector_store_config.get("connection_string", "")
    if not conn_str:
        raise RuntimeError("未配置 vector_store_config.connection_string")
    psycopg_conn_str = conn_str.replace("postgresql+psycopg://", "postgresql://")

    updated: dict[str, int] = {}
    conn = psycopg.connect(psycopg_conn_str)
    cur = conn.cursor()
    try:
        for name in COLLECTIONS:
            cur.execute(
                """
                UPDATE langchain_pg_embedding e
                SET cmetadata = e.cmetadata || '{"is_active": true}'::jsonb
                FROM langchain_pg_collection c
                WHERE e.collection_id = c.uuid
                  AND c.name = %s
                  AND NOT (e.cmetadata ? 'is_active')
                """,
                (name,),
            )
            updated[name] = cur.rowcount
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()
    return updated


if __name__ == "__main__":
    result = backfill()
    for name, count in result.items():
        print(f"collection={name}: 更新 {count} 行（补 is_active=true）")
