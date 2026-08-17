"""PostgreSQL 控制面表结构（Spec V2 5.1–5.5）。

- ensure_schema() 幂等建表（IF NOT EXISTS）。
- table_prefix 用于测试隔离：测试用 "test_rag_" 前缀建一套，用完 drop_schema() 删除。
- 表名拼接唯一入口是 t()，仓储层不得自行拼表名。
"""

import psycopg

BASE_TABLES = (
    "rag_documents",
    "rag_document_versions",
    "rag_ingest_tasks",
    "rag_file_states",
    "rag_kb_state",
)


def t(name: str, table_prefix: str = "") -> str:
    """表名拼接的唯一入口（所有仓储类共用）。"""
    return f"{table_prefix}{name}"


def normalize_conn_str(conn_str: str) -> str:
    """兼容 SQLAlchemy 风格连接串：postgresql+psycopg:// → postgresql://。"""
    return conn_str.replace("postgresql+psycopg://", "postgresql://")


def _ddl_statements(table_prefix: str = "") -> list[str]:
    docs = t("rag_documents", table_prefix)
    versions = t("rag_document_versions", table_prefix)
    tasks = t("rag_ingest_tasks", table_prefix)
    files = t("rag_file_states", table_prefix)
    kb = t("rag_kb_state", table_prefix)

    return [
        # ── 5.1 rag_documents：每个知识库文档一条记录 ──────────────
        f"""
        CREATE TABLE IF NOT EXISTS {docs} (
            document_id       TEXT PRIMARY KEY,
            kb_id             TEXT NOT NULL,
            source            TEXT NOT NULL,
            source_filename   TEXT,
            status            TEXT NOT NULL DEFAULT 'active',  -- active/deleted
            active_version_id TEXT,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (kb_id, source)
        )
        """,
        # ── 5.2 rag_document_versions：每个内容版本一条记录 ────────
        f"""
        CREATE TABLE IF NOT EXISTS {versions} (
            document_id     TEXT NOT NULL REFERENCES {docs}(document_id),
            version_id      TEXT NOT NULL,
            content_hash    TEXT,
            status          TEXT NOT NULL DEFAULT 'staging',
                -- staging → validated → active → retired；staging/validated → failed
            parent_count    INTEGER NOT NULL DEFAULT 0,
            child_count     INTEGER NOT NULL DEFAULT 0,
            failure_stage   TEXT,
            failure_reason  TEXT,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            validated_at    TIMESTAMPTZ,
            activated_at    TIMESTAMPTZ,
            retired_at      TIMESTAMPTZ,
            PRIMARY KEY (document_id, version_id)
        )
        """,
        # 同一 document_id 最多一个 active version（部分唯一索引兜底）
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS {table_prefix}rag_versions_one_active_idx
        ON {versions} (document_id) WHERE status = 'active'
        """,
        f"""
        CREATE INDEX IF NOT EXISTS {table_prefix}rag_versions_status_idx
        ON {versions} (status)
        """,
        # ── 5.3 rag_ingest_tasks：持久化任务队列 ──────────────────
        f"""
        CREATE TABLE IF NOT EXISTS {tasks} (
            task_id          BIGSERIAL PRIMARY KEY,
            task_type        TEXT NOT NULL,
                -- ingest_file / ingest_text / delete_document / rebuild_bm25 / reconcile_version
            dedup_key        TEXT NOT NULL,
            payload          JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            priority         INTEGER NOT NULL DEFAULT 0,
            status           TEXT NOT NULL DEFAULT 'pending',
                -- pending → processing → completed；processing → failed/pending
            current_stage    TEXT,
            retries          INTEGER NOT NULL DEFAULT 0,
            max_retries      INTEGER NOT NULL DEFAULT 3,
            locked_by        TEXT,
            lease_expires_at TIMESTAMPTZ,
            heartbeat_at     TIMESTAMPTZ,
            error_message    TEXT,
            lease_generation BIGINT NOT NULL DEFAULT 0,
                -- WRK-002 租约代次：每次 claim（含过期回收）原子 +1，作为 fencing token
            created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
            completed_at     TIMESTAMPTZ
        )
        """,
        # 相同 dedup_key 的 pending/processing 任务只能存在一个（部分唯一索引）
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS {table_prefix}rag_tasks_dedup_idx
        ON {tasks} (dedup_key) WHERE status IN ('pending', 'processing')
        """,
        f"""
        CREATE INDEX IF NOT EXISTS {table_prefix}rag_tasks_status_idx
        ON {tasks} (status)
        """,
        # ── 5.4 rag_file_states：文件状态 ─────────────────────────
        # 简化方案：source（规范化绝对路径）全局唯一，直接作 PRIMARY KEY，
        # 不再引入 (kb_id, source) 复合键 —— Spec 6.4 要求 document_id 由
        # kb_id + 规范化绝对 source 派生，source 本身已能唯一定位文件。
        f"""
        CREATE TABLE IF NOT EXISTS {files} (
            source            TEXT PRIMARY KEY,
            document_id       TEXT,
            content_hash      TEXT,
            file_mtime        DOUBLE PRECISION,
            file_size         BIGINT,
            status            TEXT NOT NULL DEFAULT 'active',  -- active/deleted
            active_version_id TEXT,
            last_indexed_at   TIMESTAMPTZ,
            updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """,
        # ── 5.5 rag_kb_state：知识库级状态 ────────────────────────
        f"""
        CREATE TABLE IF NOT EXISTS {kb} (
            kb_id           TEXT PRIMARY KEY,
            active_revision BIGINT NOT NULL DEFAULT 0,
            bm25_revision   BIGINT NOT NULL DEFAULT 0,
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """,
        # ── 存量库兼容：既有 rag_ingest_tasks 表补 lease_generation 列 ──
        # CREATE TABLE IF NOT EXISTS 不会给已存在的表加列，用幂等 ALTER 兜底
        f"""
        ALTER TABLE {tasks}
        ADD COLUMN IF NOT EXISTS lease_generation BIGINT NOT NULL DEFAULT 0
        """,
    ]


def ensure_schema(conn_str: str, table_prefix: str = "") -> None:
    """幂等创建控制面 5 张表及索引。"""
    with psycopg.connect(normalize_conn_str(conn_str)) as conn:
        for stmt in _ddl_statements(table_prefix):
            conn.execute(stmt)
        conn.commit()


def drop_schema(conn_str: str, table_prefix: str = "") -> None:
    """删除指定前缀的全部控制面表（测试清理用）。"""
    with psycopg.connect(normalize_conn_str(conn_str)) as conn:
        for name in BASE_TABLES:
            conn.execute(f"DROP TABLE IF EXISTS {t(name, table_prefix)} CASCADE")
        conn.commit()
