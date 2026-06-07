"""SQLite 数据库建表 DDL。"""

import sqlite3

SCHEMA_SQL = """
-- 任务队列
CREATE TABLE IF NOT EXISTS tasks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    task_type    TEXT    NOT NULL,       -- ingest_file | delete_file | reindex_all
    payload      TEXT    NOT NULL,       -- JSON: {"file_path": "...", "source": "..."}
    status       TEXT    NOT NULL DEFAULT 'pending',  -- pending | processing | completed | failed
    priority     INTEGER DEFAULT 0,
    retries      INTEGER DEFAULT 0,
    max_retries  INTEGER DEFAULT 3,
    error_msg    TEXT,
    created_at   TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at   TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- 文件状态追踪（替代 md5.text）
CREATE TABLE IF NOT EXISTS file_states (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path       TEXT    UNIQUE NOT NULL,
    md5_hash        TEXT    NOT NULL,
    file_mtime      REAL    NOT NULL,
    file_size       INTEGER NOT NULL,
    last_indexed_at TEXT,
    status          TEXT    DEFAULT 'active',
    parent_ids      TEXT              -- JSON 数组，记录在大 collection 中的 parent_id
);

CREATE INDEX IF NOT EXISTS idx_tasks_status  ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at);
CREATE INDEX IF NOT EXISTS idx_file_path      ON file_states(file_path);
"""


def init_db(db_path: str) -> sqlite3.Connection:
    """初始化 SQLite 数据库，创建所有表。"""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn
