"""
中期记忆 - SQLite 会话存储

参考 Hermes-Agent 设计：
- SQLite + WAL 模式，支持多读者单写者
- sessions 表存储会话元数据（按 user_id 隔离）
- messages 表存储完整消息历史（按 user_id 隔离）
- 应用级重试处理写冲突

配置来源：config/memory_config.yaml -> session_store
"""

import json
import random
import sqlite3
import time
import threading
from pathlib import Path

from memory.models import MessageRecord, SessionInfo
from utils.config_handler import memory_config
from utils.path_tools import get_abs_path
from utils.logger import logger

# ---- 配置读取 ----
_SESSION_CFG: dict = memory_config.get("session_store", {})

DEFAULT_DB_PATH: str = _SESSION_CFG.get("db_path", "logs/memory_sessions.db")
DEFAULT_WAL_MODE: bool = _SESSION_CFG.get("wal_mode", True)
DEFAULT_MAX_RETRIES: int = _SESSION_CFG.get("max_retries", 5)

# 写冲突重试：基础延迟范围（毫秒）
RETRY_DELAY_MIN_MS: int = 20
RETRY_DELAY_MAX_MS: int = 100


class SessionStore:
    """SQLite 会话存储引擎

    设计原则：
    1. WAL 模式提高并发性能
    2. 应用级重试处理写冲突
    3. 自动列迁移（声明式）
    4. 按 user_id 隔离会话数据
    """

    def __init__(self, db_path: str | None = None):
        """初始化会话存储

        Args:
            db_path: SQLite 数据库路径，None 时读取配置
        """
        self.db_path: str = db_path or get_abs_path(DEFAULT_DB_PATH)
        self._lock: threading.Lock = threading.Lock()
        self._local: threading.local = threading.local()

        # 确保目录存在
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """获取线程本地连接"""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        """初始化数据库 Schema（含向后兼容的列迁移）"""
        with self._lock:
            conn: sqlite3.Connection = sqlite3.connect(self.db_path)

            if DEFAULT_WAL_MODE:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")

            # 会话表
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT DEFAULT 'default',
                    title TEXT,
                    created_at REAL,
                    updated_at REAL,
                    message_count INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    source TEXT DEFAULT 'api',
                    is_active INTEGER DEFAULT 1,
                    end_reason TEXT,
                    parent_session_id TEXT,
                    metadata TEXT
                )
                """
            )

            # 消息表
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    user_id TEXT DEFAULT 'default',
                    role TEXT,
                    content TEXT,
                    tool_call_id TEXT,
                    tool_calls TEXT,
                    name TEXT,
                    timestamp REAL,
                    token_count INTEGER,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
                """
            )

            # 向后兼容：如果旧表没有 user_id 列，自动添加
            self._ensure_column(conn, "sessions", "user_id", "TEXT DEFAULT 'default'")
            self._ensure_column(conn, "messages", "user_id", "TEXT DEFAULT 'default'")

            # 索引
            conn.execute("CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_msg_user ON messages(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_active ON sessions(is_active)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at)")

            conn.commit()
            conn.close()
            logger.info(f"[SessionStore] 数据库初始化完成: {self.db_path}")

    def _ensure_column(
        self, conn: sqlite3.Connection, table: str, column: str, col_type: str
    ) -> None:
        """确保表中有指定列（向后兼容迁移）"""
        cursor = conn.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in cursor.fetchall()]
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
            logger.info(f"[SessionStore] 迁移: {table} 表添加 {column} 列")

    def _retry_execute(self, func, *args, **kwargs):
        """带重试的数据库操作

        参考 Hermes-Agent 设计：应用级随机抖动重试，避免 SQLite 内部退避造成的车队效应。
        """
        for attempt in range(DEFAULT_MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < DEFAULT_MAX_RETRIES - 1:
                    sleep_ms: int = random.randint(
                        RETRY_DELAY_MIN_MS, RETRY_DELAY_MAX_MS
                    ) * (attempt + 1)
                    time.sleep(sleep_ms / 1000)
                    continue
                raise

    # ==================== 会话操作 ====================

    def create_session(
        self,
        session_id: str,
        user_id: str = "default",
        title: str | None = None,
        source: str = "api",
        metadata: dict | None = None,
    ) -> SessionInfo:
        """创建新会话"""
        now: float = time.time()
        info: SessionInfo = SessionInfo(
            session_id=session_id,
            user_id=user_id,
            title=title or f"会话 {session_id[:8]}",
            created_at=now,
            updated_at=now,
            source=source,
            metadata=metadata or {},
        )

        def _insert() -> None:
            conn: sqlite3.Connection = self._get_conn()
            conn.execute(
                """
                INSERT INTO sessions
                (session_id, user_id, title, created_at, updated_at, source, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    info.session_id,
                    info.user_id,
                    info.title,
                    info.created_at,
                    info.updated_at,
                    info.source,
                    json.dumps(info.metadata),
                ),
            )
            conn.commit()

        self._retry_execute(_insert)
        logger.info(f"[SessionStore] 创建会话: {session_id} | user={user_id}")
        return info

    def get_session(self, session_id: str, user_id: str | None = None) -> SessionInfo | None:
        """获取会话信息

        Args:
            session_id: 会话ID
            user_id: 用户ID（提供时校验归属，None 时仅按 session_id 查询）
        """
        conn: sqlite3.Connection = self._get_conn()
        if user_id is not None:
            row: sqlite3.Row | None = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ? AND user_id = ?",
                (session_id, user_id),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()

        if not row:
            return None

        return SessionInfo(
            session_id=row["session_id"],
            user_id=row["user_id"] or "default",
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            message_count=row["message_count"],
            total_tokens=row["total_tokens"],
            source=row["source"],
            is_active=bool(row["is_active"]),
            end_reason=row["end_reason"],
            parent_session_id=row["parent_session_id"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    def list_sessions(
        self,
        user_id: str = "default",
        limit: int = 50,
        offset: int = 0,
        active_only: bool = False,
    ) -> list[SessionInfo]:
        """列出指定用户的会话列表"""
        conn: sqlite3.Connection = self._get_conn()

        sql: str = "SELECT * FROM sessions WHERE user_id = ?"
        params: list = [user_id]
        if active_only:
            sql += " AND is_active = 1"
        sql += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows: list[sqlite3.Row] = conn.execute(sql, params).fetchall()

        results: list[SessionInfo] = []
        for row in rows:
            results.append(
                SessionInfo(
                    session_id=row["session_id"],
                    user_id=row["user_id"] or "default",
                    title=row["title"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    message_count=row["message_count"],
                    total_tokens=row["total_tokens"],
                    source=row["source"],
                    is_active=bool(row["is_active"]),
                    end_reason=row["end_reason"],
                    parent_session_id=row["parent_session_id"],
                    metadata=json.loads(row["metadata"] or "{}"),
                )
            )
        return results

    def update_session(self, session_id: str, user_id: str | None = None, **kwargs) -> bool:
        """更新会话信息

        Args:
            session_id: 会话ID
            user_id: 用户ID（提供时增加归属校验）
        """
        allowed: set[str] = {
            "title",
            "is_active",
            "end_reason",
            "metadata",
            "total_tokens",
            "message_count",
        }
        updates: dict = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return False

        updates["updated_at"] = time.time()

        if "metadata" in updates:
            updates["metadata"] = json.dumps(updates["metadata"])

        fields: str = ", ".join(f"{k} = ?" for k in updates.keys())
        values: list = list(updates.values())
        values.append(session_id)

        def _update() -> None:
            conn: sqlite3.Connection = self._get_conn()
            if user_id is not None:
                conn.execute(
                    f"UPDATE sessions SET {fields} WHERE session_id = ? AND user_id = ?",
                    values + [user_id],
                )
            else:
                conn.execute(f"UPDATE sessions SET {fields} WHERE session_id = ?", values)
            conn.commit()

        self._retry_execute(_update)
        return True

    def end_session(self, session_id: str, user_id: str | None = None, reason: str = "user_close") -> None:
        """结束指定会话"""
        self.update_session(session_id, user_id=user_id, is_active=False, end_reason=reason)
        logger.info(f"[SessionStore] 结束会话: {session_id}, 原因: {reason}")

    def delete_session(self, session_id: str, user_id: str | None = None) -> bool:
        """删除会话及所有消息（带用户归属校验）"""
        deleted = False

        def _delete() -> None:
            nonlocal deleted
            conn: sqlite3.Connection = self._get_conn()
            if user_id is not None:
                # 先校验归属
                row = conn.execute(
                    "SELECT 1 FROM sessions WHERE session_id = ? AND user_id = ?",
                    (session_id, user_id),
                ).fetchone()
                if not row:
                    deleted = False
                    return  # 不属于该用户，静默跳过
                conn.execute(
                    "DELETE FROM messages WHERE session_id = ? AND user_id = ?",
                    (session_id, user_id),
                )
                conn.execute(
                    "DELETE FROM sessions WHERE session_id = ? AND user_id = ?",
                    (session_id, user_id),
                )
            else:
                conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
            deleted = True

        self._retry_execute(_delete)
        logger.info(f"[SessionStore] 删除会话: {session_id} | success={deleted}")
        return deleted

    # ==================== 消息操作 ====================

    def append_message(self, session_id: str, msg: MessageRecord, user_id: str = "default") -> bool:
        """追加消息到会话"""

        def _insert() -> None:
            conn: sqlite3.Connection = self._get_conn()
            conn.execute(
                """
                INSERT INTO messages
                (session_id, user_id, role, content, tool_call_id, tool_calls, name, timestamp, token_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    user_id,
                    msg.role,
                    msg.content,
                    msg.tool_call_id,
                    json.dumps(msg.tool_calls) if msg.tool_calls else None,
                    msg.name,
                    msg.timestamp,
                    msg.token_count,
                ),
            )
            conn.commit()

        self._retry_execute(_insert)

        # 更新会话统计
        self._update_session_stats(session_id)
        return True

    def _update_session_stats(self, session_id: str) -> None:
        """更新会话消息统计"""

        def _update() -> None:
            conn: sqlite3.Connection = self._get_conn()
            row: sqlite3.Row = conn.execute(
                """
                SELECT COUNT(*) as cnt, COALESCE(SUM(token_count), 0) as tokens
                FROM messages WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

            conn.execute(
                """
                UPDATE sessions SET message_count = ?, total_tokens = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (row["cnt"], row["tokens"], time.time(), session_id),
            )
            conn.commit()

        self._retry_execute(_update)

    def get_messages(
        self,
        session_id: str,
        user_id: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[MessageRecord]:
        """获取会话消息列表

        Args:
            session_id: 会话ID
            user_id: 用户ID（提供时校验归属）
        """
        conn: sqlite3.Connection = self._get_conn()
        if user_id is not None:
            rows: list[sqlite3.Row] = conn.execute(
                """
                SELECT m.* FROM messages m
                JOIN sessions s ON m.session_id = s.session_id
                WHERE m.session_id = ? AND s.user_id = ?
                ORDER BY m.timestamp ASC, m.id ASC LIMIT ? OFFSET ?
                """,
                (session_id, user_id, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM messages WHERE session_id = ?
                ORDER BY timestamp ASC, id ASC LIMIT ? OFFSET ?
                """,
                (session_id, limit, offset),
            ).fetchall()

        results: list[MessageRecord] = []
        for row in rows:
            results.append(
                MessageRecord(
                    role=row["role"],
                    content=row["content"] or "",
                    tool_call_id=row["tool_call_id"],
                    tool_calls=json.loads(row["tool_calls"]) if row["tool_calls"] else None,
                    name=row["name"],
                    timestamp=row["timestamp"],
                    token_count=row["token_count"],
                )
            )
        return results

    def search_messages(
        self,
        query: str,
        user_id: str = "default",
        session_id: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """简单关键词搜索消息（LIKE 匹配，生产环境可升级为 FTS5）"""
        conn: sqlite3.Connection = self._get_conn()

        sql: str = "SELECT * FROM messages WHERE user_id = ? AND content LIKE ?"
        params: list = [user_id, f"%{query}%"]

        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)

        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows: list[sqlite3.Row] = conn.execute(sql, params).fetchall()

        results: list[dict] = []
        for row in rows:
            results.append(
                {
                    "session_id": row["session_id"],
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": row["timestamp"],
                }
            )
        return results
