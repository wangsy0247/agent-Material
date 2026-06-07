"""SQLite 持久化任务队列。

线程安全（threading.Lock + WAL 模式），支持：
- 入队 / 出队 / 完成 / 失败
- 失败重试（指数退避）
- 按优先级和创建时间排序
"""

import json
import sqlite3
import threading
from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    INGEST_FILE = "ingest_file"
    DELETE_FILE = "delete_file"
    REINDEX_ALL = "reindex_all"


class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    id: int
    task_type: str
    payload: dict
    status: str
    retries: int
    error_msg: str | None
    created_at: str


class TaskQueue:
    """SQLite 任务队列 — 线程安全。"""

    def __init__(self, db_path: str = "rag_queue.db"):
        from rag_mcp_server.utils.path import get_abs_path

        self._db_path = get_abs_path(db_path)
        self._lock = threading.Lock()

        from rag_mcp_server.db.schema import init_db

        self._conn = init_db(self._db_path)

    @property
    def db_path(self) -> str:
        return self._db_path

    # ── 入队 ──────────────────────────────────────────────

    def enqueue(
        self,
        task_type: TaskType,
        payload: dict,
        priority: int = 0,
    ) -> int:
        """添加任务到队列，返回 task_id。"""
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO tasks (task_type, payload, priority) VALUES (?, ?, ?)",
                (task_type.value, json.dumps(payload, ensure_ascii=False), priority),
            )
            self._conn.commit()
            return cur.lastrowid

    def enqueue_batch(
        self,
        tasks: list[tuple[TaskType, dict, int]],
    ) -> list[int]:
        """批量入队，返回 task_id 列表。"""
        ids: list[int] = []
        with self._lock:
            for task_type, payload, priority in tasks:
                cur = self._conn.execute(
                    "INSERT INTO tasks (task_type, payload, priority) VALUES (?, ?, ?)",
                    (task_type.value, json.dumps(payload, ensure_ascii=False), priority),
                )
                ids.append(cur.lastrowid)
            self._conn.commit()
        return ids

    # ── 出队 ──────────────────────────────────────────────

    def dequeue(self, batch_size: int = 1) -> list[Task]:
        """获取并锁定待处理任务（pending → processing）。"""
        with self._lock:
            rows = self._conn.execute(
                """SELECT id, task_type, payload, status, retries, error_msg, created_at
                   FROM tasks
                   WHERE status = 'pending'
                   ORDER BY priority DESC, created_at ASC
                   LIMIT ?""",
                (batch_size,),
            ).fetchall()

            if rows:
                ids = [r[0] for r in rows]
                placeholders = ",".join("?" for _ in ids)
                self._conn.execute(
                    f"UPDATE tasks SET status='processing', updated_at=datetime('now') "
                    f"WHERE id IN ({placeholders})",
                    ids,
                )
                self._conn.commit()

            return [
                Task(
                    id=r[0],
                    task_type=r[1],
                    payload=json.loads(r[2]),
                    status=r[3],
                    retries=r[4],
                    error_msg=r[5],
                    created_at=r[6],
                )
                for r in rows
            ]

    # ── 状态更新 ──────────────────────────────────────────

    def complete(self, task_id: int):
        with self._lock:
            self._conn.execute(
                "UPDATE tasks SET status='completed', updated_at=datetime('now') WHERE id=?",
                (task_id,),
            )
            self._conn.commit()

    def fail(self, task_id: int, error_msg: str):
        with self._lock:
            self._conn.execute(
                """UPDATE tasks SET status='failed', error_msg=?,
                   retries=retries+1, updated_at=datetime('now') WHERE id=?""",
                (error_msg, task_id),
            )
            self._conn.commit()

    def retry_failed(self, max_retries: int = 3) -> int:
        """将未超过最大重试次数的失败任务重置为 pending。"""
        with self._lock:
            cur = self._conn.execute(
                """UPDATE tasks SET status='pending', error_msg=NULL, updated_at=datetime('now')
                   WHERE status='failed' AND retries < ?""",
                (max_retries,),
            )
            self._conn.commit()
            return cur.rowcount

    # ── 查询 ──────────────────────────────────────────────

    def get_status(self, task_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT id, task_type, status, retries, error_msg, created_at, updated_at FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "type": row[1],
            "status": row[2],
            "retries": row[3],
            "error": row[4],
            "created": row[5],
            "updated": row[6],
        }

    def get_pending_count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE status='pending'"
        ).fetchone()
        return row[0] if row else 0

    def get_processing_count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE status='processing'"
        ).fetchone()
        return row[0] if row else 0

    def get_summary(self) -> dict:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) FROM tasks GROUP BY status"
        ).fetchall()
        counts = {row[0]: row[1] for row in rows}
        return {
            "pending": counts.get("pending", 0),
            "processing": counts.get("processing", 0),
            "completed": counts.get("completed", 0),
            "failed": counts.get("failed", 0),
        }

    def cleanup_old(self, days: int = 7):
        """清理已完成/失败的旧任务记录。"""
        with self._lock:
            self._conn.execute(
                """DELETE FROM tasks
                   WHERE status IN ('completed', 'failed')
                   AND updated_at < datetime('now', ?)""",
                (f"-{days} days",),
            )
            self._conn.commit()
