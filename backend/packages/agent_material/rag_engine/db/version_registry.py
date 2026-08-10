"""文档版本注册表（OFF-002）。

管理 document_versions 表的状态机：
staging → validated → active → retired，失败进入 failed。

特性：
- 线程安全（threading.Lock + WAL 模式）
- 激活是单事务操作：旧 active 版本退役 + 新版本激活原子完成
- 同一 document_id 最多一个 active（由 schema 部分唯一索引兜底）
"""

import threading

# 允许被激活的状态
_ACTIVATABLE_STATES = ("staging", "validated", "failed")

# 查询返回的字段顺序（dict key 与列名一致）
_COLUMNS = (
    "document_id", "version_id", "source", "content_hash", "status",
    "parent_count", "child_count", "failure_stage", "failure_reason",
    "created_at", "activated_at", "retired_at",
)


class DocumentVersionRegistry:
    """文档版本注册表 — 线程安全。"""

    def __init__(self, db_path: str = "rag_queue.db"):
        from agent_material.utils.path_tools import get_abs_path

        self._db_path = get_abs_path(db_path)
        self._lock = threading.Lock()

        from agent_material.rag_engine.db.schema import init_db

        self._conn = init_db(self._db_path)

    # ── 状态写入 ──────────────────────────────────────────

    def register_staging(self, document_id: str, version_id: str, source: str, content_hash: str):
        """注册 staging 版本（幂等）。

        已存在的同 (document_id, version_id) 记录保持不变，
        避免重复处理把已激活/已校验的版本降级回 staging。
        """
        with self._lock:
            self._conn.execute(
                """INSERT OR IGNORE INTO document_versions
                   (document_id, version_id, source, content_hash, status)
                   VALUES (?, ?, ?, ?, 'staging')""",
                (document_id, version_id, source, content_hash),
            )
            self._conn.commit()

    def mark_validated(self, document_id: str, version_id: str, parent_count: int, child_count: int):
        """staging → validated，并记录父/子块数量。"""
        with self._lock:
            self._conn.execute(
                """UPDATE document_versions
                   SET status='validated', parent_count=?, child_count=?
                   WHERE document_id=? AND version_id=?""",
                (parent_count, child_count, document_id, version_id),
            )
            self._conn.commit()

    def mark_failed(self, document_id: str, version_id: str, stage: str, reason: str):
        """状态置 failed，记录失败阶段与原因。"""
        with self._lock:
            self._conn.execute(
                """UPDATE document_versions
                   SET status='failed', failure_stage=?, failure_reason=?
                   WHERE document_id=? AND version_id=?""",
                (stage, reason, document_id, version_id),
            )
            self._conn.commit()

    def activate(self, document_id: str, version_id: str) -> str | None:
        """激活指定版本（单事务）。

        同一事务内：当前 active 版本置 retired（写 retired_at），
        目标版本置 active（写 activated_at，清 failure 字段）。
        返回被退役的旧 version_id（无则 None）。

        目标版本必须是 staging/validated/failed 之一，否则抛 ValueError。
        """
        with self._lock:
            target = self._get_row(document_id, version_id)
            if target is None:
                raise ValueError(f"版本不存在: {document_id}/{version_id}")
            if target["status"] not in _ACTIVATABLE_STATES:
                raise ValueError(
                    f"版本状态不可激活: {document_id}/{version_id} status={target['status']}"
                )

            old = self._get_active_row(document_id)
            try:
                if old is not None:
                    self._conn.execute(
                        """UPDATE document_versions
                           SET status='retired', retired_at=datetime('now')
                           WHERE document_id=? AND version_id=?""",
                        (document_id, old["version_id"]),
                    )
                self._conn.execute(
                    """UPDATE document_versions
                       SET status='active', activated_at=datetime('now'),
                           retired_at=NULL, failure_stage=NULL, failure_reason=NULL
                       WHERE document_id=? AND version_id=?""",
                    (document_id, version_id),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            return old["version_id"] if old else None

    # ── 查询 ──────────────────────────────────────────────

    def get_active(self, document_id: str) -> dict | None:
        """获取 document_id 当前 active 版本。"""
        return self._get_active_row(document_id)

    def get(self, document_id: str, version_id: str) -> dict | None:
        """按主键获取单个版本记录。"""
        return self._get_row(document_id, version_id)

    def list_versions(self, document_id: str) -> list[dict]:
        """列出 document_id 的全部版本（按创建时间升序）。"""
        rows = self._conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM document_versions "
            "WHERE document_id=? ORDER BY created_at ASC",
            (document_id,),
        ).fetchall()
        return [dict(zip(_COLUMNS, r)) for r in rows]

    def get_retired(self, older_than_days: int) -> list[dict]:
        """获取退役超过指定天数的版本（供延迟清理）。"""
        rows = self._conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM document_versions "
            "WHERE status='retired' AND retired_at < datetime('now', ?)",
            (f"-{older_than_days} days",),
        ).fetchall()
        return [dict(zip(_COLUMNS, r)) for r in rows]

    # ── 内部 ──────────────────────────────────────────────

    def _get_row(self, document_id: str, version_id: str) -> dict | None:
        row = self._conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM document_versions "
            "WHERE document_id=? AND version_id=?",
            (document_id, version_id),
        ).fetchone()
        return dict(zip(_COLUMNS, row)) if row else None

    def _get_active_row(self, document_id: str) -> dict | None:
        row = self._conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM document_versions "
            "WHERE document_id=? AND status='active'",
            (document_id,),
        ).fetchone()
        return dict(zip(_COLUMNS, row)) if row else None
