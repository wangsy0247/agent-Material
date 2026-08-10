"""PostgreSQL 控制面仓储层（Spec V2 第 5–8 章）。

设计要点：
- 每个方法内部自管理短连接（psycopg.connect），控制面 QPS 很低，无需连接池。
- 单语句方法用 autocommit；多语句方法用显式事务（with conn.transaction()），
  异常时整体回滚。
- 写路径严格受限：文件状态只能在激活/删除事务成功后更新（Spec 5.4），
  本模块不提供"扫描时写状态"的方法。
"""

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from agent_material.rag_engine.db.pg.schema import normalize_conn_str, t


class TaskRepository:
    """rag_ingest_tasks 仓储 — 持久化任务队列（Spec 5.3 / 6.2）。"""

    def __init__(self, conn_str: str, table_prefix: str = ""):
        self._conn_str = normalize_conn_str(conn_str)
        self._table = t("rag_ingest_tasks", table_prefix)

    def _connect(self):
        return psycopg.connect(self._conn_str, row_factory=dict_row)

    # ── 入队 ──────────────────────────────────────────────

    def enqueue(
        self,
        task_type: str,
        payload: dict,
        dedup_key: str,
        max_retries: int = 3,
        priority: int = 0,
    ) -> int | None:
        """入队，返回 task_id。

        相同 dedup_key 已存在 pending/processing 任务时返回 None（不重复入队），
        依赖 rag_tasks_dedup_idx 部分唯一索引保证并发安全。
        """
        with self._connect() as conn:
            row = conn.execute(
                f"""
                INSERT INTO {self._table}
                    (task_type, payload, dedup_key, max_retries, priority)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (dedup_key) WHERE status IN ('pending', 'processing')
                DO NOTHING
                RETURNING task_id
                """,
                (task_type, Jsonb(payload), dedup_key, max_retries, priority),
            ).fetchone()
            conn.commit()
            return row["task_id"] if row else None

    # ── 领取 / 心跳 ───────────────────────────────────────

    def claim(self, worker_id: str, lease_seconds: int = 300) -> dict | None:
        """原子领取一个任务（单条 UPDATE ... FOR UPDATE SKIP LOCKED）。

        可领取：pending，或租约已过期的 processing（Worker 崩溃恢复，
        不增加 retries —— 重试计数只在显式 fail() 时增加）。
        返回任务 dict；无可领取任务返回 None。
        """
        with self._connect() as conn:
            row = conn.execute(
                f"""
                UPDATE {self._table}
                SET status = 'processing',
                    locked_by = %s,
                    lease_expires_at = now() + make_interval(secs => %s),
                    heartbeat_at = now(),
                    updated_at = now()
                WHERE task_id = (
                    SELECT task_id FROM {self._table}
                    WHERE status = 'pending'
                       OR (status = 'processing' AND lease_expires_at < now())
                    ORDER BY priority DESC, created_at ASC
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING *
                """,
                (worker_id, lease_seconds),
            ).fetchone()
            conn.commit()
            return row

    def heartbeat(self, task_id: int, worker_id: str, lease_seconds: int = 300) -> bool:
        """刷新租约和心跳（仅当 locked_by 匹配且任务仍在 processing）。"""
        with self._connect() as conn:
            cur = conn.execute(
                f"""
                UPDATE {self._table}
                SET heartbeat_at = now(),
                    lease_expires_at = now() + make_interval(secs => %s),
                    updated_at = now()
                WHERE task_id = %s AND locked_by = %s AND status = 'processing'
                """,
                (lease_seconds, task_id, worker_id),
            )
            conn.commit()
            return cur.rowcount > 0

    # ── 完成 / 失败 ───────────────────────────────────────

    def complete(self, task_id: int, worker_id: str) -> bool:
        """标记完成（仅当 locked_by 匹配）。"""
        with self._connect() as conn:
            cur = conn.execute(
                f"""
                UPDATE {self._table}
                SET status = 'completed',
                    completed_at = now(),
                    updated_at = now(),
                    locked_by = NULL,
                    lease_expires_at = NULL
                WHERE task_id = %s AND locked_by = %s AND status = 'processing'
                """,
                (task_id, worker_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def fail(
        self,
        task_id: int,
        worker_id: str,
        error: str,
        stage: str | None = None,
    ) -> str | None:
        """记录失败：retries+1，记录 current_stage / error_message。

        retries 达到 max_retries → failed（保留锁信息便于排障）；
        否则回到 pending 并释放锁，等待重新领取。
        返回新状态；任务不属于该 worker 时返回 None。
        """
        with self._connect() as conn:
            row = conn.execute(
                f"""
                UPDATE {self._table}
                SET retries = retries + 1,
                    status = CASE WHEN retries + 1 >= max_retries
                                  THEN 'failed' ELSE 'pending' END,
                    current_stage = %s,
                    error_message = %s,
                    updated_at = now(),
                    locked_by = CASE WHEN retries + 1 >= max_retries
                                     THEN locked_by ELSE NULL END,
                    lease_expires_at = CASE WHEN retries + 1 >= max_retries
                                            THEN lease_expires_at ELSE NULL END
                WHERE task_id = %s AND locked_by = %s AND status = 'processing'
                RETURNING status
                """,
                (stage, error, task_id, worker_id),
            ).fetchone()
            conn.commit()
            return row["status"] if row else None

    # ── 查询（运维性：阶段 / 重试次数 / 失败原因可查）─────

    def get(self, task_id: int) -> dict | None:
        with self._connect() as conn:
            return conn.execute(
                f"SELECT * FROM {self._table} WHERE task_id = %s",
                (task_id,),
            ).fetchone()

    def summary(self) -> dict[str, int]:
        """按状态计数：{status: count}。"""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT status, COUNT(*) AS cnt FROM {self._table} GROUP BY status"
            ).fetchall()
        return {r["status"]: r["cnt"] for r in rows}

    def list_stale(self) -> list[dict]:
        """租约已过期的 processing 任务（可被其他 Worker 恢复）。"""
        with self._connect() as conn:
            return conn.execute(
                f"""
                SELECT * FROM {self._table}
                WHERE status = 'processing' AND lease_expires_at < now()
                ORDER BY created_at ASC
                """
            ).fetchall()

    def list_by_status(self, status: str, limit: int = 100) -> list[dict]:
        with self._connect() as conn:
            return conn.execute(
                f"""
                SELECT * FROM {self._table}
                WHERE status = %s ORDER BY created_at ASC LIMIT %s
                """,
                (status, limit),
            ).fetchall()


class FileStateRepository:
    """rag_file_states 仓储 — 文件状态（Spec 5.4）。

    写路径只有 upsert_indexed / mark_deleted / remove：
    扫描比对只读不写 —— 新版本激活成功后才能提交文件状态，
    删除事务成功后才能标记 deleted。
    """

    def __init__(self, conn_str: str, table_prefix: str = ""):
        self._conn_str = normalize_conn_str(conn_str)
        self._table = t("rag_file_states", table_prefix)

    def _connect(self):
        return psycopg.connect(self._conn_str, row_factory=dict_row)

    def get(self, source: str) -> dict | None:
        with self._connect() as conn:
            return conn.execute(
                f"SELECT * FROM {self._table} WHERE source = %s",
                (source,),
            ).fetchone()

    def upsert_indexed(
        self,
        source: str,
        document_id: str,
        content_hash: str,
        file_mtime: float,
        file_size: int,
        active_version_id: str,
    ) -> None:
        """激活事务成功后提交文件状态（唯一的索引写入口）。"""
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {self._table}
                    (source, document_id, content_hash, file_mtime, file_size,
                     status, active_version_id, last_indexed_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, 'active', %s, now(), now())
                ON CONFLICT (source) DO UPDATE SET
                    document_id = excluded.document_id,
                    content_hash = excluded.content_hash,
                    file_mtime = excluded.file_mtime,
                    file_size = excluded.file_size,
                    status = 'active',
                    active_version_id = excluded.active_version_id,
                    last_indexed_at = excluded.last_indexed_at,
                    updated_at = now()
                """,
                (source, document_id, content_hash, file_mtime, file_size,
                 active_version_id),
            )
            conn.commit()

    def mark_deleted(self, source: str) -> bool:
        """删除事务成功后标记 deleted。"""
        with self._connect() as conn:
            cur = conn.execute(
                f"""
                UPDATE {self._table}
                SET status = 'deleted', updated_at = now()
                WHERE source = %s
                """,
                (source,),
            )
            conn.commit()
            return cur.rowcount > 0

    def list_tracked(self, directory_prefix: str = "") -> list[dict]:
        """列出 active 状态的已追踪文件（扫描比对用，只读）。"""
        with self._connect() as conn:
            if directory_prefix:
                return conn.execute(
                    f"""
                    SELECT * FROM {self._table}
                    WHERE status = 'active' AND source LIKE %s
                    ORDER BY source ASC
                    """,
                    (f"{directory_prefix}%",),
                ).fetchall()
            return conn.execute(
                f"SELECT * FROM {self._table} WHERE status = 'active' ORDER BY source ASC"
            ).fetchall()

    def remove(self, source: str) -> bool:
        """物理移除追踪记录（清理场景用，非常规路径）。"""
        with self._connect() as conn:
            cur = conn.execute(
                f"DELETE FROM {self._table} WHERE source = %s",
                (source,),
            )
            conn.commit()
            return cur.rowcount > 0


class VersionRepository:
    """rag_documents + rag_document_versions 仓储（Spec 5.1 / 5.2 / 7 / 8）。

    激活和删除是跨表单事务：版本状态、PGVector is_active、文件状态、
    知识库 revision 在同一事务内提交，任一失败整体回滚。
    """

    def __init__(self, conn_str: str, table_prefix: str = ""):
        self._conn_str = normalize_conn_str(conn_str)
        self._docs = t("rag_documents", table_prefix)
        self._versions = t("rag_document_versions", table_prefix)
        self._files = t("rag_file_states", table_prefix)
        self._kb = t("rag_kb_state", table_prefix)

    def _connect(self):
        return psycopg.connect(self._conn_str, row_factory=dict_row)

    # ── 文档 / 版本注册 ───────────────────────────────────

    def ensure_document(
        self,
        document_id: str,
        kb_id: str,
        source: str,
        source_filename: str | None = None,
    ) -> None:
        """登记文档（幂等，已存在则不变）。"""
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {self._docs}
                    (document_id, kb_id, source, source_filename)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (document_id, kb_id, source, source_filename),
            )
            conn.commit()

    def register_staging(
        self, document_id: str, version_id: str, content_hash: str
    ) -> None:
        """注册 staging 版本（幂等，已有记录不降级）。"""
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {self._versions} (document_id, version_id, content_hash)
                VALUES (%s, %s, %s)
                ON CONFLICT (document_id, version_id) DO NOTHING
                """,
                (document_id, version_id, content_hash),
            )
            conn.commit()

    def mark_validated(
        self,
        document_id: str,
        version_id: str,
        parent_count: int,
        child_count: int,
    ) -> bool:
        """staging → validated，记录父子块数量（active/retired 不允许回退）。"""
        with self._connect() as conn:
            cur = conn.execute(
                f"""
                UPDATE {self._versions}
                SET status = 'validated',
                    parent_count = %s,
                    child_count = %s,
                    validated_at = now()
                WHERE document_id = %s AND version_id = %s
                  AND status NOT IN ('active', 'retired')
                """,
                (parent_count, child_count, document_id, version_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def mark_failed(
        self, document_id: str, version_id: str, stage: str, reason: str
    ) -> bool:
        """置 failed 并记录失败阶段与原因（active/retired 不允许回退）。"""
        with self._connect() as conn:
            cur = conn.execute(
                f"""
                UPDATE {self._versions}
                SET status = 'failed', failure_stage = %s, failure_reason = %s
                WHERE document_id = %s AND version_id = %s
                  AND status NOT IN ('active', 'retired')
                """,
                (stage, reason, document_id, version_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def requeue_failed(self, document_id: str, version_id: str) -> bool:
        """failed → staging（Spec 5.2：failed 重试重新进 staging，不得直接激活）。"""
        with self._connect() as conn:
            cur = conn.execute(
                f"""
                UPDATE {self._versions}
                SET status = 'staging', validated_at = NULL
                WHERE document_id = %s AND version_id = %s AND status = 'failed'
                """,
                (document_id, version_id),
            )
            conn.commit()
            return cur.rowcount > 0

    # ── 查询 ──────────────────────────────────────────────

    def get_active(self, document_id: str) -> dict | None:
        with self._connect() as conn:
            return conn.execute(
                f"""
                SELECT * FROM {self._versions}
                WHERE document_id = %s AND status = 'active'
                """,
                (document_id,),
            ).fetchone()

    def get(self, document_id: str, version_id: str) -> dict | None:
        with self._connect() as conn:
            return conn.execute(
                f"""
                SELECT * FROM {self._versions}
                WHERE document_id = %s AND version_id = %s
                """,
                (document_id, version_id),
            ).fetchone()

    def list_versions(self, document_id: str) -> list[dict]:
        with self._connect() as conn:
            return conn.execute(
                f"""
                SELECT * FROM {self._versions}
                WHERE document_id = %s ORDER BY created_at ASC
                """,
                (document_id,),
            ).fetchall()

    def get_document(self, document_id: str) -> dict | None:
        with self._connect() as conn:
            return conn.execute(
                f"SELECT * FROM {self._docs} WHERE document_id = %s",
                (document_id,),
            ).fetchone()

    # ── 激活事务（Spec 第 7 章）───────────────────────────

    def activate_version(
        self,
        document_id: str,
        new_version_id: str,
        source: str,
        content_hash: str,
        file_mtime: float,
        file_size: int,
        small_collection: str,
        large_collection: str,
    ) -> str | None:
        """原子激活新版本（单事务），返回被退役的旧 version_id（无则 None）。

        事务步骤：锁 rag_documents 行 → 校验目标版本 status='validated'
        （否则抛 ValueError 整体回滚）→ 两 collection 中旧版本
        is_active=false、新版本 is_active=true → 旧版本 retired、
        新版本 active → rag_documents.active_version_id 指向新版 →
        rag_file_states upsert → rag_kb_state.active_revision+1。
        任一步失败整体回滚，旧版本继续服务。
        """
        with self._connect() as conn, conn.transaction():
            # 1. 锁定文档行
            doc = conn.execute(
                f"""
                SELECT document_id, kb_id, status, active_version_id
                FROM {self._docs} WHERE document_id = %s FOR UPDATE
                """,
                (document_id,),
            ).fetchone()
            if doc is None:
                raise ValueError(f"文档不存在: {document_id}")
            if doc["status"] == "deleted":
                raise ValueError(f"文档已删除，不可激活: {document_id}")

            # 2. 校验目标版本仍为 validated
            target = conn.execute(
                f"""
                SELECT status FROM {self._versions}
                WHERE document_id = %s AND version_id = %s
                """,
                (document_id, new_version_id),
            ).fetchone()
            if target is None:
                raise ValueError(f"版本不存在: {document_id}/{new_version_id}")
            if target["status"] != "validated":
                raise ValueError(
                    f"版本状态不可激活: {document_id}/{new_version_id} "
                    f"status={target['status']}（仅 validated 允许激活）"
                )

            old_version_id = doc["active_version_id"]

            # 3. 翻转 PGVector is_active（两 collection：旧 false、新 true）
            for col_name in (small_collection, large_collection):
                if old_version_id:
                    conn.execute(
                        """
                        UPDATE langchain_pg_embedding e
                        SET cmetadata = e.cmetadata || '{"is_active": false}'::jsonb
                        FROM langchain_pg_collection c
                        WHERE e.collection_id = c.uuid
                          AND c.name = %s
                          AND e.cmetadata->>'document_id' = %s
                          AND e.cmetadata->>'version_id' = %s
                        """,
                        (col_name, document_id, old_version_id),
                    )
                conn.execute(
                    """
                    UPDATE langchain_pg_embedding e
                    SET cmetadata = e.cmetadata || '{"is_active": true}'::jsonb
                    FROM langchain_pg_collection c
                    WHERE e.collection_id = c.uuid
                      AND c.name = %s
                      AND e.cmetadata->>'document_id' = %s
                      AND e.cmetadata->>'version_id' = %s
                    """,
                    (col_name, document_id, new_version_id),
                )

            # 4. 版本状态流转：旧 retired、新 active
            if old_version_id:
                conn.execute(
                    f"""
                    UPDATE {self._versions}
                    SET status = 'retired', retired_at = now()
                    WHERE document_id = %s AND version_id = %s
                    """,
                    (document_id, old_version_id),
                )
            conn.execute(
                f"""
                UPDATE {self._versions}
                SET status = 'active', activated_at = now(),
                    retired_at = NULL, failure_stage = NULL, failure_reason = NULL
                WHERE document_id = %s AND version_id = %s
                """,
                (document_id, new_version_id),
            )

            # 5. 文档指向新版本
            conn.execute(
                f"""
                UPDATE {self._docs}
                SET active_version_id = %s, updated_at = now()
                WHERE document_id = %s
                """,
                (new_version_id, document_id),
            )

            # 6. 文件状态（激活成功才允许提交，Spec 5.4）
            conn.execute(
                f"""
                INSERT INTO {self._files}
                    (source, document_id, content_hash, file_mtime, file_size,
                     status, active_version_id, last_indexed_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, 'active', %s, now(), now())
                ON CONFLICT (source) DO UPDATE SET
                    document_id = excluded.document_id,
                    content_hash = excluded.content_hash,
                    file_mtime = excluded.file_mtime,
                    file_size = excluded.file_size,
                    status = 'active',
                    active_version_id = excluded.active_version_id,
                    last_indexed_at = excluded.last_indexed_at,
                    updated_at = now()
                """,
                (source, document_id, content_hash, file_mtime, file_size,
                 new_version_id),
            )

            # 7. 知识库 revision +1
            conn.execute(
                f"""
                INSERT INTO {self._kb} (kb_id, active_revision)
                VALUES (%s, 1)
                ON CONFLICT (kb_id) DO UPDATE SET
                    active_revision = {self._kb}.active_revision + 1,
                    updated_at = now()
                """,
                (doc["kb_id"],),
            )

            return old_version_id

    # ── 删除事务（Spec 第 8 章，逻辑删除）─────────────────

    def delete_document(
        self,
        document_id: str,
        small_collection: str,
        large_collection: str,
    ) -> bool:
        """原子逻辑删除（单事务）。幂等：已 deleted 直接返回 True。

        事务步骤：锁文档 → active 父子块 is_active=false → active 版本
        retired → document deleted 且 active_version_id=NULL →
        file_state deleted → active_revision+1。
        不物理删除任何行。文档不存在返回 False。
        """
        with self._connect() as conn, conn.transaction():
            doc = conn.execute(
                f"""
                SELECT document_id, kb_id, source, status, active_version_id
                FROM {self._docs} WHERE document_id = %s FOR UPDATE
                """,
                (document_id,),
            ).fetchone()
            if doc is None:
                return False
            if doc["status"] == "deleted":
                return True  # 重复删除幂等

            active_version_id = doc["active_version_id"]

            # active 父子块下线
            if active_version_id:
                for col_name in (small_collection, large_collection):
                    conn.execute(
                        """
                        UPDATE langchain_pg_embedding e
                        SET cmetadata = e.cmetadata || '{"is_active": false}'::jsonb
                        FROM langchain_pg_collection c
                        WHERE e.collection_id = c.uuid
                          AND c.name = %s
                          AND e.cmetadata->>'document_id' = %s
                          AND e.cmetadata->>'version_id' = %s
                        """,
                        (col_name, document_id, active_version_id),
                    )
                conn.execute(
                    f"""
                    UPDATE {self._versions}
                    SET status = 'retired', retired_at = now()
                    WHERE document_id = %s AND status = 'active'
                    """,
                    (document_id,),
                )

            conn.execute(
                f"""
                UPDATE {self._docs}
                SET status = 'deleted', active_version_id = NULL, updated_at = now()
                WHERE document_id = %s
                """,
                (document_id,),
            )
            conn.execute(
                f"""
                UPDATE {self._files}
                SET status = 'deleted', updated_at = now()
                WHERE source = %s
                """,
                (doc["source"],),
            )
            conn.execute(
                f"""
                INSERT INTO {self._kb} (kb_id, active_revision)
                VALUES (%s, 1)
                ON CONFLICT (kb_id) DO UPDATE SET
                    active_revision = {self._kb}.active_revision + 1,
                    updated_at = now()
                """,
                (doc["kb_id"],),
            )
            return True


class KbStateRepository:
    """rag_kb_state 仓储 — 知识库 revision（Spec 5.5 / 9）。"""

    def __init__(self, conn_str: str, table_prefix: str = ""):
        self._conn_str = normalize_conn_str(conn_str)
        self._table = t("rag_kb_state", table_prefix)

    def _connect(self):
        return psycopg.connect(self._conn_str, row_factory=dict_row)

    def get(self, kb_id: str) -> dict:
        """获取知识库状态；不存在则初始化一行（revision 均为 0）。"""
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {self._table} (kb_id) VALUES (%s)
                ON CONFLICT (kb_id) DO NOTHING
                """,
                (kb_id,),
            )
            row = conn.execute(
                f"SELECT * FROM {self._table} WHERE kb_id = %s",
                (kb_id,),
            ).fetchone()
            conn.commit()
            return row

    def increment_active_revision(self, kb_id: str) -> int:
        """active_revision +1，返回新值。"""
        with self._connect() as conn:
            row = conn.execute(
                f"""
                INSERT INTO {self._table} (kb_id, active_revision)
                VALUES (%s, 1)
                ON CONFLICT (kb_id) DO UPDATE SET
                    active_revision = {self._table}.active_revision + 1,
                    updated_at = now()
                RETURNING active_revision
                """,
                (kb_id,),
            ).fetchone()
            conn.commit()
            return row["active_revision"]

    def update_bm25_revision(self, kb_id: str, revision: int) -> None:
        """记录 BM25 已构建到的 active_revision。"""
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {self._table} (kb_id, bm25_revision)
                VALUES (%s, %s)
                ON CONFLICT (kb_id) DO UPDATE SET
                    bm25_revision = excluded.bm25_revision,
                    updated_at = now()
                """,
                (kb_id, revision),
            )
            conn.commit()

    def needs_bm25_refresh(self, kb_id: str, local_revision: int) -> bool:
        """PostgreSQL active_revision 领先于本地 bm25_revision 时需要刷新。"""
        state = self.get(kb_id)
        return state["active_revision"] > local_revision
