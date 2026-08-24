"""PostgreSQL 控制面仓储层（Spec V2 第 5–8 章，V3 第 8–10 章加固）。

设计要点：
- 每个方法内部自管理短连接（psycopg.connect），控制面 QPS 很低，无需连接池。
- 单语句方法用 autocommit；多语句方法用显式事务（with conn.transaction()），
  异常时整体回滚。
- 写路径严格受限：文件状态只能在激活/删除事务成功后更新（Spec 5.4），
  本模块不提供"扫描时写状态"的方法。
- 任务所有权 fencing（INV-003 / WRK-002/003）：heartbeat/complete/fail 与
  激活/删除事务均必须匹配 task_id + status=processing + locked_by +
  lease_generation 四元组，不匹配即失败关闭（0 行或 FencingRejectedError）。
"""

import hashlib
import os
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from agent_material.rag_engine.db.pg.schema import normalize_conn_str, t


class FencingRejectedError(RuntimeError):
    """事务内任务所有权校验失败（WRK-003）：任务 fencing 四元组不匹配。

    抛出即整体回滚；Worker 捕获后视为"所有权已丢失"，只退出、不得再
    调 fail/complete 覆盖新 Worker 的状态。
    """


class RowcountMismatchError(RuntimeError):
    """激活/删除事务内实际更新行数与 validated 记录不一致（INV-005）。

    抛出即整体回滚：旧 active/旧 FileState 不变、revision 不变、
    任务不得标记 completed。
    """


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

        WRK-006：相同 dedup_key 已存在 failed 任务时同样拒绝入队（返回 None），
        防止同内容任务通过反复新建无限重置重试预算；恢复只能靠显式
        retry_failed() 重置既有 failed 任务（或管理员手工处理）。
        """
        with self._connect() as conn:
            failed = conn.execute(
                f"""
                SELECT task_id FROM {self._table}
                WHERE dedup_key = %s AND status = 'failed'
                LIMIT 1
                """,
                (dedup_key,),
            ).fetchone()
            if failed is not None:
                return None
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
        每次领取（含过期回收）原子执行 lease_generation+1（WRK-002），
        返回任务 dict 中的 lease_generation 即本次执行的 fencing token；
        无可领取任务返回 None。
        """
        with self._connect() as conn:
            row = conn.execute(
                f"""
                UPDATE {self._table}
                SET status = 'processing',
                    locked_by = %s,
                    lease_generation = lease_generation + 1,
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

    def heartbeat(
        self,
        task_id: int,
        worker_id: str,
        lease_seconds: int = 300,
        *,
        lease_generation: int,
    ) -> bool:
        """刷新租约和心跳（WRK-003：四元组任一不匹配即返回 False）。"""
        with self._connect() as conn:
            cur = conn.execute(
                f"""
                UPDATE {self._table}
                SET heartbeat_at = now(),
                    lease_expires_at = now() + make_interval(secs => %s),
                    updated_at = now()
                WHERE task_id = %s AND locked_by = %s AND status = 'processing'
                  AND lease_generation = %s
                  AND lease_expires_at > now()
                """,
                (lease_seconds, task_id, worker_id, lease_generation),
            )
            conn.commit()
            return cur.rowcount > 0

    def update_stage(
        self,
        task_id: int,
        worker_id: str,
        stage: str,
        *,
        lease_generation: int,
    ) -> bool:
        """持有有效租约时持久化当前阶段，供在线排障与进度查询。"""
        with self._connect() as conn:
            cur = conn.execute(
                f"""
                UPDATE {self._table}
                SET current_stage = %s, updated_at = now()
                WHERE task_id = %s AND locked_by = %s AND status = 'processing'
                  AND lease_generation = %s AND lease_expires_at > now()
                """,
                (stage, task_id, worker_id, lease_generation),
            )
            conn.commit()
            return cur.rowcount > 0

    # ── 完成 / 失败 ───────────────────────────────────────

    def complete(self, task_id: int, worker_id: str, *, lease_generation: int) -> bool:
        """标记完成（WRK-003：四元组任一不匹配返回 False，调用方视为所有权丢失）。"""
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
                  AND lease_generation = %s
                  AND lease_expires_at > now()
                """,
                (task_id, worker_id, lease_generation),
            )
            conn.commit()
            return cur.rowcount > 0

    def fail(
        self,
        task_id: int,
        worker_id: str,
        error: str,
        stage: str | None = None,
        *,
        lease_generation: int,
    ) -> str | None:
        """记录失败：retries+1，记录 current_stage / error_message。

        retries 达到 max_retries → failed（保留锁信息便于排障）；
        否则回到 pending 并释放锁，等待重新领取。
        返回新状态；fencing 四元组不匹配（任务已易主）时返回 None，
        retries 不增加（WRK-006：只有持有效 fencing token 的明确失败才计数）。
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
                  AND lease_generation = %s
                  AND lease_expires_at > now()
                RETURNING status
                """,
                (stage, error, task_id, worker_id, lease_generation),
            ).fetchone()
            conn.commit()
            return row["status"] if row else None

    def retry_failed(self, task_id: int) -> bool:
        """显式重置 failed 任务为 pending（WRK-006 唯一的重试预算重置入口）。

        retries 清零、释放锁，任务可被重新 claim；error_message /
        current_stage（失败现场）保留用于排障。仅 failed 状态可重置。
        """
        with self._connect() as conn:
            cur = conn.execute(
                f"""
                UPDATE {self._table}
                SET status = 'pending',
                    retries = 0,
                    locked_by = NULL,
                    lease_expires_at = NULL,
                    heartbeat_at = NULL,
                    updated_at = now()
                WHERE task_id = %s AND status = 'failed'
                """,
                (task_id,),
            )
            conn.commit()
            return cur.rowcount > 0

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
        self._artifacts = t("rag_artifacts", table_prefix)
        self._files = t("rag_file_states", table_prefix)
        self._kb = t("rag_kb_state", table_prefix)
        self._tasks = t("rag_ingest_tasks", table_prefix)

    def _connect(self):
        return psycopg.connect(self._conn_str, row_factory=dict_row)

    # ── 文档级变更锁（WRK-005 / INV-004）───────────────────

    @staticmethod
    def _lock_key(kb_id: str, document_id: str) -> int:
        """(kb_id, document_id) → 64 位 advisory lock key（确定性）。"""
        digest = hashlib.sha256(f"{kb_id}\0{document_id}".encode()).digest()
        return int.from_bytes(digest[:8], "big", signed=True)

    @contextmanager
    def document_lock(self, kb_id: str, document_id: str):
        """PostgreSQL session 级 advisory lock，串行化同一文档的变更窗口。

        专用连接持有锁，覆盖 staging cleanup → staging write → validate →
        activate/delete（或失败退出），finally 释放。不同 document_id 的
        lock key 不同，可并行；同 key 哈希冲突只会退化为串行，不影响正确性。
        """
        key = self._lock_key(kb_id, document_id)
        conn = psycopg.connect(self._conn_str, autocommit=True)
        try:
            conn.execute("SELECT pg_advisory_lock(%s)", (key,))
            yield
        finally:
            try:
                conn.execute("SELECT pg_advisory_unlock(%s)", (key,))
            finally:
                conn.close()

    # ── 事务内公共步骤（ACT-002 / DEL-002）─────────────────

    def _lock_task_checked(
        self, conn, task_id: int, worker_id: str, lease_generation: int
    ) -> None:
        """事务内锁定任务行并校验 fencing 四元组（WRK-003，失败关闭）。"""
        row = conn.execute(
            f"""
            SELECT task_id FROM {self._tasks}
            WHERE task_id = %s AND status = 'processing'
              AND locked_by = %s AND lease_generation = %s
              AND lease_expires_at > now()
            FOR UPDATE
            """,
            (task_id, worker_id, lease_generation),
        ).fetchone()
        if row is None:
            raise FencingRejectedError(
                f"任务所有权校验失败: task_id={task_id} worker={worker_id} "
                f"generation={lease_generation}（任务已易主或不在 processing）"
            )

    def _complete_task_in_tx(
        self, conn, task_id: int, noop_stage: str | None = None
    ) -> None:
        """事务内标记任务 completed（行已被 _lock_task_checked 锁定）。

        noop_stage：幂等/no-op 完成时写入 current_stage 使终态语义明确
        （如 delete_noop / delete_already_deleted）。
        """
        if noop_stage is not None:
            conn.execute(
                f"""
                UPDATE {self._tasks}
                SET status = 'completed', completed_at = now(), updated_at = now(),
                    locked_by = NULL, lease_expires_at = NULL,
                    current_stage = %s
                WHERE task_id = %s
                """,
                (noop_stage, task_id),
            )
        else:
            conn.execute(
                f"""
                UPDATE {self._tasks}
                SET status = 'completed', completed_at = now(), updated_at = now(),
                    locked_by = NULL, lease_expires_at = NULL
                WHERE task_id = %s
                """,
                (task_id,),
            )

    @staticmethod
    def _collection_plan(
        small_collection: str, large_collection: str, parent_count: int, child_count: int
    ) -> list[tuple[str, int]]:
        """(collection, 期望行数) 列表；单粒度模式去重为单 collection 合并计数。"""
        if small_collection == large_collection:
            return [(small_collection, parent_count + child_count)]
        return [(large_collection, parent_count), (small_collection, child_count)]

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

    def replace_staging_artifacts(
        self, document_id: str, version_id: str, artifacts: list[dict]
    ) -> int:
        """幂等替换目标版本的 staging 图表实体。"""
        with self._connect() as conn, conn.transaction():
            version = conn.execute(
                f"""
                SELECT status FROM {self._versions}
                WHERE document_id = %s AND version_id = %s
                FOR UPDATE
                """,
                (document_id, version_id),
            ).fetchone()
            if version is None or version["status"] not in {"staging", "failed", "validated"}:
                raise ValueError(
                    f"版本状态不允许写入图表: {document_id}/{version_id} "
                    f"status={version['status'] if version else 'missing'}"
                )
            conn.execute(
                f"DELETE FROM {self._artifacts} WHERE document_id = %s AND version_id = %s",
                (document_id, version_id),
            )
            for artifact in artifacts:
                conn.execute(
                    f"""
                    INSERT INTO {self._artifacts} (
                        artifact_id, document_id, version_id, artifact_type, ordinal,
                        page_start, page_end, bbox, caption, nearby_context,
                        searchable_text, structured_content, asset_uri, content_hash,
                        parser, visual_description, visual_summary, visual_model,
                        visual_status, visual_confidence, visual_prompt_version,
                        visual_content_hash, visual_error_code, visual_asset_uri,
                        panel_assets, readability_status, legend_mapping_status,
                        numeric_fact_status, verified_key_values, status
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, 'staging'
                    )
                    """,
                    (
                        artifact["artifact_id"], document_id, version_id,
                        artifact["artifact_type"], artifact["ordinal"],
                        artifact.get("page_start"), artifact.get("page_end"),
                        Jsonb(artifact.get("bbox")) if artifact.get("bbox") is not None else None,
                        artifact.get("caption", ""),
                        artifact.get("nearby_context", ""),
                        artifact["searchable_text"],
                        artifact.get("structured_content", ""),
                        artifact.get("asset_uri"), artifact.get("content_hash", ""),
                        artifact.get("parser", "mineru"),
                        Jsonb(artifact.get("visual_description"))
                        if artifact.get("visual_description") is not None
                        else None,
                        artifact.get("visual_summary", ""),
                        artifact.get("visual_model"),
                        artifact.get("visual_status", "not_requested"),
                        artifact.get("visual_confidence"),
                        artifact.get("visual_prompt_version"),
                        artifact.get("visual_content_hash"),
                        artifact.get("visual_error_code"),
                        artifact.get("visual_asset_uri"),
                        Jsonb(artifact.get("panel_assets") or []),
                        artifact.get("readability_status", "unknown"),
                        artifact.get("legend_mapping_status", "not_applicable"),
                        artifact.get("numeric_fact_status", "not_applicable"),
                        Jsonb(artifact.get("verified_key_values") or []),
                    ),
                )
            return len(artifacts)

    def validate_artifacts(
        self, document_id: str, version_id: str, expected_count: int
    ) -> tuple[bool, dict]:
        """校验 staging 图表数量、身份字段和逻辑资源 URI。"""
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT COUNT(*) AS artifact_count,
                       COUNT(*) FILTER (
                           WHERE artifact_id = '' OR searchable_text = ''
                              OR artifact_type NOT IN ('figure', 'table')
                              OR (asset_uri IS NOT NULL AND asset_uri NOT LIKE 'rag-asset://%%')
                       ) AS invalid_count
                FROM {self._artifacts}
                WHERE document_id = %s AND version_id = %s AND status = 'staging'
                """,
                (document_id, version_id),
            ).fetchone()
        stats = {
            "artifact_count": int(row["artifact_count"]),
            "invalid_count": int(row["invalid_count"]),
            "expected_count": expected_count,
        }
        return (
            stats["artifact_count"] == expected_count
            and stats["invalid_count"] == 0,
            stats,
        )

    def list_artifacts(
        self, document_id: str, version_id: str | None = None, *, active_only: bool = True
    ) -> list[dict]:
        clauses = ["document_id = %s"]
        params: list = [document_id]
        if version_id is not None:
            clauses.append("version_id = %s")
            params.append(version_id)
        if active_only:
            clauses.append("status = 'active'")
        with self._connect() as conn:
            return conn.execute(
                f"SELECT * FROM {self._artifacts} WHERE {' AND '.join(clauses)} ORDER BY ordinal",
                tuple(params),
            ).fetchall()

    def get_artifact(
        self, artifact_id: str, *, active_only: bool = True
    ) -> dict | None:
        """按稳定 ID 查询单个图表实体；在线读取默认只允许 active 版本。"""
        clauses = ["artifact_id = %s"]
        params: list = [artifact_id]
        if active_only:
            clauses.append("status = 'active'")
        with self._connect() as conn:
            return conn.execute(
                f"SELECT * FROM {self._artifacts} WHERE {' AND '.join(clauses)}",
                tuple(params),
            ).fetchone()

    def mark_validated(
        self,
        document_id: str,
        version_id: str,
        parent_count: int,
        child_count: int,
        artifact_count: int = 0,
        visual_count: int = 0,
    ) -> bool:
        """staging → validated，记录父子块数量（active/retired 不允许回退）。"""
        with self._connect() as conn:
            cur = conn.execute(
                f"""
                UPDATE {self._versions}
                SET status = 'validated',
                    parent_count = %s,
                    child_count = %s,
                    artifact_count = %s,
                    visual_count = %s,
                    validated_at = now()
                WHERE document_id = %s AND version_id = %s
                  AND status NOT IN ('active', 'retired')
                """,
                (
                    parent_count,
                    child_count,
                    artifact_count,
                    visual_count,
                    document_id,
                    version_id,
                ),
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
                SET status = 'staging', parent_count = 0, child_count = 0,
                    artifact_count = 0, visual_count = 0, validated_at = NULL
                WHERE document_id = %s AND version_id = %s AND status = 'failed'
                """,
                (document_id, version_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def requeue_retired_for_deleted(self, document_id: str, version_id: str) -> bool:
        """将已删除文档的同内容 retired 版本重新放回 staging。

        仅当文档仍为 deleted 且不存在 active 版本时允许恢复，避免普通历史版本
        被意外回滚。恢复后仍会完整经过清理、重写、校验和原子激活。
        """
        with self._connect() as conn, conn.transaction():
            cur = conn.execute(
                f"""
                UPDATE {self._versions} v
                SET status = 'staging', parent_count = 0, child_count = 0,
                    artifact_count = 0, visual_count = 0,
                    validated_at = NULL, activated_at = NULL, retired_at = NULL,
                    failure_stage = NULL, failure_reason = NULL
                FROM {self._docs} d
                WHERE v.document_id = %s AND v.version_id = %s
                  AND v.status = 'retired'
                  AND d.document_id = v.document_id
                  AND d.status = 'deleted' AND d.active_version_id IS NULL
                """,
                (document_id, version_id),
            )
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

    # ── 激活事务（ACT-001~004 / INV-003 / INV-005）─────────

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
        visual_collection: str | None = None,
        *,
        task_id: int,
        worker_id: str,
        lease_generation: int,
    ) -> str | None:
        """原子激活新版本（单事务），返回被退役的旧 version_id（无则 None）。

        事务步骤（ACT-002）：锁定任务行并校验 fencing 四元组 → 锁
        rag_documents 行 → SELECT ... FOR UPDATE 锁目标版本行且要求
        status='validated' → 从锁定行读 parent_count/child_count（不信任
        Worker 传入）→ 旧版本父子块 is_active=false → 新版本父子块
        is_active=true 且 rowcount 必须等于 validated 记录且 >0 →
        旧版本 retired、新版本 active → rag_documents.active_version_id →
        rag_file_states upsert → active_revision+1 → 任务 completed
        （同一事务，ACT-004）。
        任一校验失败抛异常整体回滚（ACT-003）：旧 active/旧 FileState 不变、
        新版本不得部分 active、revision 不变、任务不得标 completed。
        """
        with self._connect() as conn, conn.transaction():
            # 0. 锁定任务行，校验 fencing（INV-003）
            self._lock_task_checked(conn, task_id, worker_id, lease_generation)

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
            # 2. 锁定目标版本行，要求 validated，从锁定行读取父子块数量
            target = conn.execute(
                f"""
                SELECT status, parent_count, child_count, artifact_count, visual_count
                FROM {self._versions}
                WHERE document_id = %s AND version_id = %s
                FOR UPDATE
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
            parent_count = target["parent_count"]
            child_count = target["child_count"]
            artifact_count = target["artifact_count"]
            visual_count = target["visual_count"]
            if parent_count <= 0 or child_count <= 0:
                raise RowcountMismatchError(
                    f"validated 记录计数非法: {document_id}/{new_version_id} "
                    f"parent_count={parent_count} child_count={child_count}（INV-005 要求均 >0）"
                )

            old_version_id = doc["active_version_id"]

            # 3. 旧版本父子块下线（collection 去重，单粒度模式不重复更新）
            if old_version_id:
                old_collections = [small_collection, large_collection]
                if visual_collection:
                    old_collections.append(visual_collection)
                for col_name in dict.fromkeys(old_collections):
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
                    f"""
                    UPDATE {self._artifacts}
                    SET status = 'retired', updated_at = now()
                    WHERE document_id = %s AND version_id = %s AND status = 'active'
                    """,
                    (document_id, old_version_id),
                )

            # 4. 新版本父子块上线：实际更新行数必须严格等于 validated 记录
            for col_name, expected in self._collection_plan(
                small_collection, large_collection, parent_count, child_count
            ):
                cur = conn.execute(
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
                if cur.rowcount != expected:
                    raise RowcountMismatchError(
                        f"activation_rowcount_mismatch: collection={col_name} "
                        f"document={document_id} version={new_version_id} "
                        f"期望 {expected} 行，实际更新 {cur.rowcount} 行（INV-005）"
                    )

            if visual_count:
                if not visual_collection:
                    raise RowcountMismatchError(
                        "版本包含图片向量，但激活事务未提供 visual collection"
                    )
                visual_cur = conn.execute(
                    """
                    UPDATE langchain_pg_embedding e
                    SET cmetadata = e.cmetadata || '{"is_active": true}'::jsonb
                    FROM langchain_pg_collection c
                    WHERE e.collection_id = c.uuid
                      AND c.name = %s
                      AND e.cmetadata->>'document_id' = %s
                      AND e.cmetadata->>'version_id' = %s
                    """,
                    (visual_collection, document_id, new_version_id),
                )
                if visual_cur.rowcount != visual_count:
                    raise RowcountMismatchError(
                        f"activation_visual_count_mismatch: collection={visual_collection} "
                        f"document={document_id} version={new_version_id} "
                        f"期望 {visual_count} 行，实际更新 {visual_cur.rowcount} 行"
                    )

            artifact_cur = conn.execute(
                f"""
                UPDATE {self._artifacts}
                SET status = 'active', updated_at = now()
                WHERE document_id = %s AND version_id = %s AND status = 'staging'
                """,
                (document_id, new_version_id),
            )
            if artifact_cur.rowcount != artifact_count:
                raise RowcountMismatchError(
                    f"activation_artifact_count_mismatch: document={document_id} "
                    f"version={new_version_id} 期望 {artifact_count} 行，"
                    f"实际更新 {artifact_cur.rowcount} 行"
                )

            # 5. 版本状态流转：旧 retired、新 active
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

            # 6. 文档指向新版本
            conn.execute(
                f"""
                UPDATE {self._docs}
                SET active_version_id = %s, status = 'active', source = %s,
                    source_filename = %s, updated_at = now()
                WHERE document_id = %s
                """,
                (new_version_id, source, os.path.basename(source), document_id),
            )

            # 7. 文件状态（激活成功才允许提交，Spec 5.4）
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

            # 8. 知识库 revision +1
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

            # 9. 任务 completed（同事务提交，ACT-004：崩溃后不存在
            #    "已激活但任务未完成"的中间态）
            self._complete_task_in_tx(conn, task_id)

            return old_version_id

    # ── 删除事务（DEL-002/003，逻辑删除）───────────────────

    def delete_document(
        self,
        document_id: str,
        small_collection: str,
        large_collection: str,
        visual_collection: str | None = None,
        *,
        task_id: int,
        worker_id: str,
        lease_generation: int,
    ) -> str:
        """原子逻辑删除（单事务）。

        事务步骤（DEL-002）：锁定并校验任务 fencing → 锁 document →
        不存在则任务 no-op completed → 已 deleted 则幂等 completed
        （不重复增加 revision）→ active 父子块 is_active=false 且实际
        下线行数必须等于 active version 的 parent_count/child_count →
        active 版本 retired → document deleted 且 active_version_id=NULL →
        file_state deleted → active_revision+1 → 任务 completed（同事务）。
        任一步异常整体回滚（DEL-003）：FileState 不变、revision 不变、
        旧版本继续可检索。不物理删除任何行。

        Returns:
            "deleted" 正常删除 / "already_deleted" 幂等完成 /
            "not_found" 文档不存在 no-op 完成。
        """
        with self._connect() as conn, conn.transaction():
            # 0. 锁定任务行，校验 fencing（INV-003）
            self._lock_task_checked(conn, task_id, worker_id, lease_generation)

            doc = conn.execute(
                f"""
                SELECT document_id, kb_id, source, status, active_version_id
                FROM {self._docs} WHERE document_id = %s FOR UPDATE
                """,
                (document_id,),
            ).fetchone()
            if doc is None:
                # 文档不存在：任务 no-op completed（终态语义明确）
                self._complete_task_in_tx(conn, task_id, noop_stage="delete_noop")
                return "not_found"
            if doc["status"] == "deleted":
                # 重复删除：幂等 completed，不重复增加 revision
                self._complete_task_in_tx(conn, task_id, noop_stage="delete_already_deleted")
                return "already_deleted"

            active_version_id = doc["active_version_id"]

            # active 父子块下线：实际行数必须与 active version 记录一致
            if active_version_id:
                active_ver = conn.execute(
                    f"""
                    SELECT status, parent_count, child_count, artifact_count, visual_count
                    FROM {self._versions}
                    WHERE document_id = %s AND version_id = %s
                    FOR UPDATE
                    """,
                    (document_id, active_version_id),
                ).fetchone()
                if active_ver is None:
                    raise RowcountMismatchError(
                        f"active version 记录缺失: {document_id}/{active_version_id}"
                    )
                for col_name, expected in self._collection_plan(
                    small_collection,
                    large_collection,
                    active_ver["parent_count"],
                    active_ver["child_count"],
                ):
                    cur = conn.execute(
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
                    if cur.rowcount != expected:
                        raise RowcountMismatchError(
                            f"delete_rowcount_mismatch: collection={col_name} "
                            f"document={document_id} version={active_version_id} "
                            f"期望下线 {expected} 行，实际 {cur.rowcount} 行"
                        )
                if active_ver["visual_count"]:
                    if not visual_collection:
                        raise RowcountMismatchError(
                            "活动版本包含图片向量，但删除事务未提供 visual collection"
                        )
                    visual_cur = conn.execute(
                        """
                        UPDATE langchain_pg_embedding e
                        SET cmetadata = e.cmetadata || '{"is_active": false}'::jsonb
                        FROM langchain_pg_collection c
                        WHERE e.collection_id = c.uuid
                          AND c.name = %s
                          AND e.cmetadata->>'document_id' = %s
                          AND e.cmetadata->>'version_id' = %s
                        """,
                        (visual_collection, document_id, active_version_id),
                    )
                    if visual_cur.rowcount != active_ver["visual_count"]:
                        raise RowcountMismatchError(
                            f"delete_visual_count_mismatch: collection={visual_collection} "
                            f"document={document_id} version={active_version_id} "
                            f"期望下线 {active_ver['visual_count']} 行，"
                            f"实际 {visual_cur.rowcount} 行"
                        )
                artifact_cur = conn.execute(
                    f"""
                    UPDATE {self._artifacts}
                    SET status = 'retired', updated_at = now()
                    WHERE document_id = %s AND version_id = %s AND status = 'active'
                    """,
                    (document_id, active_version_id),
                )
                if artifact_cur.rowcount != active_ver["artifact_count"]:
                    raise RowcountMismatchError(
                        f"delete_artifact_count_mismatch: document={document_id} "
                        f"version={active_version_id} 期望下线 "
                        f"{active_ver['artifact_count']} 行，实际 {artifact_cur.rowcount} 行"
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
            self._complete_task_in_tx(conn, task_id)
            return "deleted"


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
                    bm25_revision = GREATEST(
                        {self._table}.bm25_revision, excluded.bm25_revision
                    ),
                    updated_at = now()
                """,
                (kb_id, revision),
            )
            conn.commit()

    def needs_bm25_refresh(self, kb_id: str, local_revision: int) -> bool:
        """PostgreSQL active_revision 领先于本地 bm25_revision 时需要刷新。"""
        state = self.get(kb_id)
        return state["active_revision"] > local_revision
