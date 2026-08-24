"""PostgreSQL 后台索引任务消费者。

TaskRepository.claim 原子领取 + lease_generation
  fencing（WRK-002/003）+ 租约心跳 + 丢失取消（WRK-004）+ 文档级 advisory
  lock 串行化（WRK-005）；版本激活/逻辑删除走 VersionRepository 单事务
  （事务内含所有权校验、行数校验与任务 completed）。
"""

import os
import socket
import threading
import time
from collections.abc import Callable

from agent_material.rag_engine.db.pg.repositories import (
    FencingRejectedError,
    RowcountMismatchError,
)
from agent_material.observability.structured_logger import get_logger
from agent_material.observability.metrics import (
    queue_depth,
    queue_task_duration,
    queue_worker_idle,
)

logger = get_logger("worker")


class LeaseLostError(RuntimeError):
    """任务执行期间租约/所有权已丢失（WRK-004）：安全点检查发现取消信号。

    不再执行后续写入或激活、不调 fail/complete 覆盖新 Worker 状态，
    仅记录 lease_lost 事件后退出。
    """


def validate_ingest_chunk_quality(chunks) -> list[str]:
    """对 loader 质量元数据做激活前的最后一道防线。

    非 PDF 或旧 loader 没有质量字段时保持兼容；一旦 PDF 明确声明质量失败、
    分片缺失，或需要 OCR 却没有执行 OCR，就拒绝该版本进入 store/activate。
    """
    problems: list[str] = []
    pdf_metadata = [
        chunk.metadata
        for chunk in chunks
        if chunk.metadata.get("file_type") == "pdf"
        or str(chunk.metadata.get("source", "")).lower().endswith(".pdf")
    ]
    if not pdf_metadata:
        return problems

    statuses = {
        meta.get("parse_quality_status")
        for meta in pdf_metadata
        if meta.get("parse_quality_status")
    }
    if "rejected" in statuses:
        problems.append("PDF 解析质量门禁未通过")

    failed_parts = max(int(meta.get("parts_failed") or 0) for meta in pdf_metadata)
    if failed_parts > 0:
        problems.append(f"PDF 存在 {failed_parts} 个失败分片")

    if any(
        meta.get("ocr_required") and not meta.get("ocr_route_selected")
        for meta in pdf_metadata
    ):
        problems.append("PDF 包含扫描页但未路由到 OCR 能力解析器")
    from agent_material.rag_engine.indexing.structured_artifacts import (
        validate_artifact_documents,
    )

    problems.extend(validate_artifact_documents(chunks))
    return problems


class BackgroundWorker:
    """后台 Worker — 单 daemon 线程轮询队列并处理任务。

    生命周期：
    - 首次入队时自动启动（懒初始化）
    - 进程退出时自动终止（daemon 线程）
    """

    def __init__(self, poll_interval: float = 2.0, max_retries: int = 3,
                 on_activated: Callable[[], None] | None = None,
                 task_repo=None, version_repo=None, file_state_repo=None,
                 kb_repo=None, lease_seconds: int = 300):
        self._poll_interval = poll_interval
        self._max_retries = max_retries
        self._on_activated = on_activated
        if task_repo is None or version_repo is None or file_state_repo is None:
            raise ValueError("BackgroundWorker 需要完整的 PostgreSQL 控制面仓储")
        self._task_repo = task_repo
        self._version_repo = version_repo
        self._file_state_repo = file_state_repo
        self._kb_repo = kb_repo
        self._lease_seconds = lease_seconds
        # worker_id 需在同进程内唯一（多 Worker 实例共存时避免锁归属混淆）
        self._worker_id = (
            f"{socket.gethostname()}:{os.getpid()}:{threading.current_thread().name}"
            f":{id(self):x}"
        )
        self._task_stage: str | None = None  # 当前任务阶段（pg 模式 fail 时记录）
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        # ── WRK-004 任务级执行上下文（每任务重置）──
        self._cancel_event: threading.Event | None = None  # 租约丢失取消信号
        self._lease_generation: int | None = None  # 当前任务 fencing token
        self._task_completed_in_tx = False  # 任务是否已在激活/删除事务内 completed

        # 懒加载 — 避免模块导入时触发重量级初始化
        self._multi_index = None
        self._pipeline = None

    # ── 生命周期 ──────────────────────────────────────────

    def start(self):
        """启动后台 Worker 线程（幂等）。启动后立即领取 pending /
        租约过期任务（WRK-001：恢复不依赖新写请求）。"""
        if self._started:
            return
        self._thread = threading.Thread(
            target=self._run_pg,
            daemon=True,
            name="rag-worker",
        )
        self._started = True
        self._thread.start()
        logger.info(
            "worker_started",
            extra={"mode": "pg", "worker_id": self._worker_id},
        )

    def stop(self):
        """请求停止 Worker（优雅退出）：停止领取新任务，当前任务在安全点退出。"""
        self._stop_event.set()
        logger.info("worker_stopping", extra={"worker_id": self._worker_id})

    def join(self, timeout: float | None = None):
        """等待 Worker 线程退出（shutdown 用）。"""
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        if not self.is_running():
            logger.info("worker_stopped", extra={"worker_id": self._worker_id})

    def is_running(self) -> bool:
        return self._started and self._thread is not None and self._thread.is_alive()

    # ── PostgreSQL 控制面主循环（Spec 6.2）─────────────────

    def _run_pg(self):
        """PG 模式主循环：原子 claim + 处理期间心跳续租。"""
        logger.info("worker_loop_started", extra={"mode": "pg", "worker_id": self._worker_id})
        while not self._stop_event.is_set():
            try:
                summary = self._task_repo.summary()
                for status in ("pending", "processing", "failed", "completed"):
                    queue_depth.labels(status=status).set(summary.get(status, 0))

                task = self._task_repo.claim(
                    self._worker_id, lease_seconds=self._lease_seconds
                )
                if task:
                    queue_worker_idle.set(0)
                    self._process_pg(task)
                else:
                    queue_worker_idle.inc(self._poll_interval)
            except Exception as e:
                logger.error("worker_loop_error", extra={"error": str(e)})

            self._stop_event.wait(self._poll_interval)

        logger.info("worker_loop_exited", extra={"mode": "pg"})

    def _heartbeat_loop(
        self,
        task_id: int,
        generation: int,
        stop_event: threading.Event,
        cancel_event: threading.Event,
    ):
        """长任务期间按 lease 的 1/3 间隔刷新租约（任务结束停止）。

        WRK-004 取消触发条件：
        - heartbeat 返回 False（fencing 不匹配 → 锁已易主）；
        - 心跳异常且已进入租约安全窗口，或连续 3 次异常。
        触发后置位共享 cancel_event，任务线程在安全点退出。
        """
        interval = max(self._lease_seconds / 3.0, 0.1)
        safety_window = max(interval, 0.2)
        lease_deadline = time.monotonic() + self._lease_seconds
        consecutive_errors = 0
        while not stop_event.wait(interval):
            try:
                ok = self._task_repo.heartbeat(
                    task_id,
                    self._worker_id,
                    lease_seconds=self._lease_seconds,
                    lease_generation=generation,
                )
                if not ok:
                    # 锁已易主（租约被回收、generation 已递增）—— 置取消信号，
                    # 任务线程在安全点停止，complete/fail 均不得再执行
                    if stop_event.is_set():
                        return  # 任务已结束（如事务内 completed），无需告警
                    logger.error(
                        "worker_heartbeat_lost",
                        extra={
                            "task_id": task_id,
                            "worker_id": self._worker_id,
                            "lease_generation": generation,
                        },
                    )
                    cancel_event.set()
                    return
                consecutive_errors = 0
                lease_deadline = time.monotonic() + self._lease_seconds
            except Exception as e:
                consecutive_errors += 1
                logger.error(
                    "worker_heartbeat_error",
                    extra={"task_id": task_id, "error": str(e),
                           "consecutive_errors": consecutive_errors},
                )
                lease_remaining = lease_deadline - time.monotonic()
                if consecutive_errors >= 3 or lease_remaining <= safety_window:
                    # 不等到数据库租约实际过期才停写：进入安全窗口即失败关闭。
                    logger.error(
                        "worker_heartbeat_unconfirmed",
                        extra={
                            "task_id": task_id,
                            "worker_id": self._worker_id,
                            "lease_remaining_seconds": max(lease_remaining, 0.0),
                        },
                    )
                    cancel_event.set()
                    return

    def _check_cancel(self):
        """WRK-004 安全点检查：staging 清理/写入/validate/activate/delete 前调用。"""
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise LeaseLostError("任务租约已丢失（心跳发现易主或无法确认所有权）")
        if self._stop_event.is_set():
            raise LeaseLostError("Worker 收到停止信号，任务在安全点退出")

    def _set_pg_stage(self, task_id: int, stage: str) -> None:
        """更新本地和 PG 阶段；失去有效租约时立即停止后续写入。"""
        self._task_stage = stage
        updater = getattr(self._task_repo, "update_stage", None)
        if updater is not None and not updater(
            task_id,
            self._worker_id,
            stage,
            lease_generation=self._lease_generation,
        ):
            if self._cancel_event is not None:
                self._cancel_event.set()
            raise LeaseLostError(f"更新任务阶段时租约已失效: {stage}")

    def _process_pg(self, task: dict):
        """处理单个 PG 任务（claim 返回的 dict，含 lease_generation fencing token）。"""
        t0 = time.time()
        task_id = task["task_id"]
        task_type = task["task_type"]
        generation = task["lease_generation"]
        self._task_stage = None
        self._lease_generation = generation
        self._task_completed_in_tx = False
        cancel_event = threading.Event()
        self._cancel_event = cancel_event
        logger.info(
            "worker_task_start",
            extra={
                "task_id": task_id,
                "task_type": task_type,
                "mode": "pg",
                "lease_generation": generation,
                "payload": str(task.get("payload"))[:200],
            },
        )
        hb_stop = threading.Event()
        hb_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(task_id, generation, hb_stop, cancel_event),
            daemon=True,
            name=f"rag-worker-hb-{task_id}",
        )
        hb_thread.start()
        try:
            if task_type == "ingest_file":
                self._handle_ingest_pg(task)
            elif task_type == "ingest_text":
                self._handle_ingest_text(task)
            elif task_type == "delete_document":
                self._handle_delete_pg(task)
            elif task_type == "reindex_all":
                self._handle_reindex_pg(task)
            elif task_type == "rebuild_bm25":
                self._task_stage = "bm25_refresh"
                if self._on_activated:
                    self._on_activated()
            else:
                raise ValueError(f"未知任务类型: {task_type}")
            if not self._task_completed_in_tx:
                # ingest/delete 正常路径已在激活/删除事务内 completed（ACT-004 /
                # DEL-002）；其余路径（幂等跳过、scan、bm25）由 Worker 带 fencing 完成
                ok = self._task_repo.complete(
                    task_id, self._worker_id, lease_generation=generation
                )
                if not ok:
                    # fencing 不匹配：任务已易主，不得误报完成
                    logger.error(
                        "worker_task_complete_rejected",
                        extra={"task_id": task_id, "lease_generation": generation},
                    )
            elapsed = time.time() - t0
            queue_task_duration.labels(type=task_type).observe(elapsed)
            logger.info(
                "worker_task_done",
                extra={"task_id": task_id, "mode": "pg", "duration_ms": elapsed * 1000},
            )
        except LeaseLostError as e:
            # WRK-004：所有权已丢失，不调 fail/complete 覆盖新 Worker 状态
            elapsed = time.time() - t0
            queue_task_duration.labels(type=task_type).observe(elapsed)
            logger.error(
                "lease_lost",
                extra={
                    "task_id": task_id,
                    "lease_generation": generation,
                    "stage": self._task_stage,
                    "reason": str(e),
                    "duration_ms": elapsed * 1000,
                },
            )
        except FencingRejectedError as e:
            # WRK-003：事务内所有权校验失败 —— 已整体回滚，所有权丢失仅退出
            elapsed = time.time() - t0
            queue_task_duration.labels(type=task_type).observe(elapsed)
            logger.error(
                "fencing_rejected",
                extra={
                    "task_id": task_id,
                    "lease_generation": generation,
                    "stage": self._task_stage,
                    "error": str(e),
                    "duration_ms": elapsed * 1000,
                },
            )
        except RowcountMismatchError as e:
            # INV-005：发布条数与 validated 记录不一致 —— 已整体回滚，
            # 本 Worker 仍持有所有权，走正常 retry/failed
            elapsed = time.time() - t0
            queue_task_duration.labels(type=task_type).observe(elapsed)
            logger.error(
                "activation_rowcount_mismatch",
                extra={
                    "task_id": task_id,
                    "lease_generation": generation,
                    "stage": self._task_stage,
                    "error": str(e),
                    "duration_ms": elapsed * 1000,
                },
            )
            self._fail_pg_task(task_id, generation, e)
        except Exception as e:
            elapsed = time.time() - t0
            queue_task_duration.labels(type=task_type).observe(elapsed)
            self._fail_pg_task(task_id, generation, e)
        finally:
            hb_stop.set()
            hb_thread.join(timeout=5)
            self._cancel_event = None

    def _fail_pg_task(self, task_id: int, generation: int, exc: Exception):
        """明确失败时 fail（WRK-006：仅持有效 fencing token 才增加 retries）。"""
        new_status = self._task_repo.fail(
            task_id,
            self._worker_id,
            str(exc),
            stage=self._task_stage,
            lease_generation=generation,
        )
        if new_status is None:
            # fencing 不匹配：执行期间所有权已丢失，不得覆盖新 Worker 状态
            logger.error(
                "lease_lost",
                extra={
                    "task_id": task_id,
                    "lease_generation": generation,
                    "stage": self._task_stage,
                    "reason": f"fail 被 fencing 拒绝: {exc}",
                },
            )
            return
        logger.error(
            "worker_task_failed",
            extra={
                "task_id": task_id,
                "mode": "pg",
                "stage": self._task_stage,
                "status": new_status,
                "error": str(exc),
            },
        )

    # ── 具体处理逻辑 ──────────────────────────────────────

    def _get_multi_index(self):
        if self._multi_index is None:
            from agent_material.rag_engine.indexing.multi_granularity import MultiGranularityIndex
            from agent_material.utils.config_handler import vector_store_config

            enabled = vector_store_config.get("multi_granularity", {}).get("enabled", False)
            self._multi_index = MultiGranularityIndex(enabled=enabled)
        return self._multi_index

    def _get_pipeline(self):
        if self._pipeline is None:
            from agent_material.rag_engine.document_pipeline import ProcessingPipeline
            self._pipeline = ProcessingPipeline()
        return self._pipeline

    def _handle_ingest(self, task):
        """兼容内部调用名；所有任务均使用 PostgreSQL 版本化入库。"""
        return self._handle_ingest_pg(task)

    def _atomic_bm25_refresh_enabled(self) -> bool:
        from agent_material.utils.config_handler import vector_store_config
        return vector_store_config.get("feature_flags", {}).get("atomic_bm25_refresh", True)

    # ── PG 控制面入库流程（Spec V2 第 6/7 章）──────────────

    @staticmethod
    def _pg_collections(multi_index) -> tuple[str, str]:
        """返回 (small_collection, large_collection)；单粒度模式两者同为 base。"""
        if getattr(multi_index, "enabled", True):
            return multi_index.small_name, multi_index.large_name
        return multi_index.base_name, multi_index.base_name

    def _handle_ingest_pg(self, task: dict):
        """PG 模式文件入库。"""
        from agent_material.rag_engine import versioning

        payload = task["payload"]
        file_path = payload["file_path"]  # 物理路径，仅用于读取文件
        source = payload.get("source") or versioning.canonicalize_source(file_path)
        with open(file_path, "rb") as f:
            data = f.read()
        content_hash = versioning.compute_content_hash(data)
        stat = os.stat(file_path)
        self._run_pg_ingest(
            task,
            source=source,
            content_hash=content_hash,
            file_mtime=stat.st_mtime,
            file_size=stat.st_size,
            produce_chunks=lambda ctx: self._get_pipeline().process(
                file_path, version_context=ctx
            ),
        )

    def _handle_ingest_text(self, task: dict):
        """PG 模式文本入库（Spec 11：/add-documents 统一走版本化索引服务）。"""
        from agent_material.rag_engine import versioning

        payload = task["payload"]
        text = payload["text"]
        source = payload["source"]
        data = text.encode("utf-8")
        self._run_pg_ingest(
            task,
            source=source,
            content_hash=versioning.compute_content_hash(data),
            file_mtime=time.time(),
            file_size=len(data),
            produce_chunks=lambda ctx: self._split_text_pg(text, source, ctx),
        )

    @staticmethod
    def _split_text_pg(text: str, source: str, version_context: dict):
        """纯文本切分：recursive → parent 两步链 + 统一版本上下文。

        ParentFirstSplitter 依赖 Markdown 标题结构，纯文本走
        HybridSplitter(["recursive", "parent"])（与 .txt 的 extension_pipeline 一致），
        再复用 Pipeline 的版本上下文逻辑生成确定性 ID。
        """
        from langchain_core.documents import Document

        from agent_material.rag_engine.document_pipeline import ProcessingPipeline
        from agent_material.rag_engine.text_spliter import HybridSplitter

        splitter = HybridSplitter(["recursive", "parent"])
        chunks = splitter.split_documents(
            [Document(page_content=text, metadata={"source": source})]
        )
        chunks = ProcessingPipeline._enhance_metadata(chunks)
        return ProcessingPipeline._apply_version_context(chunks, version_context)

    def _run_pg_ingest(
        self,
        task: dict,
        *,
        source: str,
        content_hash: str,
        file_mtime: float,
        file_size: int,
        produce_chunks,
    ):
        """PG 版本化入库主流程（ingest_file / ingest_text 共用）。

        阶段：pipeline → store → validate → activate → bm25_refresh。
        - 空 chunks 明确 failed（stage=pipeline），不得 silently completed（Spec 6.6）；
        - staging 清理失败抛错终止任务（Spec 6.5）；
        - validate 对比 pipeline 预期父子块数量与实际入库数量；
        - activate 是 VersionRepository 单事务（内含 is_active 翻转、版本流转、
          file_states 写入、active_revision+1），失败整体回滚、任务进重试；
        - rag_file_states 只能由激活事务写入（Spec 5.4），本流程不直接触碰。
        """
        from agent_material.rag_engine import versioning

        multi_index = self._get_multi_index()
        document_id = versioning.compute_document_id(source, kb_id=multi_index.base_name)
        version_id = versioning.compute_version_id(content_hash)
        small_col, large_col = self._pg_collections(multi_index)
        visual_col = getattr(multi_index, "visual_name", None)
        version_repo = self._version_repo
        log_ctx = {
            "task_id": task["task_id"],
            "document_id": document_id,
            "version_id": version_id,
            "lease_generation": self._lease_generation,
        }
        t0 = time.time()

        # 幂等跳过：相同内容版本已激活，仅补文件状态
        active = version_repo.get_active(document_id)
        if active and active["version_id"] == version_id:
            self._file_state_repo.upsert_indexed(
                source, document_id, content_hash, file_mtime, file_size, version_id
            )
            logger.info("ingest_skip_idempotent", extra=log_ctx)
            return

        # 登记文档 + staging 版本；failed 版本重试必须重新进 staging（Spec 5.2）
        version_repo.ensure_document(
            document_id, multi_index.base_name, source, os.path.basename(source)
        )
        version_repo.register_staging(document_id, version_id, content_hash)
        existing = version_repo.get(document_id, version_id)
        if existing and existing["status"] == "failed":
            version_repo.requeue_failed(document_id, version_id)
        elif existing and existing["status"] == "retired":
            document = version_repo.get_document(document_id)
            if document and document["status"] == "deleted":
                if not version_repo.requeue_retired_for_deleted(document_id, version_id):
                    raise ValueError(
                        f"已删除文档的版本无法重新进入 staging: {document_id}/{version_id}"
                    )

        def _stage(stage: str, fn):
            st = time.time()
            self._set_pg_stage(task["task_id"], stage)
            try:
                result = fn()
            except Exception as e:
                logger.error(
                    "ingest_stage_failed",
                    extra={**log_ctx, "stage": stage, "error": str(e)},
                )
                raise
            logger.info(
                "ingest_stage_done",
                extra={
                    **log_ctx,
                    "stage": stage,
                    "duration_ms": (time.time() - st) * 1000,
                    **({"detail": result} if isinstance(result, dict) else {}),
                },
            )
            return result

        # stage=pipeline：加载 + 清洗 + 切分 + 版本元数据
        chunks = _stage(
            "pipeline",
            lambda: produce_chunks(
                {"document_id": document_id, "version_id": version_id}
            ),
        )
        if not chunks:
            reason = "pipeline 产出空 chunks"
            version_repo.mark_failed(document_id, version_id, "pipeline", reason)
            self._task_stage = "pipeline"
            raise ValueError(reason)
        quality_problems = validate_ingest_chunk_quality(chunks)
        if quality_problems:
            reason = "; ".join(quality_problems)
            version_repo.mark_failed(document_id, version_id, "pipeline", reason)
            self._task_stage = "pipeline"
            raise ValueError(reason)

        from agent_material.rag_engine.indexing.structured_artifacts import (
            artifact_rows_from_documents,
        )
        from agent_material.rag_engine.indexing.visual_vector_index import (
            build_visual_index_documents,
        )

        artifact_rows = artifact_rows_from_documents(chunks)
        expected_visual_count = (
            len(build_visual_index_documents(chunks)) if visual_col else 0
        )

        # stage=store：清 staging 残留（失败抛错终止）→ 确定性写入
        def _store():
            multi_index.delete_by_version(document_id, version_id)
            version_repo.replace_staging_artifacts(
                document_id, version_id, artifact_rows
            )
            return multi_index.insert(chunks, source=source, versioned=True)

        # stage=validate：SQL 校验 + 预期/实际父子块数量比对（Spec 6.6）
        def _validate():
            expected_parents = len(
                {c.metadata.get("parent_id") for c in chunks if c.metadata.get("parent_id")}
            )
            expected_children = len(chunks)
            ok, stats = multi_index.validate_version(document_id, version_id)
            problems = []
            if not ok:
                problems.append(f"SQL 校验未通过: {stats}")
            if stats.get("parent_count") != expected_parents:
                problems.append(
                    f"父块数量不符: 预期 {expected_parents} 实际 {stats.get('parent_count')}"
                )
            if stats.get("child_count") != expected_children:
                problems.append(
                    f"子块数量不符: 预期 {expected_children} 实际 {stats.get('child_count')}"
                )
            if stats.get("visual_count", 0) != expected_visual_count:
                problems.append(
                    "图片向量数量不符: "
                    f"预期 {expected_visual_count} 实际 {stats.get('visual_count', 0)}"
                )
            artifacts_ok, artifact_stats = version_repo.validate_artifacts(
                document_id, version_id, len(artifact_rows)
            )
            if not artifacts_ok:
                problems.append(f"图表实体校验未通过: {artifact_stats}")
            if problems:
                reason = "; ".join(problems)
                version_repo.mark_failed(document_id, version_id, "validate", reason)
                raise ValueError(reason)
            version_repo.mark_validated(
                document_id,
                version_id,
                stats["parent_count"],
                stats["child_count"],
                artifact_count=len(artifact_rows),
                visual_count=expected_visual_count,
            )
            return {**stats, **artifact_stats}

        # stage=activate：单事务翻牌（内含 fencing 校验、行数校验、任务
        # completed）；失败整体回滚（仓储保证），任务进重试
        def _activate():
            old = version_repo.activate_version(
                document_id, version_id, source, content_hash,
                file_mtime, file_size, small_col, large_col, visual_col,
                task_id=task["task_id"], worker_id=self._worker_id,
                lease_generation=self._lease_generation,
            )
            # 任务已在激活事务内 completed（ACT-004），Worker 不再调 complete
            self._task_completed_in_tx = True
            return {"old_version_id": old}

        # ── staging 变更窗口（WRK-005 / INV-004）：文档级 advisory lock 串行化，
        #    覆盖 staging cleanup → staging write → validate → activate；
        #    每个阶段入口是 WRK-004 取消安全点，finally 释放锁 ──
        self._check_cancel()  # staging 清理前
        lock_t0 = time.time()
        logger.info(
            "document_lock_wait",
            extra={**log_ctx, "kb_id": multi_index.base_name},
        )
        with version_repo.document_lock(multi_index.base_name, document_id):
            logger.info(
                "document_lock_acquired",
                extra={
                    **log_ctx,
                    "wait_ms": (time.time() - lock_t0) * 1000,
                },
            )
            # 锁内幂等复查：等待锁期间同版本可能已被其他 Worker 激活
            current = version_repo.get_active(document_id)
            if current and current["version_id"] == version_id:
                logger.info("ingest_activate_skip", extra=log_ctx)
                return
            self._check_cancel()  # 父块/子块写入前
            parent_ids = _stage("store", _store)
            self._check_cancel()  # validate 前
            _stage("validate", _validate)
            self._check_cancel()  # activate 前
            _stage("activate", _activate)

        # stage=bm25_refresh：失败只记 error，不 fail 任务（revision 兜底后续阶段实现）
        if self._atomic_bm25_refresh_enabled() and self._on_activated:
            # 激活事务已把任务置为 completed；BM25 是事务后的 best-effort，
            # 仅保留本地/日志阶段，不能再用 processing fencing 更新任务行。
            self._task_stage = "bm25_refresh"
            st = time.time()
            try:
                self._on_activated()
                logger.info(
                    "ingest_stage_done",
                    extra={
                        **log_ctx,
                        "stage": "bm25_refresh",
                        "duration_ms": (time.time() - st) * 1000,
                    },
                )
            except Exception as e:
                logger.error(
                    "ingest_bm25_refresh_failed",
                    extra={**log_ctx, "stage": "bm25_refresh", "error": str(e)},
                )

        logger.info(
            "ingest_pg_done",
            extra={
                **log_ctx,
                "parent_count": len(parent_ids) if parent_ids else 0,
                "child_count": len(chunks),
                "duration_ms": (time.time() - t0) * 1000,
            },
        )

    def _handle_delete(self, task):
        """兼容内部调用名；所有删除均走 PostgreSQL 逻辑删除事务。"""
        return self._handle_delete_pg(task)

    def _handle_delete_pg(self, task: dict):
        """PG 模式逻辑删除（DEL-002/003）：单事务，失败抛错且不更新文件状态。

        事务内含 fencing 校验 / is_active 下线（行数校验）/ 版本 retired /
        文档 deleted / file_state deleted / active_revision+1 / 任务 completed。
        失败直接抛出：事务整体回滚，file_state 不变，任务进重试 ——
        事务外不再触碰文件状态。删除事务同样走文档级 advisory lock（INV-004）。
        """
        from agent_material.rag_engine import versioning

        payload = task["payload"]
        source = payload.get("source") or payload.get("file_path") or ""
        if not source:
            raise ValueError("删除任务缺少 source")
        source = versioning.canonicalize_source(source)
        self._set_pg_stage(task["task_id"], "delete")
        multi_index = self._get_multi_index()
        document_id = versioning.compute_document_id(source, kb_id=multi_index.base_name)
        small_col, large_col = self._pg_collections(multi_index)
        visual_col = getattr(multi_index, "visual_name", None)
        log_ctx = {
            "task_id": task["task_id"],
            "document_id": document_id,
            "source": source,
            "lease_generation": self._lease_generation,
        }

        self._check_cancel()  # delete 前安全点（WRK-004）
        lock_t0 = time.time()
        logger.info(
            "document_lock_wait",
            extra={**log_ctx, "kb_id": multi_index.base_name},
        )
        with self._version_repo.document_lock(multi_index.base_name, document_id):
            logger.info(
                "document_lock_acquired",
                extra={**log_ctx, "wait_ms": (time.time() - lock_t0) * 1000},
            )
            result = self._version_repo.delete_document(
                document_id, small_col, large_col, visual_col,
                task_id=task["task_id"], worker_id=self._worker_id,
                lease_generation=self._lease_generation,
            )
        # 任务已在删除事务内 completed（含 not_found / already_deleted 幂等完成）
        self._task_completed_in_tx = True
        if result == "not_found":
            # 从未入库过 → 幂等成功（no-op completed，current_stage=delete_noop）
            logger.info("delete_document_not_found", extra=log_ctx)
            return
        if result == "already_deleted":
            logger.info("delete_document_idempotent", extra=log_ctx)
            return
        logger.info("delete_document_done", extra=log_ctx)

        # 删除成功后 best-effort 刷新 BM25（失败只记 error，不 fail 任务）
        if self._on_activated:
            try:
                self._on_activated()
            except Exception as e:
                logger.error(
                    "delete_bm25_refresh_failed",
                    extra={**log_ctx, "stage": "bm25_refresh", "error": str(e)},
                )

    def _handle_reindex_pg(self, task: dict):
        """PG 模式全量重索引：扫描只创建任务，不提前改任何状态（Spec 6.1）。"""
        from agent_material.utils.config_handler import vector_store_config

        payload = task["payload"]
        directory = payload.get(
            "directory", vector_store_config.get("document_directory", "datas")
        )
        allow_types = payload.get(
            "allow_types", vector_store_config.get("allow_type", [".pdf", ".txt", ".md"])
        )
        self._task_stage = "scan"
        task_ids = pg_scan_and_enqueue(
            directory=directory,
            allow_types=allow_types,
            task_repo=self._task_repo,
            file_state_repo=self._file_state_repo,
            kb_id=self._get_multi_index().base_name,
        )
        logger.info(
            "reindex_scan_enqueued",
            extra={"task_id": task["task_id"], "enqueued": len(task_ids)},
        )

    def _handle_reindex(self, task):
        """兼容内部调用名；所有重索引均使用 PostgreSQL 控制面。"""
        return self._handle_reindex_pg(task)


# ── PG 控制面目录扫描（Spec 6.1）───────────────────────────


def pg_scan_and_enqueue(
    directory: str,
    allow_types: list[str],
    task_repo,
    file_state_repo,
    kb_id: str,
) -> list[int]:
    """PG 模式目录扫描：只创建任务，不提前更新任何状态（Spec 6.1）。

    - mtime + size 快速路径；变化时计算完整内容 hash 二次确认（内容一致则跳过）；
    - 新增/修改文件 → ingest_file 任务
      （dedup_key = ingest:{document_id}:{content_hash}，同文件同内容不重复入队）；
    - 消失的已追踪文件 → delete_document 任务（dedup_key = delete:{document_id}）；
    - 不提前更新 content_hash、不提前标记 deleted、不改 active_version_id。

    Returns:
        成功入队的 task_id 列表（dedup 命中的重复任务返回 None，不计入）。
    """
    from agent_material.rag_engine import versioning
    from agent_material.utils.path_tools import get_abs_path

    physical_files: dict[str, dict] = {}
    abs_scan_dir = get_abs_path(directory)
    if os.path.isdir(abs_scan_dir):
        allowed = {ext.lower() for ext in allow_types}
        for root, _, files in os.walk(abs_scan_dir):
            for name in files:
                path = os.path.join(root, name)
                if os.path.splitext(name)[1].lower() not in allowed:
                    continue
                try:
                    stat = os.stat(path)
                except OSError:
                    continue
                physical_files[path] = {"mtime": stat.st_mtime, "size": stat.st_size}
    # source 是跨 Windows/WSL、REST/Worker/迁移脚本共享的逻辑主键；物理路径
    # 仅用于本机 open/stat，不能直接进入控制面或向量 metadata。
    on_disk = {
        versioning.canonicalize_source(path): {**info, "file_path": path}
        for path, info in physical_files.items()
    }
    # 只比对扫描目录下的已追踪文件（目录外的追踪文件不属于本次扫描范围）
    abs_dir = versioning.canonicalize_source(get_abs_path(directory))
    tracked = {
        versioning.canonicalize_source(row["source"]): row
        for row in file_state_repo.list_tracked(abs_dir)
    }

    task_ids: list[int] = []
    for source, info in on_disk.items():
        row = tracked.get(source)
        if (
            row is not None
            and info["mtime"] == row["file_mtime"]
            and info["size"] == row["file_size"]
        ):
            continue  # 快速路径：mtime + size 未变
        try:
            with open(info["file_path"], "rb") as f:
                data = f.read()
        except OSError as e:
            logger.warning(f"[pg_scan] 无法读取文件 {info['file_path']}: {e}")
            continue
        content_hash = versioning.compute_content_hash(data)
        if row is not None and row["content_hash"] == content_hash:
            continue  # mtime/size 变了但内容一致，无需重建
        document_id = versioning.compute_document_id(source, kb_id=kb_id)
        task_id = task_repo.enqueue(
            "ingest_file",
            {"file_path": info["file_path"], "source": source},
            dedup_key=f"ingest:{document_id}:{content_hash}",
        )
        if task_id is not None:
            task_ids.append(task_id)

    # 消失的已追踪文件 → 删除任务（不提前标记 deleted）
    for source in tracked:
        if source not in on_disk:
            document_id = versioning.compute_document_id(source, kb_id=kb_id)
            task_id = task_repo.enqueue(
                "delete_document",
                {"source": source},
                dedup_key=f"delete:{document_id}",
            )
            if task_id is not None:
                task_ids.append(task_id)

    logger.info(
        f"[pg_scan] 扫描入队完成 | 磁盘 {len(on_disk)} | "
        f"已追踪 {len(tracked)} | 入队 {len(task_ids)}"
    )
    return task_ids
