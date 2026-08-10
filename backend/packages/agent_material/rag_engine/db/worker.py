"""后台任务消费者 — daemon 线程轮询处理队列任务。

双控制面（feature_flags.pg_control_plane 切换）：
- SQLite（回滚路径）：TaskQueue.dequeue + DocumentVersionRegistry（V1 流程）。
- PostgreSQL（Spec V2）：TaskRepository.claim 原子领取 + 租约心跳，
  版本激活/逻辑删除走 VersionRepository 单事务。
"""

import os
import socket
import threading
import time
from collections.abc import Callable

from agent_material.rag_engine.db.task_queue import TaskQueue, TaskType
from agent_material.observability.structured_logger import get_logger
from agent_material.observability.metrics import (
    queue_depth,
    queue_task_duration,
    queue_worker_idle,
)

logger = get_logger("worker")


class BackgroundWorker:
    """后台 Worker — 单 daemon 线程轮询队列并处理任务。

    生命周期：
    - 首次入队时自动启动（懒初始化）
    - 进程退出时自动终止（daemon 线程）
    """

    def __init__(self, task_queue: TaskQueue | None = None, file_tracker=None,
                 poll_interval: float = 2.0, max_retries: int = 3,
                 on_activated: Callable[[], None] | None = None,
                 version_registry=None,
                 task_repo=None, version_repo=None, file_state_repo=None,
                 kb_repo=None, lease_seconds: int = 300):
        self._queue = task_queue
        self._tracker = file_tracker
        self._poll_interval = poll_interval
        self._max_retries = max_retries
        self._on_activated = on_activated
        self._version_registry = version_registry
        # ── PG 控制面（Spec V2）：task_repo 非空即 pg 模式 ──
        self._pg = task_repo is not None
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

        # 懒加载 — 避免模块导入时触发重量级初始化
        self._multi_index = None
        self._pipeline = None

    # ── 生命周期 ──────────────────────────────────────────

    def start(self):
        """启动后台 Worker 线程（幂等）。"""
        if self._started:
            return
        self._thread = threading.Thread(
            target=self._run_pg if self._pg else self._run,
            daemon=True,
            name="rag-worker",
        )
        self._started = True
        if not self._pg:
            # 启动前先恢复崩溃遗留的超时 processing 任务（OFF-004）
            # pg 模式无需显式恢复：租约过期的 processing 任务会被 claim 自动回收
            recovered = self._queue.recover_stale_processing()
            if recovered > 0:
                logger.info("worker_recovered_stale_tasks", extra={"count": recovered})
        self._thread.start()
        logger.info(
            f"[Worker] 后台 Worker 已启动（{'PG' if self._pg else 'SQLite'} 控制面）"
        )

    def stop(self):
        """请求停止 Worker（优雅退出）。"""
        self._stop_event.set()
        logger.info("[Worker] 收到停止信号")

    def is_running(self) -> bool:
        return self._started and self._thread is not None and self._thread.is_alive()

    # ── 主循环 ────────────────────────────────────────────

    def _run(self):
        """Worker 主循环。"""
        logger.info("worker_loop_started")
        while not self._stop_event.is_set():
            try:
                # 更新队列深度指标
                summary = self._queue.get_summary()
                queue_depth.labels(status="pending").set(summary.get("pending", 0))
                queue_depth.labels(status="processing").set(summary.get("processing", 0))
                queue_depth.labels(status="failed").set(summary.get("failed", 0))
                queue_depth.labels(status="completed").set(summary.get("completed", 0))

                # 先复活可重试的失败任务
                revived = self._queue.retry_failed(max_retries=self._max_retries)
                if revived:
                    logger.info("worker_revived_tasks", extra={"count": revived})

                # 获取待处理任务
                tasks = self._queue.dequeue(batch_size=1)
                if tasks:
                    queue_worker_idle.set(0)
                for task in tasks:
                    self._process(task)
                if not tasks:
                    queue_worker_idle.inc(self._poll_interval)
            except Exception as e:
                logger.error("worker_loop_error", extra={"error": str(e)})

            self._stop_event.wait(self._poll_interval)

        logger.info("worker_loop_exited")

    # ── PG 控制面主循环（Spec 6.2）─────────────────────────

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

    def _heartbeat_loop(self, task_id: int, stop_event: threading.Event):
        """长任务期间按 lease 的 1/3 间隔刷新租约（任务结束停止）。"""
        interval = max(self._lease_seconds / 3.0, 0.1)
        while not stop_event.wait(interval):
            try:
                ok = self._task_repo.heartbeat(
                    task_id, self._worker_id, lease_seconds=self._lease_seconds
                )
                if not ok:
                    # 锁已易主（租约曾被回收）—— 停止心跳，complete/fail 会落空并记日志
                    logger.error(
                        "worker_heartbeat_lost",
                        extra={"task_id": task_id, "worker_id": self._worker_id},
                    )
                    return
            except Exception as e:
                logger.error(
                    "worker_heartbeat_error",
                    extra={"task_id": task_id, "error": str(e)},
                )

    def _process_pg(self, task: dict):
        """处理单个 PG 任务（claim 返回的 dict）。"""
        t0 = time.time()
        task_id = task["task_id"]
        task_type = task["task_type"]
        self._task_stage = None
        logger.info(
            "worker_task_start",
            extra={
                "task_id": task_id,
                "task_type": task_type,
                "mode": "pg",
                "payload": str(task.get("payload"))[:200],
            },
        )
        hb_stop = threading.Event()
        hb_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(task_id, hb_stop),
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
            elif task_type == TaskType.REINDEX_ALL.value:
                self._handle_reindex_pg(task)
            elif task_type == "rebuild_bm25":
                self._task_stage = "bm25_refresh"
                if self._on_activated:
                    self._on_activated()
            else:
                raise ValueError(f"未知任务类型: {task_type}")
            self._task_repo.complete(task_id, self._worker_id)
            elapsed = time.time() - t0
            queue_task_duration.labels(type=task_type).observe(elapsed)
            logger.info(
                "worker_task_done",
                extra={"task_id": task_id, "mode": "pg", "duration_ms": elapsed * 1000},
            )
        except Exception as e:
            elapsed = time.time() - t0
            queue_task_duration.labels(type=task_type).observe(elapsed)
            new_status = self._task_repo.fail(
                task_id, self._worker_id, str(e), stage=self._task_stage
            )
            logger.error(
                "worker_task_failed",
                extra={
                    "task_id": task_id,
                    "mode": "pg",
                    "stage": self._task_stage,
                    "status": new_status,
                    "error": str(e),
                    "duration_ms": elapsed * 1000,
                },
            )
        finally:
            hb_stop.set()
            hb_thread.join(timeout=5)

    # ── 任务分发 ──────────────────────────────────────────

    def _process(self, task):
        """处理单个任务。"""
        t0 = time.time()
        task_type = task.task_type
        logger.info(
            "worker_task_start",
            extra={"task_id": task.id, "task_type": task_type, "payload": str(task.payload)[:200]},
        )
        try:
            if task_type == TaskType.INGEST_FILE.value:
                self._handle_ingest(task)
            elif task_type == TaskType.DELETE_FILE.value:
                self._handle_delete(task)
            elif task.task_type == TaskType.REINDEX_ALL.value:
                self._handle_reindex(task)
            else:
                raise ValueError(f"未知任务类型: {task.task_type}")
            self._queue.complete(task.id)
            elapsed = time.time() - t0
            queue_task_duration.labels(type=task_type).observe(elapsed)
            logger.info(
                "worker_task_done",
                extra={"task_id": task.id, "duration_ms": elapsed * 1000},
            )
        except Exception as e:
            elapsed = time.time() - t0
            queue_task_duration.labels(type=task_type).observe(elapsed)
            logger.error(
                "worker_task_failed",
                extra={"task_id": task.id, "error": str(e), "duration_ms": elapsed * 1000},
            )
            self._queue.fail(task.id, str(e))

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

    def _get_registry(self):
        """获取文档版本注册表（与任务队列同一 db）。"""
        if self._version_registry is None:
            from agent_material.rag_engine.db.version_registry import DocumentVersionRegistry
            self._version_registry = DocumentVersionRegistry(db_path=self._queue.db_path)
        return self._version_registry

    def _versioned_ingest_enabled(self) -> bool:
        from agent_material.utils.config_handler import vector_store_config
        return vector_store_config.get("feature_flags", {}).get("versioned_ingest", True)

    def _handle_ingest(self, task):
        """处理文件入库任务。"""
        if self._pg:
            return self._handle_ingest_pg(task)
        if self._versioned_ingest_enabled():
            return self._handle_ingest_versioned(task)

        file_path = task.payload["file_path"]

        # 阶段 1：Pipeline 处理（加载 + 清洗 + 切分 + 元数据增强）
        pipeline = self._get_pipeline()
        child_chunks = pipeline.process(file_path)
        if not child_chunks:
            logger.warning(f"[Worker] 文件处理后为空: {file_path}")
            return

        # 阶段 2：写入多粒度索引
        multi_index = self._get_multi_index()
        parent_ids = multi_index.insert(child_chunks, source=file_path)

        # 阶段 3：更新文件追踪
        if self._tracker:
            self._tracker.record_indexed(file_path, parent_ids)

        logger.info(
            f"[Worker] 文件入库完成: {file_path} | "
            f"子chunk数={len(child_chunks)} | 父chunk数={len(parent_ids) if parent_ids else 0}"
        )

    # ── 版本化入库流程（OFF-002 / OFF-003 / OFF-004）──────

    def _handle_ingest_versioned(self, task):
        """版本化入库：staging → validated → active，逐阶段可恢复。"""
        from agent_material.rag_engine import versioning

        file_path = task.payload["file_path"]
        t0 = time.time()

        # 读取文件字节，计算确定性 ID
        with open(file_path, "rb") as f:
            data = f.read()
        content_hash = versioning.compute_content_hash(data)
        multi_index = self._get_multi_index()
        document_id = versioning.compute_document_id(
            file_path, kb_id=multi_index.base_name
        )
        version_id = versioning.compute_version_id(content_hash)

        registry = self._get_registry()
        log_ctx = {
            "task_id": task.id,
            "document_id": document_id,
            "version_id": version_id,
        }

        # 幂等跳过：相同内容版本已激活
        active = registry.get_active(document_id)
        if active and active["version_id"] == version_id:
            if self._tracker:
                self._tracker.record_indexed(
                    file_path, None,
                    **self._file_state_kwargs(file_path, task.payload),
                )
            logger.info("ingest_skip_idempotent", extra=log_ctx)
            return

        registry.register_staging(document_id, version_id, file_path, content_hash)

        def _stage(stage: str, fn, mark_failed: bool = True):
            """执行一个阶段；失败时按需标记 registry failed 后抛出。"""
            st = time.time()
            try:
                result = fn()
            except Exception as e:
                if mark_failed:
                    registry.mark_failed(document_id, version_id, stage, str(e))
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
            lambda: self._get_pipeline().process(
                file_path,
                version_context={"document_id": document_id, "version_id": version_id},
            ),
        )
        if not chunks:
            logger.warning(
                "ingest_empty_chunks", extra={**log_ctx, "stage": "pipeline"}
            )
            return  # 空结果不标 failed

        # stage=store：清理 staging 残留（重试幂等）→ 确定性写入
        parent_ids = _stage(
            "store",
            lambda: (
                multi_index.delete_by_version(document_id, version_id),
                multi_index.insert(chunks, source=file_path, versioned=True),
            )[1],
        )

        # stage=validate：OFF-003 最低校验条件
        def _validate():
            ok, stats = multi_index.validate_version(document_id, version_id)
            if not ok:
                raise ValueError(f"版本校验未通过: {stats}")
            registry.mark_validated(
                document_id, version_id,
                stats["parent_count"], stats["child_count"],
            )
            return stats

        _stage("validate", _validate)

        # stage=activate：registry 事务翻牌 + PG is_active 翻牌
        def _activate():
            current = registry.get_active(document_id)
            if current and current["version_id"] == version_id:
                # 崩溃恢复幂等：registry 已激活，只需补 PG 翻牌
                old = None
                logger.info("ingest_activate_skip_registry", extra=log_ctx)
            else:
                try:
                    old = registry.activate(document_id, version_id)
                except Exception as e:
                    registry.mark_failed(document_id, version_id, "activate", str(e))
                    raise
            # PG 失败时抛出但不标 failed：registry 已激活、PG 未翻牌 = 旧版本
            # 继续在线服务，重试时上面的幂等分支可续跑
            multi_index.set_active_version(document_id, version_id, old)

        _stage("activate", _activate, mark_failed=False)

        # stage=bm25_refresh：失败只记 error，不 fail 任务（6.3 降级表）
        if self._atomic_bm25_refresh_enabled() and self._on_activated:
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

        # 激活成功后才更新 FileTracker（OFF-002）
        if self._tracker:
            self._tracker.record_indexed(
                file_path,
                parent_ids,
                **self._file_state_kwargs(file_path, task.payload),
            )

        logger.info(
            "ingest_versioned_done",
            extra={
                **log_ctx,
                "parent_count": len(parent_ids) if parent_ids else 0,
                "child_count": len(chunks),
                "duration_ms": (time.time() - t0) * 1000,
            },
        )

    def _atomic_bm25_refresh_enabled(self) -> bool:
        from agent_material.utils.config_handler import vector_store_config
        return vector_store_config.get("feature_flags", {}).get("atomic_bm25_refresh", True)

    @staticmethod
    def _file_state_kwargs(file_path: str, payload: dict) -> dict:
        """从 payload 取 md5/mtime/size；缺失时现场 stat/算 md5 兜底。"""
        md5_hash = payload.get("md5_hash")
        file_mtime = payload.get("file_mtime")
        file_size = payload.get("file_size")
        if md5_hash and file_mtime is not None and file_size is not None:
            return {
                "md5_hash": md5_hash,
                "file_mtime": file_mtime,
                "file_size": file_size,
            }
        import os

        from agent_material.rag_engine.db.file_tracker import _compute_md5
        try:
            stat = os.stat(file_path)
            md5_hash = md5_hash or _compute_md5(file_path)
            if md5_hash is None:
                return {}
            return {
                "md5_hash": md5_hash,
                "file_mtime": stat.st_mtime,
                "file_size": stat.st_size,
            }
        except OSError:
            return {}

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

        file_path = task["payload"]["file_path"]
        with open(file_path, "rb") as f:
            data = f.read()
        content_hash = versioning.compute_content_hash(data)
        stat = os.stat(file_path)
        self._run_pg_ingest(
            task,
            source=file_path,
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
        version_repo = self._version_repo
        log_ctx = {
            "task_id": task["task_id"],
            "document_id": document_id,
            "version_id": version_id,
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

        def _stage(stage: str, fn):
            st = time.time()
            self._task_stage = stage
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

        # stage=store：清 staging 残留（失败抛错终止）→ 确定性写入
        def _store():
            multi_index.delete_by_version(document_id, version_id)
            return multi_index.insert(chunks, source=source, versioned=True)

        parent_ids = _stage("store", _store)

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
            if problems:
                reason = "; ".join(problems)
                version_repo.mark_failed(document_id, version_id, "validate", reason)
                raise ValueError(reason)
            version_repo.mark_validated(
                document_id, version_id, stats["parent_count"], stats["child_count"]
            )
            return stats

        _stage("validate", _validate)

        # stage=activate：单事务翻牌；失败整体回滚（仓储保证），任务进重试
        def _activate():
            current = version_repo.get_active(document_id)
            if current and current["version_id"] == version_id:
                # 崩溃恢复幂等：已激活则跳过
                logger.info("ingest_activate_skip", extra=log_ctx)
                return {"skipped": True}
            old = version_repo.activate_version(
                document_id, version_id, source, content_hash,
                file_mtime, file_size, small_col, large_col,
            )
            return {"old_version_id": old}

        _stage("activate", _activate)

        # stage=bm25_refresh：失败只记 error，不 fail 任务（revision 兜底后续阶段实现）
        if self._atomic_bm25_refresh_enabled() and self._on_activated:
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
        """处理删除任务。"""
        if self._pg:
            return self._handle_delete_pg(task)
        source = task.payload.get("source", task.payload.get("file_path", ""))
        if not source:
            logger.error("[Worker] 删除任务缺少 source")
            return

        multi_index = self._get_multi_index()
        multi_index.delete_by_source(source)

        if self._tracker:
            self._tracker.remove_source(source)
        logger.info(f"[Worker] 删除完成: {source}")

    def _handle_delete_pg(self, task: dict):
        """PG 模式逻辑删除（Spec 8）：单事务，失败抛错且不更新文件状态。

        事务内含 is_active 下线 / 版本 retired / 文档 deleted /
        file_state deleted / active_revision+1。失败直接抛出：事务整体回滚，
        file_state 不变，任务进重试 —— 事务外不再触碰文件状态。
        """
        from agent_material.rag_engine import versioning

        payload = task["payload"]
        source = payload.get("source") or payload.get("file_path") or ""
        if not source:
            raise ValueError("删除任务缺少 source")
        self._task_stage = "delete"
        multi_index = self._get_multi_index()
        document_id = versioning.compute_document_id(source, kb_id=multi_index.base_name)
        small_col, large_col = self._pg_collections(multi_index)
        log_ctx = {
            "task_id": task["task_id"],
            "document_id": document_id,
            "source": source,
        }

        deleted = self._version_repo.delete_document(document_id, small_col, large_col)
        if not deleted:
            # 从未入库过 → 幂等成功（无需任何状态变更）
            logger.info("delete_document_not_found", extra=log_ctx)
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
        """处理全量重索引任务（重新扫描目录）。"""
        if self._pg:
            return self._handle_reindex_pg(task)
        from agent_material.utils.config_handler import vector_store_config

        directory = task.payload.get("directory", vector_store_config.get("document_directory", "datas"))
        allow_types = task.payload.get("allow_types", vector_store_config.get("allow_type", [".pdf", ".txt", ".md"]))

        if not self._tracker:
            logger.error("[Worker] FileTracker 未初始化，无法执行增量扫描")
            return

        versioned = self._versioned_ingest_enabled()
        result = self._tracker.scan_directory(
            directory, allow_types, defer_state_update=versioned
        )
        file_info = result.get("file_info", {})

        enqueued = 0
        # 新增和修改的文件入队
        for path in result["new"] + result["modified"]:
            payload = {"file_path": path}
            if versioned:
                payload.update(file_info.get(path, {}))
            self._queue.enqueue(TaskType.INGEST_FILE, payload)
            enqueued += 1

        # 已删除的文件入队删除
        for path in result["deleted"]:
            self._queue.enqueue(TaskType.DELETE_FILE, {"source": path, "file_path": path})
            enqueued += 1

        logger.info(
            f"[Worker] 扫描入队完成 | 新增 {len(result['new'])} | "
            f"修改 {len(result['modified'])} | 未变 {len(result['unchanged'])} | "
            f"已删 {len(result['deleted'])} | 共入队 {enqueued}"
        )


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
    from agent_material.rag_engine.db.file_tracker import scan_directory_files

    on_disk = scan_directory_files(directory, allow_types)
    # 只比对扫描目录下的已追踪文件（目录外的追踪文件不属于本次扫描范围）
    from agent_material.utils.path_tools import get_abs_path

    abs_dir = get_abs_path(directory)
    tracked = {
        row["source"]: row for row in file_state_repo.list_tracked(abs_dir)
    }

    task_ids: list[int] = []
    for path, info in on_disk.items():
        row = tracked.get(path)
        if (
            row is not None
            and info["mtime"] == row["file_mtime"]
            and info["size"] == row["file_size"]
        ):
            continue  # 快速路径：mtime + size 未变
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError as e:
            logger.warning(f"[pg_scan] 无法读取文件 {path}: {e}")
            continue
        content_hash = versioning.compute_content_hash(data)
        if row is not None and row["content_hash"] == content_hash:
            continue  # mtime/size 变了但内容一致，无需重建
        document_id = versioning.compute_document_id(path, kb_id=kb_id)
        task_id = task_repo.enqueue(
            "ingest_file",
            {"file_path": path},
            dedup_key=f"ingest:{document_id}:{content_hash}",
        )
        if task_id is not None:
            task_ids.append(task_id)

    # 消失的已追踪文件 → 删除任务（不提前标记 deleted）
    for path in tracked:
        if path not in on_disk:
            document_id = versioning.compute_document_id(path, kb_id=kb_id)
            task_id = task_repo.enqueue(
                "delete_document",
                {"source": path},
                dedup_key=f"delete:{document_id}",
            )
            if task_id is not None:
                task_ids.append(task_id)

    logger.info(
        f"[pg_scan] 扫描入队完成 | 磁盘 {len(on_disk)} | "
        f"已追踪 {len(tracked)} | 入队 {len(task_ids)}"
    )
    return task_ids
