"""后台任务消费者 — daemon 线程轮询处理队列任务。"""

import threading
import time
import traceback

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

    POLL_INTERVAL = 2.0

    def __init__(self, task_queue: TaskQueue, file_tracker=None):
        self._queue = task_queue
        self._tracker = file_tracker
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
            target=self._run,
            daemon=True,
            name="rag-worker",
        )
        self._started = True
        self._thread.start()
        logger.info("[Worker] 后台 Worker 已启动")

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
                revived = self._queue.retry_failed(max_retries=3)
                if revived:
                    logger.info("worker_revived_tasks", extra={"count": revived})

                # 获取待处理任务
                tasks = self._queue.dequeue(batch_size=1)
                if tasks:
                    queue_worker_idle.set(0)
                for task in tasks:
                    self._process(task)
                if not tasks:
                    queue_worker_idle.inc(self.POLL_INTERVAL)
            except Exception as e:
                logger.error("worker_loop_error", extra={"error": str(e)})

            self._stop_event.wait(self.POLL_INTERVAL)

        logger.info("worker_loop_exited")

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

    def _handle_ingest(self, task):
        """处理文件入库任务。"""
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

    def _handle_delete(self, task):
        """处理删除任务。"""
        source = task.payload.get("source", task.payload.get("file_path", ""))
        if not source:
            logger.error("[Worker] 删除任务缺少 source")
            return

        multi_index = self._get_multi_index()
        multi_index.delete_by_source(source)

        if self._tracker:
            self._tracker.remove_source(source)
        logger.info(f"[Worker] 删除完成: {source}")

    def _handle_reindex(self, task):
        """处理全量重索引任务（重新扫描目录）。"""
        from agent_material.utils.config_handler import vector_store_config

        directory = task.payload.get("directory", vector_store_config.get("document_directory", "datas"))
        allow_types = task.payload.get("allow_types", vector_store_config.get("allow_type", [".pdf", ".txt", ".md"]))

        if not self._tracker:
            logger.error("[Worker] FileTracker 未初始化，无法执行增量扫描")
            return

        result = self._tracker.scan_directory(directory, allow_types)

        enqueued = 0
        # 新增和修改的文件入队
        for path in result["new"] + result["modified"]:
            self._queue.enqueue(TaskType.INGEST_FILE, {"file_path": path})
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
