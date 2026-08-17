"""统一索引应用服务（Spec V3 第 7 章，ENT-001 / ENT-002）。

`RagIndexingService` 是所有生产写入/删除入口（REST 路由、Agent 本地工具、
VectorStore 兼容方法）的唯一入口：

- 写入/删除只向 PG 控制面（rag_ingest_tasks）创建任务，不直接写删
  PGVector —— 实际写删由后台 Worker 消费任务后经版本化事务完成（INV-002）；
- source 统一经 `versioning.canonicalize_source` 规范化后再用于
  document_id / dedup_key / 任务 payload，删除与入库同一契约（ENT-002）；
- 进程内单例工厂 `get_indexing_service()` 保证 REST 与 Agent Tool
  共用同一服务实例（ENT-004）。
"""

from __future__ import annotations

import threading

from agent_material.rag_engine import versioning
from agent_material.utils.logger import get_logger

logger = get_logger("indexing_service")


class DirectWriteForbiddenError(RuntimeError):
    """pg_control_plane 开启时禁止兼容方法直写 PGVector（ENT-005 / INV-002）。"""


class RagIndexingService:
    """统一索引服务（ENT-001）：任务入队与任务状态查询的唯一入口。

    内部只调用 PG 控制面仓储（TaskRepository / FileStateRepository），
    不直接执行 PGVector 写删。
    """

    def __init__(
        self,
        conn_str: str,
        kb_id: str = "agent",
        table_prefix: str = "",
        ensure_worker=None,
    ):
        from agent_material.rag_engine.db.pg import (
            FileStateRepository,
            TaskRepository,
            ensure_schema,
        )

        ensure_schema(conn_str, table_prefix)
        self._tasks = TaskRepository(conn_str, table_prefix)
        self._file_states = FileStateRepository(conn_str, table_prefix)
        self._kb_id = kb_id
        # 入队后唤醒后台 Worker 的回调（由 VectorStore 注入，可为 None）
        self._ensure_worker = ensure_worker

    # ── Worker 唤醒 ───────────────────────────────────────

    def set_ensure_worker(self, callback) -> None:
        """补充注入 Worker 启动钩子（仅当尚未绑定）。"""
        if self._ensure_worker is None:
            self._ensure_worker = callback

    def _start_worker(self) -> None:
        if self._ensure_worker is not None:
            try:
                self._ensure_worker()
            except Exception as e:
                # Worker 启动失败不阻断入队（任务已持久化，租约恢复兜底）
                logger.error(f"[IndexingService] Worker 启动失败（任务已入队）: {e}")

    # ── 入队（写删唯一入口，INV-002）──────────────────────

    def enqueue_file_scan(
        self,
        directory: str | None = None,
        allow_types: list[str] | None = None,
    ) -> list[int]:
        """扫描目录并入队 ingest_file / delete_document 任务（只创建任务）。

        Returns:
            成功入队的 task_id 列表（dedup 命中的重复任务不计入）。
        """
        from agent_material.rag_engine.db.worker import pg_scan_and_enqueue
        from agent_material.utils.config_handler import vector_store_config

        directory = directory or vector_store_config["document_directory"]
        allow_types = allow_types or vector_store_config["allow_type"]
        task_ids = pg_scan_and_enqueue(
            directory=directory,
            allow_types=allow_types,
            task_repo=self._tasks,
            file_state_repo=self._file_states,
            kb_id=self._kb_id,
        )
        self._start_worker()
        return task_ids

    def enqueue_file(self, file_path: str) -> int | None:
        """单个已落盘文件封装 ingest_file 任务入队（上传接口专用）。

        与 pg_scan_and_enqueue 的单文件逻辑同一契约：
        canonical source / document_id / dedup_key 完全一致，因此上传入队
        与目录扫描入队对同一文件互为幂等。已追踪且内容 hash 未变的文件
        直接跳过（幂等），不重复入库。

        Returns:
            task_id；内容未变化或已有相同 pending/processing 任务时返回 None。
        """
        import os

        from agent_material.utils.path_tools import get_abs_path

        abs_path = get_abs_path(file_path)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        source = versioning.canonicalize_source(abs_path)
        with open(abs_path, "rb") as f:
            data = f.read()
        content_hash = versioning.compute_content_hash(data)
        tracked = self._file_states.get(source)
        if tracked is not None and tracked.get("content_hash") == content_hash:
            return None  # 已追踪且内容一致，幂等跳过
        document_id = versioning.compute_document_id(source, kb_id=self._kb_id)
        task_id = self._tasks.enqueue(
            "ingest_file",
            {"file_path": abs_path, "source": source},
            dedup_key=f"ingest:{document_id}:{content_hash}",
        )
        self._start_worker()
        return task_id

    def enqueue_text(self, text: str, source: str) -> int | None:
        """文本封装 ingest_text 任务入队。

        dedup_key 由内容 hash + canonical source 派生，同内容同来源不重复入队。
        Returns:
            task_id；已有相同 pending/processing 任务时返回 None。
        """
        canonical = versioning.canonicalize_source(source)
        content_hash = versioning.compute_content_hash(text.encode("utf-8"))
        document_id = versioning.compute_document_id(canonical, kb_id=self._kb_id)
        task_id = self._tasks.enqueue(
            "ingest_text",
            {"text": text, "source": canonical},
            dedup_key=f"ingest_text:{document_id}:{content_hash}",
        )
        self._start_worker()
        return task_id

    def enqueue_delete(self, source: str) -> dict:
        """按来源创建 delete_document 任务（source 不存在也允许，幂等）。

        Returns:
            {"source", "canonical_source", "task_id", "status"}；
            status=accepted 新建任务，deduplicated 已有相同进行中任务
            （task_id 为 None）。
        """
        canonical = versioning.canonicalize_source(source)
        document_id = versioning.compute_document_id(canonical, kb_id=self._kb_id)
        task_id = self._tasks.enqueue(
            "delete_document",
            {"source": canonical},
            dedup_key=f"delete:{document_id}",
        )
        self._start_worker()
        return {
            "source": source,
            "canonical_source": canonical,
            "task_id": task_id,
            "status": "accepted" if task_id is not None else "deduplicated",
        }

    def enqueue_batch_delete(self, sources: list[str]) -> list[dict]:
        """批量删除：每个 source 独立创建任务，互不影响。

        Returns:
            逐 source 明细列表（accepted / deduplicated / failed，
            failed 带 error 字段）。
        """
        results: list[dict] = []
        for source in sources:
            try:
                results.append(self.enqueue_delete(source))
            except Exception as e:
                logger.error(f"[IndexingService] 创建删除任务失败: {source}: {e}")
                results.append(
                    {
                        "source": source,
                        "canonical_source": None,
                        "task_id": None,
                        "status": "failed",
                        "error": str(e),
                    }
                )
        return results

    # ── 任务状态查询（只读）───────────────────────────────

    def get_task(self, task_id: int) -> dict | None:
        """查询单个任务状态（含阶段 / 锁 / 租约 / 重试 / 失败原因）。"""
        row = self._tasks.get(int(task_id))
        if row is None:
            return None
        return {
            k: (v.isoformat() if hasattr(v, "isoformat") else v)
            for k, v in row.items()
        }

    def get_task_summary(self) -> dict:
        """任务队列状态汇总（四个状态计数齐全，缺省补 0）。"""
        raw = self._tasks.summary()
        return {
            k: int(raw.get(k, 0))
            for k in ("pending", "processing", "completed", "failed")
        }

    def retry_failed_task(self, task_id: int) -> dict:
        """显式重试一个 failed 任务；其他状态失败关闭，不隐式重置预算。"""
        task_id = int(task_id)
        current = self._tasks.get(task_id)
        if current is None:
            raise LookupError(f"任务不存在: {task_id}")
        if current["status"] != "failed":
            raise ValueError(
                f"仅 failed 任务允许显式重试，当前状态: {current['status']}"
            )
        if not self._tasks.retry_failed(task_id):
            raise RuntimeError(f"任务状态已变化，重试未生效: {task_id}")
        self._start_worker()
        return self.get_task(task_id)


# ── 进程内单例工厂（ENT-004：REST 与 Agent Tool 共用）────────

_service: RagIndexingService | None = None
_factory_lock = threading.Lock()


def get_indexing_service(ensure_worker=None) -> RagIndexingService:
    """返回进程内唯一的 RagIndexingService 实例（懒加载）。

    ensure_worker：入队后唤醒后台 Worker 的回调（通常由 VectorStore
    注入其 _ensure_worker_started）；仅首次创建或尚未绑定时生效。
    """
    global _service
    with _factory_lock:
        if _service is None:
            from agent_material.utils.config_handler import vector_store_config

            _service = RagIndexingService(
                conn_str=vector_store_config.get("connection_string"),
                kb_id=vector_store_config.get("character_name", "agent"),
                ensure_worker=ensure_worker,
            )
        elif ensure_worker is not None:
            _service.set_ensure_worker(ensure_worker)
    return _service


def reset_indexing_service() -> None:
    """重置单例（仅测试用）。"""
    global _service
    with _factory_lock:
        _service = None
