"""文档版本注册表 + 任务队列恢复 单元测试（OFF-002 / OFF-004）。

使用 tmp_path 下的临时 sqlite 文件（绝对路径，绕过 get_abs_path 的项目根解析）。
"""

import sqlite3

import pytest

from agent_material.rag_engine.db.task_queue import TaskQueue, TaskType
from agent_material.rag_engine.db.version_registry import DocumentVersionRegistry


@pytest.fixture()
def db_path(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture()
def registry(db_path):
    return DocumentVersionRegistry(db_path=db_path)


def _register(registry, doc="doc1", ver="ver1"):
    registry.register_staging(doc, ver, source=f"datas/{doc}.pdf", content_hash=f"hash-{ver}")


class TestVersionStateMachine:
    def test_register_staging(self, registry):
        _register(registry)
        row = registry.get("doc1", "ver1")
        assert row is not None
        assert row["status"] == "staging"
        assert row["source"] == "datas/doc1.pdf"
        assert row["content_hash"] == "hash-ver1"

    def test_full_state_machine(self, registry):
        _register(registry)
        registry.mark_validated("doc1", "ver1", parent_count=3, child_count=12)
        row = registry.get("doc1", "ver1")
        assert row["status"] == "validated"
        assert row["parent_count"] == 3
        assert row["child_count"] == 12

        old = registry.activate("doc1", "ver1")
        assert old is None  # 首个激活版本，无旧版本可退役
        row = registry.get("doc1", "ver1")
        assert row["status"] == "active"
        assert row["activated_at"] is not None
        assert registry.get_active("doc1")["version_id"] == "ver1"

    def test_activate_retires_old_version(self, registry):
        _register(registry)
        registry.activate("doc1", "ver1")

        _register(registry, ver="ver2")
        registry.mark_validated("doc1", "ver2", 4, 16)
        old = registry.activate("doc1", "ver2")

        assert old == "ver1"
        old_row = registry.get("doc1", "ver1")
        assert old_row["status"] == "retired"
        assert old_row["retired_at"] is not None
        assert registry.get_active("doc1")["version_id"] == "ver2"
        assert len(registry.list_versions("doc1")) == 2

    def test_activate_rejects_bad_status(self, registry):
        _register(registry)
        registry.activate("doc1", "ver1")
        # 已 active / 已 retired 的版本不可再次激活
        with pytest.raises(ValueError):
            registry.activate("doc1", "ver1")
        _register(registry, ver="ver2")
        registry.activate("doc1", "ver2")
        with pytest.raises(ValueError):
            registry.activate("doc1", "ver1")  # retired
        # 不存在的版本
        with pytest.raises(ValueError):
            registry.activate("doc1", "no-such")

    def test_activate_clears_failure_fields(self, registry):
        _register(registry)
        registry.mark_failed("doc1", "ver1", stage="embed", reason="boom")
        registry.activate("doc1", "ver1")  # failed 允许重新激活
        row = registry.get("doc1", "ver1")
        assert row["status"] == "active"
        assert row["failure_stage"] is None
        assert row["failure_reason"] is None

    def test_single_active_enforced_by_index(self, registry, db_path):
        _register(registry)
        registry.activate("doc1", "ver1")
        # 直接 SQL 插入第二个 active，部分唯一索引应触发 IntegrityError
        conn = sqlite3.connect(db_path)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO document_versions (document_id, version_id, source, status) "
                "VALUES ('doc1', 'verX', 'datas/doc1.pdf', 'active')"
            )
        conn.close()

    def test_mark_failed_records_stage(self, registry):
        _register(registry)
        registry.mark_failed("doc1", "ver1", stage="split", reason="空文本 chunk")
        row = registry.get("doc1", "ver1")
        assert row["status"] == "failed"
        assert row["failure_stage"] == "split"
        assert "空文本" in row["failure_reason"]

    def test_register_staging_idempotent(self, registry):
        _register(registry)
        registry.activate("doc1", "ver1")
        # 重复注册已 active 的版本不得降级状态
        _register(registry)
        assert registry.get("doc1", "ver1")["status"] == "active"

    def test_get_retired(self, registry):
        _register(registry)
        registry.activate("doc1", "ver1")
        _register(registry, ver="ver2")
        registry.activate("doc1", "ver2")
        # 刚退役的记录不超过 30 天
        assert registry.get_retired(older_than_days=30) == []
        # 把 retired_at 拨到 2 天前，older_than_days=1 应能查到
        registry._conn.execute(
            "UPDATE document_versions SET retired_at=datetime('now', '-2 days') "
            "WHERE document_id='doc1' AND version_id='ver1'"
        )
        registry._conn.commit()
        retired = registry.get_retired(older_than_days=1)
        assert len(retired) == 1
        assert retired[0]["version_id"] == "ver1"

    def test_get_active_none(self, registry):
        assert registry.get_active("nope") is None
        assert registry.get("nope", "v") is None
        assert registry.list_versions("nope") == []


class TestRecoverStaleProcessing:
    def test_stale_processing_reset_to_pending(self, db_path):
        queue = TaskQueue(db_path=db_path)
        task_id = queue.enqueue(TaskType.INGEST_FILE, {"file_path": "a.pdf"})
        tasks = queue.dequeue()
        assert tasks[0].status == "pending"  # dequeue 返回快照
        assert queue.get_processing_count() == 1

        # 模拟崩溃：把 updated_at 拨到 1 小时前，并人为累加 retries
        queue._conn.execute(
            "UPDATE tasks SET updated_at=datetime('now', '-3600 seconds'), retries=2 WHERE id=?",
            (task_id,),
        )
        queue._conn.commit()

        recovered = queue.recover_stale_processing(timeout_seconds=600)
        assert recovered == 1
        status = queue.get_status(task_id)
        assert status["status"] == "pending"
        assert status["retries"] == 2  # retries 不增加、不丢失

    def test_fresh_processing_untouched(self, db_path):
        queue = TaskQueue(db_path=db_path)
        task_id = queue.enqueue(TaskType.INGEST_FILE, {"file_path": "a.pdf"})
        queue.dequeue()  # → processing，updated_at 为当前时间

        recovered = queue.recover_stale_processing(timeout_seconds=600)
        assert recovered == 0
        assert queue.get_status(task_id)["status"] == "processing"
