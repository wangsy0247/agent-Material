"""会话删除与 SSE 断点续传端点的集成测试。

覆盖：
- DELETE /api/v1/memory/sessions/{id}：未认证 401、不存在/非本人 404、成功 200
- POST /api/v1/agent/chat/resume：未认证 401、buffer 无 run 404、
  回放已缓冲帧、Last-Event-ID 过滤
"""

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# backend/ 与 backend/packages 加入路径（pyproject 仅配置了 packages）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.gateway.app import app  # noqa: E402
from app.gateway.auth.dependencies import get_current_user, get_memory_manager  # noqa: E402
from app.gateway.auth.schemas import UserInfo  # noqa: E402
from app.gateway.sse import format_sse_event  # noqa: E402
from app.gateway.stream_buffer import stream_buffer  # noqa: E402

FAKE_USER_ID = "test-u"


def _fake_user() -> UserInfo:
    return UserInfo(user_id=FAKE_USER_ID, username="tester", created_at=time.time())


class FakeMemoryManager:
    """仅实现 delete_session 端点用到的接口。"""

    def __init__(self, delete_ok: bool):
        self.delete_ok = delete_ok
        self.deleted = []

    def delete_session(self, session_id, user_id=None):
        self.deleted.append((session_id, user_id))
        return self.delete_ok


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def authed_client():
    app.dependency_overrides[get_current_user] = _fake_user
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.pop(get_current_user, None)


def _override_mm(mm):
    app.dependency_overrides[get_memory_manager] = lambda: mm


def _clear_mm():
    app.dependency_overrides.pop(get_memory_manager, None)


# ------------------------------------------------------------------
# DELETE /memory/sessions/{id}
# ------------------------------------------------------------------

class TestDeleteSession:
    def test_unauthenticated_401(self, client):
        resp = client.delete("/api/v1/memory/sessions/s-any")
        assert resp.status_code == 401

    def test_not_found_or_not_owned_404(self, authed_client):
        mm = FakeMemoryManager(delete_ok=False)
        _override_mm(mm)
        try:
            resp = authed_client.delete("/api/v1/memory/sessions/s-other-user")
            assert resp.status_code == 404
            # 归属校验：user_id 必须传给底层删除
            assert mm.deleted == [("s-other-user", FAKE_USER_ID)]
        finally:
            _clear_mm()

    def test_delete_success_200(self, authed_client):
        mm = FakeMemoryManager(delete_ok=True)
        _override_mm(mm)
        try:
            resp = authed_client.delete("/api/v1/memory/sessions/s-mine")
            assert resp.status_code == 200
            assert resp.json()["code"] == 200
            assert mm.deleted == [("s-mine", FAKE_USER_ID)]
        finally:
            _clear_mm()


# ------------------------------------------------------------------
# POST /agent/chat/resume
# ------------------------------------------------------------------

def _seed_run(session_id, tokens, done=True):
    """向 stream_buffer 单例写入一个 run，返回 record。"""
    import asyncio

    async def seed():
        record = stream_buffer.begin_run(FAKE_USER_ID, session_id)
        for i, t in enumerate(tokens, start=1):
            frame = format_sse_event("token", t, event_id=record.next_id, index=i)
            await stream_buffer.append(record, frame)
        if done:
            await stream_buffer.finish(record)
        return record

    return asyncio.run(seed())


class TestChatResume:
    def test_unauthenticated_401(self, client):
        resp = client.post("/api/v1/agent/chat/resume", json={"session_id": "s-x"})
        assert resp.status_code == 401

    def test_missing_run_404(self, authed_client):
        resp = authed_client.post(
            "/api/v1/agent/chat/resume", json={"session_id": "s-not-exists"}
        )
        assert resp.status_code == 404

    def test_replays_buffered_frames(self, authed_client):
        _seed_run("s-resume-1", ["你好", "世界"])
        resp = authed_client.post(
            "/api/v1/agent/chat/resume", json={"session_id": "s-resume-1"}
        )
        assert resp.status_code == 200
        body = resp.text
        assert "id: 1\n" in body and "id: 2\n" in body
        assert "你好" in body and "世界" in body

    def test_last_event_id_filters_frames(self, authed_client):
        _seed_run("s-resume-2", ["旧帧", "新帧"])
        resp = authed_client.post(
            "/api/v1/agent/chat/resume",
            json={"session_id": "s-resume-2"},
            headers={"Last-Event-ID": "1"},
        )
        assert resp.status_code == 200
        assert "旧帧" not in resp.text
        assert "新帧" in resp.text
        assert "id: 1\n" not in resp.text
        assert "id: 2\n" in resp.text


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
