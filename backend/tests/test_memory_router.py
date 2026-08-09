"""记忆系统路由端点的集成测试。

覆盖 backend/app/gateway/routers/memory.py 中未被其他测试覆盖的 7 个端点：
- POST   /memory/sessions                 创建会话
- GET    /memory/sessions                 列出会话
- POST   /memory/sessions/{id}/load       加载会话（404 分支）
- POST   /memory/sessions/{id}/end        结束会话
- GET    /memory/sessions/{id}/messages   获取消息（归属校验 404）
- POST   /memory/recall                   召回记忆
- GET    /memory/stats                    记忆统计

每个端点均测：未认证 401 + 认证后正常路径 + 关键 404/归属校验分支。
DELETE /memory/sessions/{id} 已由 test_session_delete.py 覆盖，此处不重复。
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

FAKE_USER_ID = "test-u"


def _fake_user() -> UserInfo:
    return UserInfo(user_id=FAKE_USER_ID, username="tester", created_at=time.time())


class FakeSessionInfo:
    """对齐 SessionInfoResponse 所需字段。"""

    def __init__(self, session_id="s-1", title="测试会话"):
        self.session_id = session_id
        self.title = title
        self.created_at = 1000.0
        self.updated_at = 2000.0
        self.message_count = 3
        self.total_tokens = 128
        self.source = "api"
        self.is_active = True
        self.end_reason = None


class FakeMessage:
    """对齐消息序列化所需字段（role/content/timestamp/name）。"""

    def __init__(self, role, content, name=None):
        self.role = role
        self.content = content
        self.timestamp = 1500.0
        self.name = name


class FakeMemoryEntry:
    """对齐 MemoryEntryResponse 所需字段。"""

    def __init__(self, category="fact", content="用户研究深度势能", confidence=0.123456789):
        self.category = category
        self.content = content
        self.confidence = confidence
        self.created_at = 1234.5
        self.tags = ["dp", "材料"]


class FakeSessionStore:
    """仅实现 get_session / get_messages（messages 端点用到）。"""

    def __init__(self, session_info=None, messages=None):
        self._session_info = session_info
        self._messages = messages or []
        self.get_session_calls = []
        self.get_messages_calls = []

    def get_session(self, session_id, user_id=None):
        self.get_session_calls.append((session_id, user_id))
        return self._session_info

    def get_messages(self, session_id, user_id=None):
        self.get_messages_calls.append((session_id, user_id))
        return self._messages


class FakeMemoryManager:
    """实现 7 个端点用到的全部 MemoryManager 接口。"""

    def __init__(
        self,
        sessions=None,
        load_ok=True,
        store=None,
        recall_entries=None,
        stats=None,
    ):
        self._sessions = sessions if sessions is not None else [FakeSessionInfo()]
        self._load_ok = load_ok
        self.session_store = store if store is not None else FakeSessionStore()
        self._recall_entries = recall_entries if recall_entries is not None else [FakeMemoryEntry()]
        self._stats = stats if stats is not None else {"session_count": 1, "memory_count": 2}

        self.created_with = None
        self.list_calls = []
        self.load_calls = []
        self.end_calls = []
        self.recall_calls = []
        self.stats_calls = []

    def create_session(self, user_id, title=None, source="api"):
        self.created_with = {"user_id": user_id, "title": title, "source": source}
        return "s-new"

    def list_sessions(self, user_id, limit=20):
        self.list_calls.append((user_id, limit))
        return self._sessions

    def load_session(self, session_id, user_id=None):
        self.load_calls.append((session_id, user_id))
        return self._load_ok

    async def end_session(self, session_id, user_id=None, reason=None):
        self.end_calls.append((session_id, user_id, reason))

    def recall_memory(self, user_id, query, top_k=5):
        self.recall_calls.append((user_id, query, top_k))
        return self._recall_entries

    def get_stats(self, user_id=None):
        self.stats_calls.append(user_id)
        return self._stats


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
# POST /memory/sessions
# ------------------------------------------------------------------

class TestCreateSession:
    def test_unauthenticated_401(self, client):
        resp = client.post("/api/v1/memory/sessions", json={})
        assert resp.status_code == 401

    def test_create_success_200(self, authed_client):
        mm = FakeMemoryManager()
        _override_mm(mm)
        try:
            resp = authed_client.post(
                "/api/v1/memory/sessions",
                json={"title": "深度势能讨论", "source": "web"},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["code"] == 200
            assert body["data"]["session_id"] == "s-new"
            # user_id 必须来自认证用户，title/source 透传
            assert mm.created_with == {
                "user_id": FAKE_USER_ID,
                "title": "深度势能讨论",
                "source": "web",
            }
        finally:
            _clear_mm()

    def test_create_with_defaults(self, authed_client):
        mm = FakeMemoryManager()
        _override_mm(mm)
        try:
            resp = authed_client.post("/api/v1/memory/sessions", json={})
            assert resp.status_code == 200
            assert mm.created_with["title"] is None
            assert mm.created_with["source"] == "api"
        finally:
            _clear_mm()


# ------------------------------------------------------------------
# GET /memory/sessions
# ------------------------------------------------------------------

class TestListSessions:
    def test_unauthenticated_401(self, client):
        resp = client.get("/api/v1/memory/sessions")
        assert resp.status_code == 401

    def test_list_success_200(self, authed_client):
        mm = FakeMemoryManager(sessions=[FakeSessionInfo("s-1"), FakeSessionInfo("s-2", "另一个会话")])
        _override_mm(mm)
        try:
            resp = authed_client.get("/api/v1/memory/sessions?limit=10")
            assert resp.status_code == 200
            body = resp.json()
            assert body["code"] == 200
            assert body["total"] == 2
            assert len(body["data"]) == 2
            first = body["data"][0]
            assert first["session_id"] == "s-1"
            assert first["title"] == "测试会话"
            assert first["message_count"] == 3
            assert first["total_tokens"] == 128
            assert first["is_active"] is True
            assert first["end_reason"] is None
            # user_id 与 limit 透传
            assert mm.list_calls == [(FAKE_USER_ID, 10)]
        finally:
            _clear_mm()

    def test_list_empty(self, authed_client):
        mm = FakeMemoryManager(sessions=[])
        _override_mm(mm)
        try:
            resp = authed_client.get("/api/v1/memory/sessions")
            assert resp.status_code == 200
            body = resp.json()
            assert body["total"] == 0
            assert body["data"] == []
        finally:
            _clear_mm()


# ------------------------------------------------------------------
# POST /memory/sessions/{id}/load
# ------------------------------------------------------------------

class TestLoadSession:
    def test_unauthenticated_401(self, client):
        resp = client.post("/api/v1/memory/sessions/s-x/load")
        assert resp.status_code == 401

    def test_load_not_found_404(self, authed_client):
        mm = FakeMemoryManager(load_ok=False)
        _override_mm(mm)
        try:
            resp = authed_client.post("/api/v1/memory/sessions/s-missing/load")
            assert resp.status_code == 404
            assert mm.load_calls == [("s-missing", FAKE_USER_ID)]
        finally:
            _clear_mm()

    def test_load_success_200(self, authed_client):
        mm = FakeMemoryManager(load_ok=True)
        _override_mm(mm)
        try:
            resp = authed_client.post("/api/v1/memory/sessions/s-mine/load")
            assert resp.status_code == 200
            body = resp.json()
            assert body["code"] == 200
            assert body["data"]["session_id"] == "s-mine"
            assert mm.load_calls == [("s-mine", FAKE_USER_ID)]
        finally:
            _clear_mm()


# ------------------------------------------------------------------
# POST /memory/sessions/{id}/end
# ------------------------------------------------------------------

class TestEndSession:
    def test_unauthenticated_401(self, client):
        resp = client.post("/api/v1/memory/sessions/s-x/end")
        assert resp.status_code == 401

    def test_end_success_200(self, authed_client):
        mm = FakeMemoryManager()
        _override_mm(mm)
        try:
            resp = authed_client.post("/api/v1/memory/sessions/s-mine/end")
            assert resp.status_code == 200
            assert resp.json()["code"] == 200
            # reason 固定为 user_close，user_id 来自认证用户
            assert mm.end_calls == [("s-mine", FAKE_USER_ID, "user_close")]
        finally:
            _clear_mm()


# ------------------------------------------------------------------
# GET /memory/sessions/{id}/messages
# ------------------------------------------------------------------

class TestGetSessionMessages:
    def test_unauthenticated_401(self, client):
        resp = client.get("/api/v1/memory/sessions/s-x/messages")
        assert resp.status_code == 401

    def test_not_owned_404(self, authed_client):
        """会话不属于当前用户（get_session 返回 None）→ 404。"""
        store = FakeSessionStore(session_info=None)
        mm = FakeMemoryManager(store=store)
        _override_mm(mm)
        try:
            resp = authed_client.get("/api/v1/memory/sessions/s-other-user/messages")
            assert resp.status_code == 404
            # 归属校验：user_id 必须传入底层查询
            assert store.get_session_calls == [("s-other-user", FAKE_USER_ID)]
            # 归属校验失败时不得再取消息
            assert store.get_messages_calls == []
        finally:
            _clear_mm()

    def test_messages_success_200(self, authed_client):
        store = FakeSessionStore(
            session_info=FakeSessionInfo("s-mine"),
            messages=[
                FakeMessage("user", "什么是深度势能？"),
                FakeMessage("assistant", "深度势能是一种机器学习势函数……", name="agent"),
            ],
        )
        mm = FakeMemoryManager(store=store)
        _override_mm(mm)
        try:
            resp = authed_client.get("/api/v1/memory/sessions/s-mine/messages")
            assert resp.status_code == 200
            body = resp.json()
            assert body["code"] == 200
            assert "2 条消息" in body["message"]
            data = body["data"]
            assert len(data) == 2
            assert data[0]["role"] == "user"
            assert data[0]["content"] == "什么是深度势能？"
            assert data[0]["name"] is None
            assert data[1]["role"] == "assistant"
            assert data[1]["name"] == "agent"
            assert store.get_messages_calls == [("s-mine", FAKE_USER_ID)]
        finally:
            _clear_mm()

    def test_messages_empty_200(self, authed_client):
        store = FakeSessionStore(session_info=FakeSessionInfo("s-mine"), messages=[])
        mm = FakeMemoryManager(store=store)
        _override_mm(mm)
        try:
            resp = authed_client.get("/api/v1/memory/sessions/s-mine/messages")
            assert resp.status_code == 200
            assert resp.json()["data"] == []
        finally:
            _clear_mm()


# ------------------------------------------------------------------
# POST /memory/recall
# ------------------------------------------------------------------

class TestRecallMemory:
    def test_unauthenticated_401(self, client):
        resp = client.post("/api/v1/memory/recall", json={"query": "深度势能"})
        assert resp.status_code == 401

    def test_recall_success_200(self, authed_client):
        mm = FakeMemoryManager(recall_entries=[FakeMemoryEntry()])
        _override_mm(mm)
        try:
            resp = authed_client.post(
                "/api/v1/memory/recall",
                json={"query": "深度势能", "top_k": 3},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["code"] == 200
            assert "1 条记忆" in body["message"]
            entry = body["data"][0]
            assert entry["category"] == "fact"
            assert entry["content"] == "用户研究深度势能"
            # confidence 保留 4 位小数
            assert entry["confidence"] == round(0.123456789, 4)
            assert entry["created_at"] == 1234.5
            assert entry["tags"] == ["dp", "材料"]
            # user_id / query / top_k 透传
            assert mm.recall_calls == [(FAKE_USER_ID, "深度势能", 3)]
        finally:
            _clear_mm()

    def test_recall_empty(self, authed_client):
        mm = FakeMemoryManager(recall_entries=[])
        _override_mm(mm)
        try:
            resp = authed_client.post(
                "/api/v1/memory/recall",
                json={"query": "不存在的内容"},
            )
            assert resp.status_code == 200
            assert resp.json()["data"] == []
        finally:
            _clear_mm()

    def test_recall_invalid_top_k_422(self, authed_client):
        mm = FakeMemoryManager()
        _override_mm(mm)
        try:
            resp = authed_client.post(
                "/api/v1/memory/recall",
                json={"query": "q", "top_k": 0},
            )
            assert resp.status_code == 422
        finally:
            _clear_mm()


# ------------------------------------------------------------------
# GET /memory/stats
# ------------------------------------------------------------------

class TestMemoryStats:
    def test_unauthenticated_401(self, client):
        resp = client.get("/api/v1/memory/stats")
        assert resp.status_code == 401

    def test_stats_success_200(self, authed_client):
        stats = {"session_count": 5, "memory_count": 42, "compression_count": 1}
        mm = FakeMemoryManager(stats=stats)
        _override_mm(mm)
        try:
            resp = authed_client.get("/api/v1/memory/stats")
            assert resp.status_code == 200
            body = resp.json()
            assert body["code"] == 200
            assert body["data"] == stats
            assert mm.stats_calls == [FAKE_USER_ID]
        finally:
            _clear_mm()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
