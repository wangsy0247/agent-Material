"""system 与 agent 路由端点测试。

覆盖：
- GET /api/v1/system/health：200 与 HealthResponse 结构
- POST /api/v1/agent/chat：未认证 401、缺 question 422、空问题 400、
  非流式 200、SSE 流式 200（token/done 帧）
- POST /api/v1/agent/approve：未认证 401、非法 action 422、
  approve 动作的 SSE 正常路径

（/api/v1/agent/chat/resume 已由 test_session_delete.py 覆盖，此处不重复。）
"""

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# backend/ 与 backend/packages 加入路径（pyproject 仅配置了 packages）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.gateway.app import app  # noqa: E402
from app.gateway.auth.dependencies import (  # noqa: E402
    get_agent_with_memory,
    get_current_user,
    get_memory_manager,
)
from app.gateway.auth.schemas import UserInfo  # noqa: E402

FAKE_USER_ID = "test-u"
FAKE_SESSION_ID = "sess-fake"


def _fake_user() -> UserInfo:
    return UserInfo(user_id=FAKE_USER_ID, username="tester", created_at=time.time())


class FakeLongTerm:
    """长期记忆 fake：user_file/project_file 均不存在。"""

    class _Missing:
        def exists(self):
            return False

    user_file = _Missing()
    project_file = _Missing()


class FakeMemoryManager:
    """仅实现 /agent/chat 端点用到的接口（session_id 走 default 分支）。"""

    def __init__(self, total_tokens: int = 42):
        self._session_pool: dict = {}
        self._total_tokens = total_tokens
        self.created_sessions = []

    def create_session(self, user_id=None, title=None):
        self.created_sessions.append((user_id, title))
        return FAKE_SESSION_ID

    def _get_long_term(self, user_id):
        return FakeLongTerm()

    def get_session_stats(self, session_id, user_id=None):
        return {"total_tokens": self._total_tokens}


class FakeAgent:
    """ReactAgent fake：excutor / excutor_stream / resume_stream。"""

    def __init__(self, invoke_result="fake answer", stream_chunks=None, resume_chunks=None):
        self.invoke_result = invoke_result
        self.stream_chunks = stream_chunks or []
        self.resume_chunks = resume_chunks or []
        self.stream_calls = []
        self.resume_calls = []

    async def excutor(self, question, session_id=None, user_id=None):
        return self.invoke_result

    def excutor_stream(self, question, session_id=None, user_id=None):
        self.stream_calls.append((question, session_id, user_id))

        async def gen():
            for chunk in self.stream_chunks:
                yield chunk

        return gen()

    def resume_stream(self, session_id, approval_result, user_id=None):
        self.resume_calls.append((session_id, approval_result, user_id))

        async def gen():
            for chunk in self.resume_chunks:
                yield chunk

        return gen()


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


def _override_agent(agent):
    app.dependency_overrides[get_agent_with_memory] = lambda: agent


def _override_mm(mm):
    app.dependency_overrides[get_memory_manager] = lambda: mm


def _clear_overrides(*deps):
    for dep in deps:
        app.dependency_overrides.pop(dep, None)


# ------------------------------------------------------------------
# GET /api/v1/system/health
# ------------------------------------------------------------------

class TestSystemHealth:
    def test_health_200_and_structure(self, client):
        resp = client.get("/api/v1/system/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 200
        assert body["message"] == "服务运行正常"
        data = body["data"]
        assert data["status"] == "healthy"
        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0
        assert data["version"] == "0.1.0"


# ------------------------------------------------------------------
# POST /api/v1/agent/chat
# ------------------------------------------------------------------

class TestAgentChat:
    def test_unauthenticated_401(self, client):
        resp = client.post("/api/v1/agent/chat", json={"question": "你好"})
        assert resp.status_code == 401

    def test_missing_question_422(self, authed_client):
        resp = authed_client.post("/api/v1/agent/chat", json={})
        assert resp.status_code == 422

    def test_blank_question_400(self, authed_client):
        # 全空白通过 min_length=1 校验，由端点逻辑判空返回 400
        resp = authed_client.post("/api/v1/agent/chat", json={"question": "   "})
        assert resp.status_code == 400
        assert resp.json()["detail"] == "问题不能为空"

    def test_non_stream_200(self, authed_client):
        agent = FakeAgent(invoke_result="这是完整回答")
        mm = FakeMemoryManager()
        _override_agent(agent)
        _override_mm(mm)
        try:
            resp = authed_client.post(
                "/api/v1/agent/chat",
                json={"question": "什么是 DP 模型？", "stream": False},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["answer"] == "这是完整回答"
            assert body["session_id"] == FAKE_SESSION_ID
        finally:
            _clear_overrides(get_agent_with_memory, get_memory_manager)

    def test_stream_sse_tokens_and_done(self, authed_client):
        agent = FakeAgent(stream_chunks=[
            {"type": "token", "content": "你好"},
            {"type": "token", "content": "世界"},
        ])
        mm = FakeMemoryManager(total_tokens=42)
        _override_agent(agent)
        _override_mm(mm)
        try:
            resp = authed_client.post(
                "/api/v1/agent/chat",
                json={"question": "打个招呼"},
            )
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            body = resp.text
            # start 帧携带 session_id
            assert "event: start\n" in body
            assert FAKE_SESSION_ID in body
            # token 帧带递增序号
            assert "event: token\n" in body
            assert "你好" in body and "世界" in body
            assert '"index":1' in body and '"index":2' in body
            # done 终态帧携带 session_id 与 token 统计
            assert "event: done\n" in body
            assert '"total_tokens":42' in body
            # agent 收到正确的 session_id 与 user_id
            assert agent.stream_calls == [("打个招呼", FAKE_SESSION_ID, FAKE_USER_ID)]
        finally:
            _clear_overrides(get_agent_with_memory, get_memory_manager)


# ------------------------------------------------------------------
# POST /api/v1/agent/approve
# ------------------------------------------------------------------

class TestAgentApprove:
    def test_unauthenticated_401(self, client):
        resp = client.post(
            "/api/v1/agent/approve",
            json={"session_id": "s-x", "action": "approve"},
        )
        assert resp.status_code == 401

    def test_invalid_action_422(self, authed_client):
        resp = authed_client.post(
            "/api/v1/agent/approve",
            json={"session_id": "s-x", "action": "bogus"},
        )
        assert resp.status_code == 422

    def test_approve_stream_success(self, authed_client):
        agent = FakeAgent(resume_chunks=[
            {"type": "token", "content": "已恢复"},
            {"type": "token", "content": "执行"},
        ])
        _override_agent(agent)
        try:
            resp = authed_client.post(
                "/api/v1/agent/approve",
                json={"session_id": "s-hitl", "action": "approve", "comment": "确认下载"},
            )
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            body = resp.text
            assert "event: start\n" in body
            assert "event: token\n" in body
            assert "已恢复" in body and "执行" in body
            assert "event: done\n" in body
            # approval_result 原样透传给 agent.resume_stream
            assert agent.resume_calls == [
                ("s-hitl", {"action": "approve", "comment": "确认下载"}, FAKE_USER_ID)
            ]
        finally:
            _clear_overrides(get_agent_with_memory)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
