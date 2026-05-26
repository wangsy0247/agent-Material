"""
认证与多租户隔离模块测试

运行：pytest tests/test_auth.py -v
"""

import pytest
from fastapi.testclient import TestClient

from api_server import app


@pytest.fixture(scope="module")
def client():
    """FastAPI 测试客户端"""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_user(client):
    """创建一个测试用户并返回 token"""
    # 注册
    resp = client.post("/api/v1/auth/register", json={
        "username": "test_auth_user",
        "password": "testpass123",
        "email": "test@example.com",
    })
    # 如果用户已存在（重复运行测试），直接登录
    if resp.status_code == 409:
        resp = client.post("/api/v1/auth/login", json={
            "username": "test_auth_user",
            "password": "testpass123",
        })
    else:
        assert resp.status_code == 200
        resp = client.post("/api/v1/auth/login", json={
            "username": "test_auth_user",
            "password": "testpass123",
        })
    assert resp.status_code == 200
    data = resp.json()
    return {
        "token": data["access_token"],
        "user_id": data["user"]["user_id"],
        "username": data["user"]["username"],
    }


class TestAuthentication:
    """认证基础功能测试"""

    def test_health_public(self, client):
        """健康检查无需认证"""
        resp = client.get("/api/v1/system/health")
        assert resp.status_code == 200
        assert resp.json()["code"] == 200

    def test_protected_without_auth(self, client):
        """未认证访问受保护接口返回 401"""
        resp = client.get("/api/v1/memory/sessions")
        assert resp.status_code == 401
        assert "未提供认证凭证" in resp.json()["detail"]

    def test_register(self, client):
        """用户注册"""
        resp = client.post("/api/v1/auth/register", json={
            "username": "new_test_user",
            "password": "newpass123",
            "email": "new@example.com",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "new_test_user"
        assert "user_id" in data

    def test_register_duplicate_username(self, client):
        """重复用户名注册失败"""
        resp = client.post("/api/v1/auth/register", json={
            "username": "new_test_user",
            "password": "anypass123",
            "email": "other@example.com",
        })
        assert resp.status_code == 409
        assert "用户名已存在" in resp.json()["detail"]

    def test_login_success(self, client):
        """正确密码登录成功"""
        resp = client.post("/api/v1/auth/login", json={
            "username": "new_test_user",
            "password": "newpass123",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["user"]["username"] == "new_test_user"

    def test_login_wrong_password(self, client):
        """错误密码登录失败"""
        resp = client.post("/api/v1/auth/login", json={
            "username": "new_test_user",
            "password": "wrongpassword",
        })
        assert resp.status_code == 401
        assert "用户名或密码错误" in resp.json()["detail"]

    def test_me_endpoint(self, client, auth_user):
        """获取当前用户信息"""
        resp = client.get("/api/v1/auth/me", headers={
            "Authorization": f"Bearer {auth_user['token']}"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == auth_user["username"]
        assert data["user_id"] == auth_user["user_id"]

    def test_refresh_token(self, client, auth_user):
        """刷新 Token"""
        # 先获取 refresh_token
        login_resp = client.post("/api/v1/auth/login", json={
            "username": auth_user["username"],
            "password": "testpass123",
        })
        refresh_token = login_resp.json()["refresh_token"]

        resp = client.post("/api/v1/auth/refresh", json={
            "refresh_token": refresh_token,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data


class TestTenantIsolation:
    """多租户数据隔离测试"""

    @pytest.fixture
    def user_a(self, client):
        """创建用户 A"""
        username = "tenant_user_a"
        resp = client.post("/api/v1/auth/register", json={
            "username": username,
            "password": "pass123",
            "email": "a_tenant@example.com",
        })
        resp = client.post("/api/v1/auth/login", json={
            "username": username,
            "password": "pass123",
        })
        data = resp.json()
        return {"token": data["access_token"], "user_id": data["user"]["user_id"]}

    @pytest.fixture
    def user_b(self, client):
        """创建用户 B"""
        username = "tenant_user_b"
        resp = client.post("/api/v1/auth/register", json={
            "username": username,
            "password": "pass123",
            "email": "b_tenant@example.com",
        })
        resp = client.post("/api/v1/auth/login", json={
            "username": username,
            "password": "pass123",
        })
        data = resp.json()
        return {"token": data["access_token"], "user_id": data["user"]["user_id"]}

    def test_session_isolation(self, client, user_a, user_b):
        """会话隔离：用户只能看到自己的会话"""
        # 用户 A 创建会话
        resp = client.post("/api/v1/memory/sessions",
            headers={"Authorization": f"Bearer {user_a['token']}"},
            json={"title": "A的私密会话", "source": "test"})
        assert resp.status_code == 200
        session_a = resp.json()["data"]["session_id"]

        # 用户 B 创建会话
        resp = client.post("/api/v1/memory/sessions",
            headers={"Authorization": f"Bearer {user_b['token']}"},
            json={"title": "B的私密会话", "source": "test"})
        assert resp.status_code == 200
        session_b = resp.json()["data"]["session_id"]

        # 用户 A 只能看到 A 的会话
        resp = client.get("/api/v1/memory/sessions",
            headers={"Authorization": f"Bearer {user_a['token']}"})
        sessions = resp.json()["data"]
        titles = [s["title"] for s in sessions]
        assert "A的私密会话" in titles
        assert "B的私密会话" not in titles

        # 用户 B 只能看到 B 的会话
        resp = client.get("/api/v1/memory/sessions",
            headers={"Authorization": f"Bearer {user_b['token']}"})
        sessions = resp.json()["data"]
        titles = [s["title"] for s in sessions]
        assert "B的私密会话" in titles
        assert "A的私密会话" not in titles

    def test_cross_user_access_denied(self, client, user_a, user_b):
        """越权访问：用户 B 无法操作用户 A 的会话"""
        # 用户 A 创建会话
        resp = client.post("/api/v1/memory/sessions",
            headers={"Authorization": f"Bearer {user_a['token']}"},
            json={"title": "A的会话", "source": "test"})
        session_a = resp.json()["data"]["session_id"]

        # 用户 B 尝试加载 A 的会话 → 404
        resp = client.post(f"/api/v1/memory/sessions/{session_a}/load",
            headers={"Authorization": f"Bearer {user_b['token']}"})
        assert resp.status_code == 404

        # 用户 B 尝试删除 A 的会话 → 404
        resp = client.delete(f"/api/v1/memory/sessions/{session_a}",
            headers={"Authorization": f"Bearer {user_b['token']}"})
        assert resp.status_code == 404

        # 用户 A 可以正常访问自己的会话
        resp = client.post(f"/api/v1/memory/sessions/{session_a}/load",
            headers={"Authorization": f"Bearer {user_a['token']}"})
        assert resp.status_code == 200

    def test_rag_requires_auth(self, client):
        """RAG 接口需要认证"""
        resp = client.post("/api/v1/rag/retrieve", json={
            "question": "测试",
            "top_k": 3,
        })
        assert resp.status_code == 401

    def test_agent_chat_requires_auth(self, client):
        """Agent 对话接口需要认证"""
        resp = client.post("/api/v1/agent/chat", json={
            "question": "测试",
            "session_id": "default",
            "stream": False,
        })
        assert resp.status_code == 401
