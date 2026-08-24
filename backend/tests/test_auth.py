"""认证模块测试。

覆盖：
- 密码哈希与校验（bcrypt）
- JWT 签发/解析（含过期、篡改、类型不符）
- UserStore（SQLite，落临时目录，绝不触碰 logs/auth_users.db）
- /api/v1/auth/register、/login、/refresh、/me 四个端点

隔离方式：monkeypatch 替换 app.gateway.auth.auth._user_store 单例，
使其指向 tmp_path 下的临时 UserStore；路由与依赖均通过 get_user_store()
读取该全局变量，因此一处替换即可覆盖所有入口。
"""

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
import jwt

# backend/ 与 backend/packages 加入路径（pyproject 仅配置了 packages）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.gateway.app import app  # noqa: E402
from app.gateway.auth import auth as auth_mod  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def temp_store(tmp_path, monkeypatch):
    """临时 UserStore：替换全局单例，测试结束自动恢复原值。"""
    store = auth_mod.UserStore(db_path=str(tmp_path / "users.db"))
    monkeypatch.setattr(auth_mod, "_user_store", store)
    return store


def _expired_token(token_type: str = "access") -> str:
    payload = {
        "sub": "uid-expired",
        "username": "ghost",
        "type": token_type,
        "exp": time.time() - 3600,
    }
    return jwt.encode(payload, auth_mod.SECRET_KEY, algorithm=auth_mod.ALGORITHM)


# ---------- 纯逻辑：密码与 JWT ----------


class TestPasswordHashing:
    def test_hash_and_verify_roundtrip(self):
        hashed = auth_mod.get_password_hash("secret123")
        assert hashed != "secret123"
        assert auth_mod.verify_password("secret123", hashed) is True

    def test_verify_wrong_password(self):
        hashed = auth_mod.get_password_hash("secret123")
        assert auth_mod.verify_password("wrong-pass", hashed) is False


class TestJwtTokens:
    def test_weak_or_placeholder_secret_rejected(self, monkeypatch):
        monkeypatch.setattr(auth_mod, "SECRET_KEY", "change-me-in-production")
        with pytest.raises(RuntimeError, match="AUTH_SECRET_KEY"):
            auth_mod.validate_auth_configuration()

    def test_strong_secret_accepted(self, monkeypatch):
        secret = "a" * 48
        monkeypatch.setattr(auth_mod, "SECRET_KEY", secret)
        assert auth_mod.validate_auth_configuration() == secret

    def test_access_token_decodes_user_id(self):
        token = auth_mod.create_access_token("uid-1", "alice")
        data = auth_mod.decode_token(token)
        assert data is not None
        assert data.sub == "uid-1"
        assert data.username == "alice"
        assert data.type == "access"

    def test_refresh_token_type(self):
        token = auth_mod.create_refresh_token("uid-2", "bob")
        data = auth_mod.decode_token(token)
        assert data is not None
        assert data.type == "refresh"

    def test_expired_token_rejected(self):
        assert auth_mod.decode_token(_expired_token()) is None

    def test_tampered_signature_rejected(self):
        token = jwt.encode(
            {"sub": "uid-3", "username": "eve", "type": "access",
             "exp": time.time() + 600},
            "attacker-controlled-key-that-is-long-enough",
            algorithm=auth_mod.ALGORITHM,
        )
        assert auth_mod.decode_token(token) is None

    def test_garbage_token_rejected(self):
        assert auth_mod.decode_token("not-a-jwt") is None

    def test_missing_type_claim_rejected(self):
        token = jwt.encode(
            {"sub": "uid-4", "username": "noType", "exp": time.time() + 600},
            auth_mod.SECRET_KEY,
            algorithm=auth_mod.ALGORITHM,
        )
        assert auth_mod.decode_token(token) is None


# ---------- UserStore（临时 SQLite） ----------


class TestUserStore:
    def test_create_and_get_user(self, tmp_path):
        store = auth_mod.UserStore(db_path=str(tmp_path / "s.db"))
        user = store.create_user("alice", "pass123", "alice@example.com")
        assert user is not None
        assert user.username == "alice"
        assert store.get_user_by_id(user.user_id).username == "alice"

    def test_duplicate_username_returns_none(self, tmp_path):
        store = auth_mod.UserStore(db_path=str(tmp_path / "s.db"))
        assert store.create_user("alice", "pass123") is not None
        assert store.create_user("alice", "other-pass") is None

    def test_verify_user(self, tmp_path):
        store = auth_mod.UserStore(db_path=str(tmp_path / "s.db"))
        store.create_user("alice", "pass123")
        assert store.verify_user("alice", "pass123") is not None
        assert store.verify_user("alice", "bad-pass") is None
        assert store.verify_user("nobody", "pass123") is None


# ---------- HTTP 端点 ----------


class TestRegisterEndpoint:
    def test_register_success(self, client, temp_store):
        resp = client.post(
            "/api/v1/auth/register",
            json={"username": "alice", "password": "pass123",
                  "email": "alice@example.com"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["username"] == "alice"
        assert body["user_id"]
        assert body["is_admin"] is False
        assert "password" not in body
        assert "hashed_password" not in body

    def test_register_duplicate_username_409(self, client, temp_store):
        payload = {"username": "alice", "password": "pass123"}
        assert client.post("/api/v1/auth/register", json=payload).status_code == 200
        resp = client.post("/api/v1/auth/register", json=payload)
        assert resp.status_code == 409

    def test_register_short_password_422(self, client, temp_store):
        resp = client.post(
            "/api/v1/auth/register",
            json={"username": "alice", "password": "123"},
        )
        assert resp.status_code == 422


class TestLoginEndpoint:
    def test_login_success_returns_tokens(self, client, temp_store):
        temp_store.create_user("alice", "pass123")
        resp = client.post(
            "/api/v1/auth/login",
            json={"username": "alice", "password": "pass123"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["token_type"] == "bearer"
        assert body["expires_in"] == 3600
        assert body["user"]["username"] == "alice"
        # 两个 token 各自可解且类型正确
        assert auth_mod.decode_token(body["access_token"]).type == "access"
        assert auth_mod.decode_token(body["refresh_token"]).type == "refresh"

    def test_login_wrong_password_401(self, client, temp_store):
        temp_store.create_user("alice", "pass123")
        resp = client.post(
            "/api/v1/auth/login",
            json={"username": "alice", "password": "bad-pass"},
        )
        assert resp.status_code == 401

    def test_login_unknown_user_401(self, client, temp_store):
        resp = client.post(
            "/api/v1/auth/login",
            json={"username": "nobody", "password": "pass123"},
        )
        assert resp.status_code == 401


class TestMeEndpoint:
    def test_me_without_token_401(self, client, temp_store):
        resp = client.get("/api/v1/auth/me")
        assert resp.status_code == 401

    def test_me_with_valid_token(self, client, temp_store):
        user = temp_store.create_user("alice", "pass123")
        token = auth_mod.create_access_token(user.user_id, user.username)
        resp = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["user_id"] == user.user_id
        assert body["username"] == "alice"

    def test_me_with_expired_token_401(self, client, temp_store):
        resp = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {_expired_token()}"},
        )
        assert resp.status_code == 401

    def test_me_with_refresh_token_401(self, client, temp_store):
        user = temp_store.create_user("alice", "pass123")
        refresh = auth_mod.create_refresh_token(user.user_id, user.username)
        resp = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {refresh}"},
        )
        assert resp.status_code == 401


class TestRefreshEndpoint:
    def test_refresh_success(self, client, temp_store):
        user = temp_store.create_user("alice", "pass123")
        refresh = auth_mod.create_refresh_token(user.user_id, user.username)
        resp = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert auth_mod.decode_token(body["access_token"]).sub == user.user_id
        assert auth_mod.decode_token(body["refresh_token"]).type == "refresh"

    def test_refresh_invalid_token_401(self, client, temp_store):
        resp = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": "not-a-jwt"},
        )
        assert resp.status_code == 401

    def test_refresh_with_access_token_401(self, client, temp_store):
        user = temp_store.create_user("alice", "pass123")
        access = auth_mod.create_access_token(user.user_id, user.username)
        resp = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": access},
        )
        assert resp.status_code == 401

    def test_refresh_unknown_user_401(self, client, temp_store):
        refresh = auth_mod.create_refresh_token("uid-ghost", "ghost")
        resp = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh},
        )
        assert resp.status_code == 401


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
