"""MCP 路由安全测试（Phase 4）。

覆盖：/api/mcp/config（GET/PUT）与 /api/mcp/status 的鉴权要求，
以及 /api/mcp/status 的响应结构。
"""

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# backend/ 与 backend/packages 加入路径（pyproject 仅配置了 packages）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.gateway.app import app  # noqa: E402
from app.gateway.auth.dependencies import get_current_user  # noqa: E402
from app.gateway.auth.schemas import UserInfo  # noqa: E402


def _fake_user() -> UserInfo:
    return UserInfo(user_id="test-u", username="tester", created_at=time.time())


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


class TestMcpAuth:
    """无 token 一律 401（修复前这两个端点无任何鉴权）"""

    def test_get_config_unauthenticated_401(self, client):
        resp = client.get("/api/mcp/config")
        assert resp.status_code == 401

    def test_put_config_unauthenticated_401(self, client):
        resp = client.put("/api/mcp/config", json={"mcp_servers": {}})
        assert resp.status_code == 401

    def test_status_unauthenticated_401(self, client):
        resp = client.get("/api/mcp/status")
        assert resp.status_code == 401


class TestMcpEndpoints:
    def test_get_config_authenticated_200(self, authed_client):
        resp = authed_client.get("/api/mcp/config")
        assert resp.status_code == 200
        assert "mcp_servers" in resp.json()

    def test_status_response_shape(self, authed_client):
        resp = authed_client.get("/api/mcp/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 200
        assert isinstance(body["data"], dict)
        for name, info in body["data"].items():
            assert set(info) == {"enabled", "type", "transport_configured"}, name
            assert isinstance(info["enabled"], bool)
            assert isinstance(info["transport_configured"], bool)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
