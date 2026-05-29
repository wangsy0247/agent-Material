"""Tests for MCP OAuth token management."""

import pytest
from datetime import UTC, datetime, timedelta

from agent_material.config.extensions_config import ExtensionsConfig, McpOAuthConfig, McpServerConfig
from agent_material.mcp.oauth import OAuthTokenManager, _OAuthToken


class TestOAuthTokenManager:
    def test_has_oauth_servers(self):
        manager = OAuthTokenManager({
            "server1": McpOAuthConfig(enabled=True, token_url="http://test.com/token")
        })
        assert manager.has_oauth_servers() is True

    def test_no_oauth_servers(self):
        manager = OAuthTokenManager({})
        assert manager.has_oauth_servers() is False

    def test_is_expiring(self):
        oauth = McpOAuthConfig(enabled=True, token_url="http://test.com", refresh_skew_seconds=60)
        token = _OAuthToken(
            access_token="test",
            token_type="Bearer",
            expires_at=datetime.now(UTC) + timedelta(seconds=30),
        )
        assert OAuthTokenManager._is_expiring(token, oauth) is True

        token = _OAuthToken(
            access_token="test",
            token_type="Bearer",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
        assert OAuthTokenManager._is_expiring(token, oauth) is False

    def test_from_extensions_config(self):
        config = ExtensionsConfig(mcp_servers={
            "server1": McpServerConfig(
                enabled=True,
                type="http",
                url="http://test.com/mcp",
                oauth=McpOAuthConfig(
                    enabled=True,
                    token_url="http://test.com/token",
                    grant_type="client_credentials",
                    client_id="id",
                    client_secret="secret",
                ),
            ),
            "server2": McpServerConfig(
                enabled=True,
                type="stdio",
                command="python",
            ),
        })
        manager = OAuthTokenManager.from_extensions_config(config)
        assert manager.has_oauth_servers() is True
        assert manager.oauth_server_names() == ["server1"]

    def test_client_credentials_missing_secret(self):
        oauth = McpOAuthConfig(
            enabled=True,
            token_url="http://test.com",
            grant_type="client_credentials",
            client_id="id",
        )
        manager = OAuthTokenManager({"server1": oauth})
        import asyncio
        with pytest.raises(ValueError, match="client_credentials requires client_id and client_secret"):
            asyncio.run(manager._fetch_token(oauth))

    def test_unsupported_grant_type(self):
        # Pydantic Literal already validates grant_type, so we test _fetch_token directly
        # by creating a minimal mock object with an unsupported grant_type.
        class FakeOAuth:
            enabled = True
            token_url = "http://test.com"
            grant_type = "password"
            scope = None
            audience = None
            client_id = None
            client_secret = None
            refresh_token = None
            extra_token_params = {}
            token_field = "access_token"
            token_type_field = "token_type"
            expires_in_field = "expires_in"
            default_token_type = "Bearer"
            refresh_skew_seconds = 60

        fake_oauth = FakeOAuth()
        manager = OAuthTokenManager({"server1": fake_oauth})
        import asyncio
        with pytest.raises(ValueError, match="Unsupported OAuth grant type"):
            asyncio.run(manager._fetch_token(fake_oauth))
