"""Tests for MCP configuration secret masking and merging."""

import pytest
from fastapi import HTTPException

from backend.app.gateway.routers.mcp import (
    _mask_server_config,
    _merge_preserving_secrets,
    McpServerConfigResponse,
    McpOAuthConfigResponse,
)


class TestMaskServerConfig:
    def test_masks_env_values(self):
        server = McpServerConfigResponse(
            enabled=True,
            type="stdio",
            env={"GITHUB_TOKEN": "secret123", "NORMAL": "value"},
        )
        masked = _mask_server_config(server)
        assert masked.env == {"GITHUB_TOKEN": "***", "NORMAL": "***"}

    def test_masks_oauth_secrets(self):
        server = McpServerConfigResponse(
            enabled=True,
            type="http",
            oauth=McpOAuthConfigResponse(
                enabled=True,
                token_url="http://test.com",
                client_secret="my_secret",
                refresh_token="my_refresh",
            ),
        )
        masked = _mask_server_config(server)
        assert masked.oauth.client_secret is None
        assert masked.oauth.refresh_token is None


class TestMergePreservingSecrets:
    def test_preserves_existing_env(self):
        incoming = McpServerConfigResponse(
            enabled=True,
            type="stdio",
            env={"GITHUB_TOKEN": "***"},
        )
        existing = McpServerConfigResponse(
            enabled=True,
            type="stdio",
            env={"GITHUB_TOKEN": "secret123"},
        )
        merged = _merge_preserving_secrets(incoming, existing)
        assert merged.env["GITHUB_TOKEN"] == "secret123"

    def test_rejects_new_masked_env(self):
        incoming = McpServerConfigResponse(
            enabled=True,
            type="stdio",
            env={"NEW_KEY": "***"},
        )
        existing = McpServerConfigResponse(
            enabled=True,
            type="stdio",
            env={},
        )
        with pytest.raises(HTTPException):
            _merge_preserving_secrets(incoming, existing)

    def test_explicitly_clears_secret(self):
        incoming = McpServerConfigResponse(
            enabled=True,
            type="http",
            oauth=McpOAuthConfigResponse(
                enabled=True,
                token_url="http://test.com",
                client_secret="",
            ),
        )
        existing = McpServerConfigResponse(
            enabled=True,
            type="http",
            oauth=McpOAuthConfigResponse(
                enabled=True,
                token_url="http://test.com",
                client_secret="secret",
            ),
        )
        merged = _merge_preserving_secrets(incoming, existing)
        assert merged.oauth.client_secret is None
