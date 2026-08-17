import json
import logging
import os
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from agent_material.config.extensions_config import ExtensionsConfig, get_extensions_config, reload_extensions_config
from ..auth.dependencies import get_current_admin, get_current_user
from ..auth.schemas import UserInfo
from ..schemas import MCPStatusResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["mcp"])

_ALLOW_RUNTIME_CONFIG = os.environ.get(
    "AGENT_ALLOW_RUNTIME_MCP_CONFIG", "false"
).lower() in {"1", "true", "yes"}


class McpOAuthConfigResponse(BaseModel):
    """OAuth configuration for an MCP server."""

    enabled: bool = Field(default=True, description="Whether OAuth token injection is enabled")
    token_url: str = Field(default="", description="OAuth token endpoint URL")
    grant_type: Literal["client_credentials", "refresh_token"] = Field(default="client_credentials", description="OAuth grant type")
    client_id: str | None = Field(default=None, description="OAuth client ID")
    client_secret: str | None = Field(default=None, description="OAuth client secret")
    refresh_token: str | None = Field(default=None, description="OAuth refresh token")
    scope: str | None = Field(default=None, description="OAuth scope")
    audience: str | None = Field(default=None, description="OAuth audience")
    token_field: str = Field(default="access_token", description="Token response field containing access token")
    token_type_field: str = Field(default="token_type", description="Token response field containing token type")
    expires_in_field: str = Field(default="expires_in", description="Token response field containing expires-in seconds")
    default_token_type: str = Field(default="Bearer", description="Default token type when response omits token_type")
    refresh_skew_seconds: int = Field(default=60, description="Refresh this many seconds before expiry")
    extra_token_params: dict[str, str] = Field(default_factory=dict, description="Additional form params sent to token endpoint")


class McpServerConfigResponse(BaseModel):
    """Response model for MCP server configuration."""

    enabled: bool = Field(default=True, description="Whether this MCP server is enabled")
    type: str = Field(default="stdio", description="Transport type: 'stdio', 'sse', or 'http'")
    command: str | None = Field(default=None, description="Command to execute to start the MCP server (for stdio type)")
    args: list[str] = Field(default_factory=list, description="Arguments to pass to the command (for stdio type)")
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables for the MCP server")
    url: str | None = Field(default=None, description="URL of the MCP server (for sse or http type)")
    headers: dict[str, str] = Field(default_factory=dict, description="HTTP headers to send (for sse or http type)")
    oauth: McpOAuthConfigResponse | None = Field(default=None, description="OAuth configuration for MCP HTTP/SSE servers")
    description: str = Field(default="", description="Human-readable description of what this MCP server provides")


class McpConfigResponse(BaseModel):
    """Response model for MCP configuration."""

    mcp_servers: dict[str, McpServerConfigResponse] = Field(
        default_factory=dict,
        description="Map of MCP server name to configuration",
    )


class McpConfigUpdateRequest(BaseModel):
    """Request model for updating MCP configuration."""

    mcp_servers: dict[str, McpServerConfigResponse] = Field(
        ...,
        description="Map of MCP server name to configuration",
    )


_MASKED_VALUE = "***"


def _mask_server_config(server: McpServerConfigResponse) -> McpServerConfigResponse:
    """Return a copy of server config with sensitive fields masked."""
    masked_env = {k: _MASKED_VALUE for k in server.env}
    masked_headers = {k: _MASKED_VALUE for k in server.headers}
    masked_oauth = None
    if server.oauth is not None:
        masked_oauth = server.oauth.model_copy(
            update={
                "client_secret": None,
                "refresh_token": None,
            }
        )
    return server.model_copy(
        update={
            "env": masked_env,
            "headers": masked_headers,
            "oauth": masked_oauth,
        }
    )


def _merge_preserving_secrets(
    incoming: McpServerConfigResponse,
    existing: McpServerConfigResponse,
) -> McpServerConfigResponse:
    """Merge incoming config with existing, preserving secrets masked by GET."""
    merged_env = {}
    for k, v in incoming.env.items():
        if v == _MASKED_VALUE:
            if k in existing.env:
                merged_env[k] = existing.env[k]
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot set env key '{k}' to masked value '***'; provide a real value.",
                )
        else:
            merged_env[k] = v

    merged_headers = {}
    for k, v in incoming.headers.items():
        if v == _MASKED_VALUE:
            if k in existing.headers:
                merged_headers[k] = existing.headers[k]
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot set header '{k}' to masked value '***'; provide a real value.",
                )
        else:
            merged_headers[k] = v

    merged_oauth = incoming.oauth
    if incoming.oauth is not None and existing.oauth is not None:
        merged_client_secret = existing.oauth.client_secret if incoming.oauth.client_secret is None else (None if incoming.oauth.client_secret == "" else incoming.oauth.client_secret)
        merged_refresh_token = existing.oauth.refresh_token if incoming.oauth.refresh_token is None else (None if incoming.oauth.refresh_token == "" else incoming.oauth.refresh_token)
        merged_oauth = incoming.oauth.model_copy(
            update={
                "client_secret": merged_client_secret,
                "refresh_token": merged_refresh_token,
            }
        )
    return incoming.model_copy(
        update={
            "env": merged_env,
            "headers": merged_headers,
            "oauth": merged_oauth,
        }
    )


@router.get(
    "/mcp/config",
    response_model=McpConfigResponse,
    summary="Get MCP Configuration",
    description="Retrieve the current Model Context Protocol (MCP) server configurations.",
)
async def get_mcp_configuration(
    current_user: UserInfo = Depends(get_current_admin),
) -> McpConfigResponse:
    """Get the current MCP configuration."""
    config = get_extensions_config()

    servers = {name: _mask_server_config(McpServerConfigResponse(**server.model_dump())) for name, server in config.mcp_servers.items()}
    return McpConfigResponse(mcp_servers=servers)


@router.get(
    "/mcp/status",
    response_model=MCPStatusResponse,
    summary="MCP 连接状态",
    description="返回各 MCP server 的状态信息（静态配置版）。",
)
async def mcp_status(
    current_user: UserInfo = Depends(get_current_user),
) -> MCPStatusResponse:
    """返回各 MCP server 状态：enabled / 传输类型 / 传输配置是否完整。

    注：当前为静态配置信息版；实时连接健康检查（探活）需要 MCP 客户端
    支持，后续补充。
    """
    config = get_extensions_config()
    data = {}
    for name, server in config.mcp_servers.items():
        if server.type == "stdio":
            transport_configured = bool(server.command)
        else:  # sse / http
            transport_configured = bool(server.url)
        data[name] = {
            "enabled": server.enabled,
            "type": server.type,
            "transport_configured": transport_configured,
        }
    return MCPStatusResponse(data=data)


@router.put(
    "/mcp/config",
    response_model=McpConfigResponse,
    summary="Update MCP Configuration",
    description="Update Model Context Protocol (MCP) server configurations and save to file.",
)
async def update_mcp_configuration(
    request: McpConfigUpdateRequest,
    current_user: UserInfo = Depends(get_current_admin),
) -> McpConfigResponse:
    """Update the MCP configuration.

    This will:
    1. Save the new configuration to the extensions_config.json file
    2. Reload the configuration cache
    3. Reset MCP tools cache to trigger reinitialization
    """
    if not _ALLOW_RUNTIME_CONFIG:
        raise HTTPException(
            status_code=403,
            detail=(
                "运行时 MCP 配置修改默认关闭；请修改 extensions_config.json "
                "并重启服务"
            ),
        )

    try:
        config_path = ExtensionsConfig.resolve_config_path()

        if config_path is None:
            config_path = Path.cwd().parent / "extensions_config.json"
            logger.info(f"No existing extensions config found. Creating new config at: {config_path}")

        current_config = get_extensions_config()

        raw_servers: dict[str, dict] = {}
        raw_other_keys: dict = {}
        if config_path is not None and config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                raw_data = json.load(f)
            raw_servers = raw_data.get("mcpServers", {})
            for key, value in raw_data.items():
                if key not in ("mcpServers", "skills"):
                    raw_other_keys[key] = value

        merged_servers: dict[str, McpServerConfigResponse] = {}
        for name, incoming in request.mcp_servers.items():
            raw_server = raw_servers.get(name)
            if raw_server is not None:
                merged_servers[name] = _merge_preserving_secrets(
                    incoming,
                    McpServerConfigResponse(**raw_server),
                )
            else:
                merged_servers[name] = incoming

        config_data = dict(raw_other_keys)
        config_data["mcpServers"] = {name: server.model_dump() for name, server in merged_servers.items()}
        config_data["skills"] = {name: {"enabled": skill.enabled} for name, skill in current_config.skills.items()}

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"MCP configuration updated and saved to: {config_path}")

        reloaded_config = reload_extensions_config()
        servers = {name: _mask_server_config(McpServerConfigResponse(**server.model_dump())) for name, server in reloaded_config.mcp_servers.items()}
        return McpConfigResponse(mcp_servers=servers)

    except Exception as e:
        logger.error(f"Failed to update MCP configuration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update MCP configuration")
