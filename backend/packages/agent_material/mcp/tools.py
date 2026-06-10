"""Load MCP tools using langchain-mcp-adapters with stdio session pooling."""

from __future__ import annotations

from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.config import get_config

from agent_material.config.extensions_config import ExtensionsConfig
from agent_material.mcp.client import build_servers_config
from agent_material.mcp.oauth import build_oauth_tool_interceptor, get_initial_oauth_headers
from agent_material.mcp.session_pool import get_session_pool
from agent_material.reflection import resolve_variable
from agent_material.tools.sync import make_sync_tool_wrapper
from agent_material.tools.types import Runtime
from agent_material.observability.mcp_trace import inject_trace_context

logger = logging.getLogger(__name__)

# Reference held to prevent MultiServerMCPClient (and its internal temporary
# stdio sessions) from being garbage-collected, which would trigger async
# generator finalizer crashes due to anyio cancel-scope task-binding.
_client_ref: Any = None


def _extract_thread_id(runtime: Runtime | None) -> str:
    """Extract thread_id from the injected tool runtime or LangGraph config."""
    if runtime is not None:
        tid = runtime.context.get("thread_id") if runtime.context else None
        if tid is not None:
            return str(tid)
        config = runtime.config or {}
        tid = config.get("configurable", {}).get("thread_id")
        if tid is not None:
            return str(tid)

    try:
        tid = get_config().get("configurable", {}).get("thread_id")
        return str(tid) if tid is not None else "default"
    except RuntimeError:
        return "default"


def _convert_call_tool_result(call_tool_result: Any) -> Any:
    """Convert an MCP CallToolResult to the LangChain ``content_and_artifact`` format."""
    from langchain_core.messages import ToolMessage
    from langchain_core.messages.content import create_file_block, create_image_block, create_text_block
    from langchain_core.tools import ToolException
    from mcp.types import EmbeddedResource, ImageContent, ResourceLink, TextContent, TextResourceContents

    # Pass ToolMessage through directly (interceptor short-circuit).
    if isinstance(call_tool_result, ToolMessage):
        return call_tool_result, None

    # Pass LangGraph Command through directly when langgraph is installed.
    try:
        from langgraph.types import Command

        if isinstance(call_tool_result, Command):
            return call_tool_result, None
    except ImportError:
        pass

    # Convert MCP content blocks to LangChain content blocks.
    lc_content = []
    for item in call_tool_result.content:
        if isinstance(item, TextContent):
            lc_content.append(create_text_block(text=item.text))
        elif isinstance(item, ImageContent):
            lc_content.append(create_image_block(base64=item.data, mime_type=item.mimeType))
        elif isinstance(item, ResourceLink):
            mime = item.mimeType or None
            if mime and mime.startswith("image/"):
                lc_content.append(create_image_block(url=str(item.uri), mime_type=mime))
            else:
                lc_content.append(create_file_block(url=str(item.uri), mime_type=mime))
        elif isinstance(item, EmbeddedResource):
            from mcp.types import BlobResourceContents

            res = item.resource
            if isinstance(res, TextResourceContents):
                lc_content.append(create_text_block(text=res.text))
            elif isinstance(res, BlobResourceContents):
                mime = res.mimeType or None
                if mime and mime.startswith("image/"):
                    lc_content.append(create_image_block(base64=res.blob, mime_type=mime))
                else:
                    lc_content.append(create_file_block(base64=res.blob, mime_type=mime))
            else:
                lc_content.append(create_text_block(text=str(res)))
        else:
            lc_content.append(create_text_block(text=str(item)))

    if call_tool_result.isError:
        error_parts = [item["text"] for item in lc_content if isinstance(item, dict) and item.get("type") == "text"]
        raise ToolException("\n".join(error_parts) if error_parts else str(lc_content))

    artifact = None
    if call_tool_result.structuredContent is not None:
        artifact = {"structured_content": call_tool_result.structuredContent}

    return lc_content, artifact


def _make_session_pool_tool(
    tool: BaseTool,
    server_name: str,
    connection: dict[str, Any],
    tool_interceptors: list[Any] | None = None,
) -> BaseTool:
    """Wrap an MCP tool so it reuses a persistent session from the pool."""
    original_name = tool.name
    prefix = f"{server_name}_"
    if original_name.startswith(prefix):
        original_name = original_name[len(prefix) :]

    pool = get_session_pool()

    async def call_with_persistent_session(
        runtime: Runtime | None = None,
        **arguments: Any,
    ) -> Any:
        t0 = time.time()
        thread_id = _extract_thread_id(runtime)

        # ── 可观测性：注入 MCP Trace Context ──
        try:
            arguments = inject_trace_context(arguments)
        except Exception:
            pass

        try:
            from agent_material.observability.metrics import (
                mcp_tool_call_total,
                mcp_tool_call_duration,
            )
            _metrics_available = True
        except ImportError:
            _metrics_available = False

        session = await pool.get_session(server_name, thread_id, connection)

        if tool_interceptors:
            from langchain_mcp_adapters.interceptors import MCPToolCallRequest

            async def base_handler(request: MCPToolCallRequest) -> Any:
                return await session.call_tool(request.name, request.args)

            handler = base_handler
            for interceptor in reversed(tool_interceptors):
                outer = handler

                async def wrapped(req: Any, _i: Any = interceptor, _h: Any = outer) -> Any:
                    return await _i(req, _h)

                handler = wrapped

            request = MCPToolCallRequest(
                name=original_name,
                args=arguments,
                server_name=server_name,
                runtime=runtime,
            )
            call_tool_result = await handler(request)
        else:
            call_tool_result = await session.call_tool(original_name, arguments)

        content, artifact = _convert_call_tool_result(call_tool_result)
        # Convert multimodal content blocks to plain text for compatibility
        # with models that do not support list-type ToolMessage.content
        # (e.g. DeepSeek-V4-Flash).
        if isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "image":
                        text_parts.append("[image]")
                    elif block.get("type") == "file":
                        text_parts.append("[file]")
                    else:
                        text_parts.append(str(block))
                else:
                    text_parts.append(str(block))
            content = "\n".join(text_parts)

        # ── 上报 MCP 工具调用指标 ──
        if _metrics_available:
            elapsed = time.time() - t0
            status = "error" if call_tool_result.isError else "success"
            mcp_tool_call_total.labels(server=server_name, tool=original_name, status=status).inc()
            mcp_tool_call_duration.labels(server=server_name, tool=original_name).observe(elapsed)

        return content, artifact

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        coroutine=call_with_persistent_session,
        response_format="content_and_artifact",
        metadata=tool.metadata,
    )


async def get_mcp_tools() -> list[BaseTool]:
    """Get all tools from enabled MCP servers.

    Tools using stdio transport are wrapped with persistent-session logic so
    consecutive calls within the same thread reuse the same MCP session.
    HTTP/SSE tools are returned unwrapped to avoid cross-task TaskGroup
    cleanup errors.
    """
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        logger.warning("langchain-mcp-adapters not installed. Install it to enable MCP tools: pip install langchain-mcp-adapters")
        return []

    extensions_config = ExtensionsConfig.from_file()
    servers_config = build_servers_config(extensions_config)

    if not servers_config:
        logger.info("No enabled MCP servers configured")
        return []

    try:
        logger.info(f"Initializing MCP client with {len(servers_config)} server(s)")

        initial_oauth_headers = await get_initial_oauth_headers(extensions_config)
        for server_name, auth_header in initial_oauth_headers.items():
            if server_name not in servers_config:
                continue
            if servers_config[server_name].get("transport") in ("sse", "http"):
                existing_headers = dict(servers_config[server_name].get("headers", {}))
                existing_headers["Authorization"] = auth_header
                servers_config[server_name]["headers"] = existing_headers

        tool_interceptors: list[Any] = []
        oauth_interceptor = build_oauth_tool_interceptor(extensions_config)
        if oauth_interceptor is not None:
            tool_interceptors.append(oauth_interceptor)

        raw_interceptor_paths = (extensions_config.model_extra or {}).get("mcpInterceptors")
        if isinstance(raw_interceptor_paths, str):
            raw_interceptor_paths = [raw_interceptor_paths]
        elif not isinstance(raw_interceptor_paths, list):
            if raw_interceptor_paths is not None:
                logger.warning(f"mcpInterceptors must be a list of strings, got {type(raw_interceptor_paths).__name__}; skipping")
            raw_interceptor_paths = []

        for interceptor_path in raw_interceptor_paths:
            try:
                builder = resolve_variable(interceptor_path)
                interceptor = builder()
                if callable(interceptor):
                    tool_interceptors.append(interceptor)
                    logger.info(f"Loaded MCP interceptor: {interceptor_path}")
                elif interceptor is not None:
                    logger.warning(f"Builder {interceptor_path} returned non-callable {type(interceptor).__name__}; skipping")
            except Exception as e:
                logger.warning(f"Failed to load MCP interceptor {interceptor_path}: {e}", exc_info=True)

        global _client_ref
        client = MultiServerMCPClient(
            servers_config,
            tool_interceptors=tool_interceptors,
            tool_name_prefix=True,
        )
        _client_ref = client

        tools = await client.get_tools()
        logger.info(f"Successfully loaded {len(tools)} tool(s) from MCP servers")

        wrapped_tools: list[BaseTool] = []
        for tool in tools:
            tool_server: str | None = None
            for name in servers_config:
                if tool.name.startswith(f"{name}_"):
                    tool_server = name
                    break

            if tool_server is not None:
                transport = servers_config[tool_server].get("transport", "stdio")
                if transport == "stdio":
                    wrapped_tools.append(_make_session_pool_tool(tool, tool_server, servers_config[tool_server], tool_interceptors))
                else:
                    wrapped_tools.append(tool)
            else:
                wrapped_tools.append(tool)

        # Patch tools to support sync invocation
        for tool in wrapped_tools:
            if getattr(tool, "func", None) is None and getattr(tool, "coroutine", None) is not None:
                tool.func = make_sync_tool_wrapper(tool.coroutine, tool.name)

        return wrapped_tools

    except Exception as e:
        logger.error(f"Failed to load MCP tools: {e}", exc_info=True)
        return []
