"""Layer 2c: MCP 跨进程 Trace Context 传播（Agent 侧 — 发送端）。"""

from __future__ import annotations

from typing import Any


def inject_trace_context(tool_args: dict[str, Any]) -> dict[str, Any]:
    """将当前 OTel Trace Context 注入 MCP tool call arguments 的 _meta 字段。"""
    try:
        from opentelemetry.propagate import inject
        carrier: dict[str, str] = {}
        inject(carrier)
        if not carrier:
            return tool_args
        meta = tool_args.setdefault("_meta", {})
        if "traceparent" in carrier:
            meta["traceparent"] = carrier["traceparent"]
        if "tracestate" in carrier:
            meta["tracestate"] = carrier["tracestate"]
    except ImportError:
        pass
    return tool_args


def extract_trace_context(_meta: dict[str, Any] | None) -> Any | None:
    """从 MCP _meta 恢复 OTel Context（rag_mcp_server 侧使用，此处保留兼容）。"""
    if not _meta:
        return None
    try:
        from opentelemetry.propagate import extract
        carrier: dict[str, str] = {}
        if _meta.get("traceparent"):
            carrier["traceparent"] = _meta["traceparent"]
        if _meta.get("tracestate"):
            carrier["tracestate"] = _meta["tracestate"]
        if not carrier:
            return None
        return extract(carrier)
    except ImportError:
        return None
