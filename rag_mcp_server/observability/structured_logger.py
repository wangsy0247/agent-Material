"""Layer 1: 结构化 JSON 日志 — trace_id / request_id 注入 + 上下文传播。

Context Variables（自动跨 asyncio Task 传播，跨线程需手动复制）：
  - _trace_context:   {trace_id, span_id}      来自 OpenTelemetry
  - _request_context: {request_id, session_id, user_id}   来自业务入口

使用方式:
  from rag_mcp_server.observability.structured_logger import get_logger
  logger = get_logger(__name__)
  logger.info("rerank completed", extra={"input_docs": 10, "duration_ms": 85})
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import uuid
from datetime import datetime, timezone

# ── Context Variables ────────────────────────────────────────────
_trace_context: contextvars.ContextVar[dict[str, str]] = contextvars.ContextVar(
    "trace_context", default={}
)
_request_context: contextvars.ContextVar[dict[str, str]] = contextvars.ContextVar(
    "request_context", default={}
)


def set_trace_context(
    trace_id: str | None = None, span_id: str | None = None
) -> dict[str, str]:
    ctx = {
        "trace_id": trace_id or uuid.uuid4().hex[:16],
        "span_id": span_id or uuid.uuid4().hex[:8],
    }
    _trace_context.set(ctx)
    return ctx


def set_request_context(
    request_id: str | None = None,
    session_id: str = "",
    user_id: str = "",
) -> dict[str, str]:
    ctx = {
        "request_id": request_id or uuid.uuid4().hex[:12],
        "session_id": session_id,
        "user_id": user_id,
    }
    _request_context.set(ctx)
    return ctx


def get_trace_context() -> dict[str, str]:
    return _trace_context.get()


def get_request_context() -> dict[str, str]:
    return _request_context.get()


# ── JSON Formatter ───────────────────────────────────────────────


class JsonFormatter(logging.Formatter):
    """将日志记录格式化为 JSON，自动注入 trace / request context。"""

    RESERVED = {
        "name", "levelname", "levelno", "pathname", "filename", "module",
        "lineno", "funcName", "created", "asctime", "msecs", "relativeCreated",
        "thread", "threadName", "process", "message", "msg", "args", "exc_info",
        "exc_text", "stack_info",
        # 排除 context 内部属性
        "trace_context", "request_context",
    }

    def format(self, record: logging.LogRecord) -> str:
        base: dict = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "service": os.environ.get("OTEL_SERVICE_NAME", "rag_mcp_server"),
            "component": record.name,
            "file": f"{record.filename}:{record.lineno}",
            "message": record.getMessage(),
        }
        # 注入 trace / request context
        trace_ctx = _trace_context.get()
        if trace_ctx:
            base["trace_id"] = trace_ctx.get("trace_id", "")
            base["span_id"] = trace_ctx.get("span_id", "")
        req_ctx = _request_context.get()
        if req_ctx:
            base["request_id"] = req_ctx.get("request_id", "")
            base["session_id"] = req_ctx.get("session_id", "")
            base["user_id"] = req_ctx.get("user_id", "")

        # 合并业务 extra 字段
        for key, val in vars(record).items():
            if key not in self.RESERVED and not key.startswith("_"):
                base[key] = val

        if record.exc_info and record.exc_info[1]:
            base["exception"] = str(record.exc_info[1])

        return json.dumps(base, ensure_ascii=False, default=str)


# ── Logger 工厂 ──────────────────────────────────────────────────

_LOG_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "logs",
)
os.makedirs(_LOG_ROOT, exist_ok=True)

_registry: dict[str, logging.Logger] = {}


def get_logger(name: str = "rag_mcp_server") -> logging.Logger:
    """获取结构化 Logger（按 name 缓存 + 文件 + 控制台双输出）。

    文件输出: logs/<name>.jsonl   (JSON 格式，每行一条记录)
    控制台输出: stdout            (JSON 格式，便于 grep / jq 解析)
    """
    if name in _registry:
        return _registry[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # 防止重复输出

    # 控制台 handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(JsonFormatter())

    # 文件 handler (JSONL)
    safe_name = name.replace(".", "_")
    log_file = os.path.join(_LOG_ROOT, f"{safe_name}.jsonl")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JsonFormatter())

    # 只在没有 handler 时才添加
    if not logger.handlers:
        logger.addHandler(console)
        logger.addHandler(file_handler)

    _registry[name] = logger
    return logger
