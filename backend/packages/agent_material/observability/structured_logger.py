"""Layer 1: JSON 结构化日志 — 适用于 agent_material / FastAPI gateway。"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import uuid
from datetime import datetime, timezone

_trace_context: contextvars.ContextVar[dict[str, str]] = contextvars.ContextVar(
    "trace_context", default={}
)
_request_context: contextvars.ContextVar[dict[str, str]] = contextvars.ContextVar(
    "request_context", default={}
)


def set_trace_context(trace_id: str | None = None, span_id: str | None = None) -> dict[str, str]:
    ctx = {"trace_id": trace_id or uuid.uuid4().hex[:16], "span_id": span_id or uuid.uuid4().hex[:8]}
    _trace_context.set(ctx)
    return ctx


def set_request_context(request_id: str | None = None, session_id: str = "", user_id: str = "") -> dict[str, str]:
    ctx = {"request_id": request_id or uuid.uuid4().hex[:12], "session_id": session_id, "user_id": user_id}
    _request_context.set(ctx)
    return ctx


def get_trace_context() -> dict[str, str]:
    return _trace_context.get()


def get_request_context() -> dict[str, str]:
    return _request_context.get()


class JsonFormatter(logging.Formatter):
    RESERVED = {
        "name", "levelname", "levelno", "pathname", "filename", "module",
        "lineno", "funcName", "created", "asctime", "msecs", "relativeCreated",
        "thread", "threadName", "process", "message", "msg", "args", "exc_info",
        "exc_text", "stack_info",
    }

    def format(self, record: logging.LogRecord) -> str:
        base: dict = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "service": os.environ.get("OTEL_SERVICE_NAME", "agent_material"),
            "component": record.name,
            "file": f"{record.filename}:{record.lineno}",
            "message": record.getMessage(),
        }
        trace_ctx = _trace_context.get()
        if trace_ctx:
            base["trace_id"] = trace_ctx.get("trace_id", "")
            base["span_id"] = trace_ctx.get("span_id", "")
        req_ctx = _request_context.get()
        if req_ctx:
            base["request_id"] = req_ctx.get("request_id", "")
            base["session_id"] = req_ctx.get("session_id", "")
            base["user_id"] = req_ctx.get("user_id", "")
        for key, val in vars(record).items():
            if key not in self.RESERVED and not key.startswith("_"):
                base[key] = val
        if record.exc_info and record.exc_info[1]:
            base["exception"] = str(record.exc_info[1])
        return json.dumps(base, ensure_ascii=False, default=str)


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_LOG_ROOT = os.path.join(_PROJECT_ROOT, "logs")
os.makedirs(_LOG_ROOT, exist_ok=True)
_registry: dict[str, logging.Logger] = {}


def get_logger(name: str = "agent_material") -> logging.Logger:
    if name in _registry:
        return _registry[name]
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if not logger.handlers:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(JsonFormatter())
        logger.addHandler(console)
        safe_name = name.replace(".", "_")
        log_file = os.path.join(_LOG_ROOT, f"{safe_name}.jsonl")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JsonFormatter())
        logger.addHandler(file_handler)
    _registry[name] = logger
    return logger
