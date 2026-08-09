"""Agent Material 可观测性包 — 三层体系。

Layer 1: structured_logger  — JSON 结构化日志 + request_id/trace_id 注入
Layer 2: otel_setup/otel_callback/mcp_trace — OTel 全链路追踪 + MCP 跨进程传播
Layer 3: metrics — Prometheus 指标 + /metrics + Grafana Dashboard

快速启动:
  from agent_material.observability import init_all
  init_all(service_name="agent_material")
"""

from agent_material.observability.structured_logger import (
    get_logger,
    set_request_context,
    set_trace_context,
)
from agent_material.observability.otel_setup import (
    ensure_initialized,
    get_tracer,
    init_otel,
)
from agent_material.observability.mcp_trace import (
    extract_trace_context,
    inject_trace_context,
)
from agent_material.observability.metrics import (
    REGISTRY,
    StreamMetrics,
    get_metrics,
)


def init_all(
    service_name: str = "agent_material",
    enable_otel: bool = True,
    enable_metrics: bool = True,
) -> dict[str, bool]:
    status = {"logging": True, "tracing": False, "metrics": enable_metrics}
    if enable_otel:
        status["tracing"] = ensure_initialized(service_name=service_name)
    return status
