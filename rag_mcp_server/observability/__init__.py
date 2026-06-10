"""RAG MCP Server 可观测性包 — 三层体系。

Layer 1 (structured_logger): 结构化 JSON 日志 + request_id/trace_id 注入
Layer 2 (otel_*):          OpenTelemetry 全链路追踪 + MCP 跨进程传播
Layer 3 (metrics):         Prometheus 指标 + /metrics 端点 + Grafana Dashboard

快速启动:
  from rag_mcp_server.observability import init_all
  init_all(service_name="rag_mcp_server")

最小化启动（开发环境，仅日志）:
  from rag_mcp_server.observability.structured_logger import get_logger
  logger = get_logger(__name__)
"""

from rag_mcp_server.observability.structured_logger import (
    get_logger,
    set_request_context,
    set_trace_context,
)
from rag_mcp_server.observability.otel_setup import (
    ensure_initialized,
    get_tracer,
    init_otel,
)
from rag_mcp_server.observability.mcp_trace import (
    extract_trace_context,
    inject_trace_context,
)
from rag_mcp_server.observability.metrics import (
    REGISTRY,
    StreamMetrics,
    get_metrics,
    track_llm_call,
    track_stage,
)


def init_all(
    service_name: str = "rag_mcp_server",
    enable_otel: bool = True,
    enable_metrics: bool = True,
) -> dict[str, bool]:
    """一键初始化三层可观测性。

    Args:
        service_name: 服务名
        enable_otel: 是否启用 OpenTelemetry 分布式追踪
        enable_metrics: 是否启用 Prometheus 指标

    Returns:
        {"logging": True, "tracing": True/False, "metrics": True/False}
    """
    status = {
        "logging": True,   # 结构化日志始终启用
        "tracing": False,
        "metrics": enable_metrics,
    }

    if enable_otel:
        status["tracing"] = ensure_initialized(service_name=service_name)

    return status
