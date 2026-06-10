"""Layer 2a: OpenTelemetry SDK 初始化 — TracerProvider + OTLP Exporter。

特性：
- OTLP gRPC / HTTP exporter（通过环境变量 OTEL_EXPORTER_OTLP_ENDPOINT 配置）
- BatchSpanProcessor（避免每次 create_span 都发网络请求）
- contextvars 自动跨 asyncio Task 传播
- 采样策略：全部采样（开发），生产建议通过 OTEL_TRACES_SAMPLER 切换

环境变量（可选）：
  OTEL_SERVICE_NAME=rag_mcp_server
  OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317     (gRPC)
  OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf             (HTTP)
  OTEL_TRACES_SAMPLER=parentbased_always_on             (默认全部采样)

使用方式:
  from rag_mcp_server.observability.otel_setup import init_otel, get_tracer
  init_otel()  # 程序启动时调用一次
  tracer = get_tracer("component_name")
"""

from __future__ import annotations

import os
from typing import Any


def init_otel(
    service_name: str | None = None,
    otlp_endpoint: str | None = None,
    sample_rate: float = 1.0,
) -> Any:
    """初始化 OpenTelemetry SDK。

    Args:
        service_name: 服务名，默认从 OTEL_SERVICE_NAME / "rag_mcp_server"
        otlp_endpoint: OTLP collector 地址，默认 http://localhost:4318/v1/traces
        sample_rate: 采样率 (0.0~1.0)

    Returns:
        TracerProvider 实例，可用于 shutdown()
    """
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes

    service = service_name or os.environ.get("OTEL_SERVICE_NAME", "rag_mcp_server")

    # ── 资源定义 ──
    resource = Resource.create({
        ResourceAttributes.SERVICE_NAME: service,
        ResourceAttributes.SERVICE_VERSION: "1.0.0",
    })

    # ── TracerProvider ──
    provider = TracerProvider(resource=resource)

    # ── OTLP Exporter（优先 HTTP protobuf，兼容性最好）──
    try:
        endpoint = otlp_endpoint or os.environ.get(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces"
        )
        protocol = os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")

        if protocol == "http/protobuf":
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            exporter = OTLPSpanExporter(endpoint=endpoint)
        else:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            exporter = OTLPSpanExporter(endpoint=endpoint)

        provider.add_span_processor(
            BatchSpanProcessor(
                exporter,
                max_queue_size=2048,
                max_export_batch_size=512,
                schedule_delay_millis=5000,
            )
        )
    except ImportError:
        # 降级：无 exporter 时只做进程内 Span 记录（开发调试用）
        import logging
        logging.getLogger(__name__).warning(
            "opentelemetry-exporter-otlp 未安装，追踪数据仅记录到 ConsoleSpanExporter"
        )
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter
        provider.add_span_processor(
            BatchSpanProcessor(ConsoleSpanExporter())
        )

    trace.set_tracer_provider(provider)
    return provider


def get_tracer(name: str = "rag_mcp_server") -> Any:
    """获取 OpenTelemetry Tracer 实例。"""
    from opentelemetry import trace
    return trace.get_tracer(name)


_initialized = False


def ensure_initialized(service_name: str | None = None) -> bool:
    """幂等初始化：只在首次调用时初始化 OTel SDK。"""
    global _initialized
    if _initialized:
        return True
    try:
        init_otel(service_name=service_name)
        _initialized = True
        return True
    except Exception:
        import logging
        logging.getLogger(__name__).warning(
            "OpenTelemetry SDK 初始化失败，追踪不可用"
        )
        return False
