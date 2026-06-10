"""Layer 2a: OTel SDK 初始化。"""

from __future__ import annotations

import logging
import os
from typing import Any

_initialized = False
_logger = logging.getLogger(__name__)


def init_otel(service_name: str | None = None, otlp_endpoint: str | None = None) -> Any:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes

    service = service_name or os.environ.get("OTEL_SERVICE_NAME", "agent_material")
    resource = Resource.create({
        ResourceAttributes.SERVICE_NAME: service,
        ResourceAttributes.SERVICE_VERSION: "1.0.0",
    })
    provider = TracerProvider(resource=resource)

    try:
        endpoint = otlp_endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        exporter = OTLPSpanExporter(endpoint=endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter, max_queue_size=2048, max_export_batch_size=512, schedule_delay_millis=5000))
    except ImportError:
        _logger.warning("opentelemetry-exporter-otlp 未安装，使用 ConsoleSpanExporter")
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    return provider


def get_tracer(name: str = "agent_material") -> Any:
    from opentelemetry import trace
    return trace.get_tracer(name)


def ensure_initialized(service_name: str | None = None) -> bool:
    global _initialized
    if _initialized:
        return True
    try:
        init_otel(service_name=service_name)
        _initialized = True
        return True
    except Exception:
        _logger.warning("OTel SDK 初始化失败，追踪不可用")
        return False
