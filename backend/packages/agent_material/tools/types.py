"""Tool runtime types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.runnables import RunnableConfig


@dataclass
class Runtime:
    """Runtime context injected into tool calls."""

    context: dict[str, Any] = field(default_factory=dict)
    config: RunnableConfig | None = None
