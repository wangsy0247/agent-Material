"""
Agent 中间件 - 工具监控与模型调用前日志

注意：原 langchain.agents.middleware 中的 wrap_tool_call / before_model
在 LangGraph 架构下不再适用。日志功能已内化为 agent_graph.py 的节点级日志。

本模块保留简单的日志辅助函数，供需要显式日志的场景使用。
"""

from typing import Any

from langchain_core.messages import AIMessage
from utils.logger import logger


def log_tool_call(tool_name: str, args: dict[str, Any]) -> None:
    """记录工具调用日志（替代原 monitor_tool 的装饰器模式）。"""
    logger.info(f"[调用工具]：{tool_name}，参数：{args}")


def log_tool_result(tool_name: str, success: bool = True, error: str = "") -> None:
    """记录工具执行结果日志。"""
    if success:
        logger.info(f"[工具{tool_name}]调用成功")
    else:
        logger.error(f"[工具{tool_name}]调用失败，错误信息：{error}")


def log_before_model(messages: list[Any]) -> None:
    """记录模型调用前状态（替代原 before_model 中间件）。"""
    logger.info(f"[模型调用前状态]：带有{len(messages)}条信息")
    if messages:
        last = messages[-1]
        logger.debug(
            f"[模型调用前状态]{type(last).__name__}: {last}"
        )


def log_after_model(message: AIMessage) -> None:
    """记录模型调用后状态。"""
    has_tools = bool(getattr(message, "tool_calls", None))
    logger.info(f"[模型调用后状态]：has_tool_calls={has_tools}")
