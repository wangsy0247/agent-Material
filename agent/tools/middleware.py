from langchain.agents.middleware import wrap_tool_call, before_model
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from typing import Callable
from utils.logger import logger
from langchain.agents import AgentState
from langgraph.runtime import Runtime


@wrap_tool_call
def monitor_tool(
    request: ToolCallRequest,  # 请求的数据封装
    handler: Callable[[ToolCallRequest], ToolMessage | Command],  # 执行的函数本身
) -> ToolMessage | Command:

    logger.info(
        f"[调用工具]：{request.tool_call['name']}，参数：{request.tool_call['args']}"
    )

    if request.tool_call["name"] == "fill_context_for_report":
        request.runtime.context["report"] = True

    try:
        result = handler(request)
        logger.info(f"[工具{request.tool_call['name']}调用成功")
        return result
    except Exception as e:
        logger.error(f"[工具{request.tool_call['name']}调用失败，错误信息：{e}")
        raise e


@before_model
def log_before_model(
    state: AgentState,  # 整个agent智能体中的状态记录
    runtime: Runtime,  # 记录了整个执行过程中的上下文信息
):
    logger.info(f"[模型调用前状态]：带有{len(state['messages'])}条信息")
    logger.debug(
        f"[模型调用前状态]{type(state['messages'][-1]).__name__}: {state['messages']}"
    )
    return None
