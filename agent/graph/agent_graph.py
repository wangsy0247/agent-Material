"""
LangGraph Agent 状态机定义

核心特性：
1. 显式节点编排（memory → agent → human_approval → tools → agent）
2. 条件边控制循环与终止（should_continue）
3. Human-in-the-loop（human_approval_node 敏感操作审批）
4. 迭代次数上限防死循环（iteration_count > 10）
5. 支持 stream / invoke 双模式（通过编译后的 CompiledStateGraph）

图结构：
    entry ──▶ memory_node ──▶ agent_node ──▶ should_continue
                                              │
              ┌───────────────────────────────┼───────────────────────────────┐
              │                               │                               │
        iteration>10                    has tool_calls?                 无 tool_calls
              │                               │                               │
              ▼                               ▼                               ▼
         finalize                      human_approval_node               finalize
                                              │
                                       ┌──────┴──────┐
                                      批准          拒绝
                                       │             │
                                       ▼             ▼
                                    tool_node  ──▶ agent_node
                                       │
                                       └──────────▶ agent_node
"""

from __future__ import annotations

import time
from functools import partial
from typing import Annotated, Any

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from utils.logger import logger

# ---- 敏感工具判定（可扩展） ----
_SENSITIVE_TOOLS: set[str] = {"mp_download"}


def _get_sensitive_tools() -> set[str]:
    """读取敏感工具列表（支持运行时热更新）。"""
    try:
        # 尝试从 agent_config 读取（如果没有独立的 mcp_config）
        from utils.config_handler import agent_config

        keywords = agent_config.get("sensitive_keywords", [])
        if keywords:
            return set(keywords)
    except Exception:
        pass
    return _SENSITIVE_TOOLS


# ================================================================
# AgentState
# ================================================================


class AgentState(TypedDict):
    """LangGraph 全局状态。"""

    messages: Annotated[list[AnyMessage], add_messages]
    memory_context: str  # 长期记忆注入的上下文文本
    tool_calls: list[dict]  # 当前待执行的工具调用列表
    iteration_count: int  # 已执行推理轮次（防死循环）
    session_id: str  # 当前会话 ID
    pending_approval: dict | None  # HITL 挂起状态
    query: str  # 当前查询，用于记忆召回
    final_answer: str  # 最终答案（finalize_node 写入）


# ================================================================
# 工具日志包装
# ================================================================


def _make_tool_node_with_logging(tools: list[BaseTool]):
    """创建带日志的 ToolNode 闭包函数，替代原 middleware.monitor_tool。"""
    _tn = ToolNode(tools)

    async def _tool_node(state: AgentState) -> dict:
        messages: list[AnyMessage] = state.get("messages", [])
        if messages and isinstance(messages[-1], AIMessage):
            for tc in messages[-1].tool_calls or []:
                logger.info(f"[调用工具]：{tc['name']}，参数：{tc.get('args', {})}")

        result: dict = await _tn.ainvoke(state)

        for msg in result.get("messages", []):
            if isinstance(msg, ToolMessage):
                logger.info(f"[工具{msg.name}]调用成功")
        return result

    return _tool_node


# ================================================================
# 节点函数
# ================================================================


def memory_node(state: AgentState, memory_manager: Any | None) -> dict:
    """记忆注入节点：组装系统提示 + 长期记忆 + 对话历史。"""
    if memory_manager is None:
        # 无记忆模式：仅把 query 包装为 HumanMessage
        from langchain_core.messages import HumanMessage

        return {
            "messages": [HumanMessage(content=state["query"])],
            "memory_context": "",
        }

    session_id: str = state["session_id"]
    query: str = state["query"]

    # 调用 MemoryManager 构建完整上下文
    context_messages: list[dict] = memory_manager.build_context(
        session_id=session_id,
        query=query,
        include_memory=True,
    )

    # 转换为 LangChain Message 对象
    lc_messages: list[AnyMessage] = []
    for msg in context_messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "user":
            from langchain_core.messages import HumanMessage

            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        elif role == "tool":
            lc_messages.append(
                ToolMessage(
                    content=content,
                    tool_call_id=msg.get("tool_call_id", ""),
                    name=msg.get("name", ""),
                )
            )

    # 提取 memory_context 用于状态追踪
    memory_context: str = ""
    if lc_messages and isinstance(lc_messages[0], SystemMessage):
        memory_context = lc_messages[0].content

    logger.info(
        f"[memory_node] 上下文组装完成 | session={session_id} | "
        f"messages={len(lc_messages)} | has_memory={bool(memory_context)}"
    )

    return {
        "messages": lc_messages,
        "memory_context": memory_context,
    }


async def agent_node(
    state: AgentState,
    chat_model: Any,
    tools: list[BaseTool],
) -> dict:
    """Agent 推理节点：调用 LLM（已绑定工具）。"""
    iteration: int = state["iteration_count"]
    messages: list[AnyMessage] = state["messages"]

    # 替代原 log_before_model
    logger.info(
        f"[模型调用前状态]：带有{len(messages)}条信息 | "
        f"最后一条: {type(messages[-1]).__name__ if messages else 'None'}"
    )

    # 绑定工具并调用
    model_with_tools = chat_model.bind_tools(tools)
    response: AIMessage = await model_with_tools.ainvoke(messages)

    has_tools = bool(getattr(response, "tool_calls", None))
    logger.info(
        f"[agent_node] 第 {iteration + 1} 轮推理完成 | "
        f"has_tool_calls={has_tools}"
    )

    return {
        "messages": [response],
        "iteration_count": iteration + 1,
        "tool_calls": response.tool_calls if has_tools else [],
    }


def human_approval_node(state: AgentState) -> dict:
    """HITL 节点：检测敏感操作，需要时挂起等待人工确认。"""
    messages: list[AnyMessage] = state["messages"]
    last_msg = messages[-1]

    if not isinstance(last_msg, AIMessage):
        return {}

    tool_calls: list[dict] = last_msg.tool_calls or []
    if not tool_calls:
        return {}

    sensitive = _get_sensitive_tools()
    pending: list[dict] = [tc for tc in tool_calls if tc["name"] in sensitive]

    if not pending:
        # 无需审批，直接放行
        return {}

    # 调用 interrupt，graph 执行暂停，等待外部恢复
    approval: dict = interrupt(
        {
            "tool_calls": [
                {"name": tc["name"], "args": tc.get("args", {})} for tc in pending
            ],
            "message": (
                f"检测到敏感操作（{', '.join(tc['name'] for tc in pending)}），"
                f"请确认是否执行。"
            ),
        }
    )

    action: str = approval.get("action", "reject")
    comment: str = approval.get("comment", "")

    if action == "reject":
        logger.warning(
            f"[human_approval] 用户拒绝了操作: "
            f"{[tc['name'] for tc in pending]} | 原因: {comment}"
        )
        # 生成拒绝的 ToolMessage，让 Agent 重新推理
        rejected_messages: list[ToolMessage] = []
        for tc in tool_calls:
            rejected_messages.append(
                ToolMessage(
                    content=(
                        f"用户拒绝了工具 {tc['name']} 的调用。"
                        f"原因：{comment or '无'}"
                    ),
                    tool_call_id=tc["id"],
                    name=tc["name"],
                )
            )
        return {
            "messages": rejected_messages,
            "pending_approval": None,
        }

    # 用户批准：记录状态，让 tool_node 执行
    logger.info(f"[human_approval] 用户批准了操作: {[tc['name'] for tc in pending]}")
    return {
        "pending_approval": {
            "action": "approve",
            "timestamp": time.time(),
            "comment": comment,
        },
    }


def finalize_node(state: AgentState) -> dict:
    """结果组装节点：提取最终答案，处理截断提示。"""
    iteration: int = state["iteration_count"]
    messages: list[AnyMessage] = state["messages"]

    if iteration > 10:
        # 迭代上限截断
        truncation_msg = AIMessage(
            content="\n[系统提示：推理轮次已达上限（10轮），已强制终止。]"
        )
        last_content: str = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                last_content = msg.content
                break
        return {
            "messages": [truncation_msg],
            "final_answer": last_content + truncation_msg.content,
        }

    # 正常结束：从最后一条 AIMessage 提取答案
    final_answer: str = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            final_answer = msg.content
            break

    return {"final_answer": final_answer}


# ================================================================
# 条件边
# ================================================================


def should_continue(state: AgentState) -> str:
    """条件边：判断是继续推理还是返回结果。"""
    if state["iteration_count"] > 10:
        return "max_iterations"

    messages: list[AnyMessage] = state["messages"]
    if not messages:
        return "end"

    last_msg = messages[-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"

    return "end"


def after_approval(state: AgentState) -> str:
    """human_approval 节点后的条件边。"""
    messages: list[AnyMessage] = state["messages"]
    if messages and isinstance(messages[-1], ToolMessage):
        # 最后一条是 ToolMessage → human_approval_node 生成了拒绝结果
        return "rejected"
    return "approved"


# ================================================================
# Graph 构建
# ================================================================


def build_agent_graph(
    chat_model: Any,
    tools: list[BaseTool],
    memory_manager: Any | None = None,
) -> Any:
    """编译并返回 LangGraph CompiledStateGraph。

    Args:
        chat_model: 聊天模型实例（已配置 streaming）
        tools: 可用工具列表
        memory_manager: MemoryManager 实例（可选）

    Returns:
        CompiledStateGraph：已编译的状态机，支持 .stream / .ainvoke / .ainvoke
    """
    graph = StateGraph(AgentState)

    # 节点注册
    graph.add_node("memory", partial(memory_node, memory_manager=memory_manager))
    graph.add_node("agent", partial(agent_node, chat_model=chat_model, tools=tools))
    graph.add_node("human_approval", human_approval_node)

    # 工具节点：使用带日志的闭包函数
    graph.add_node("tools", _make_tool_node_with_logging(tools))

    graph.add_node("finalize", finalize_node)

    # 入口与边
    graph.set_entry_point("memory")
    graph.add_edge("memory", "agent")

    # agent_node 后的条件分支
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "max_iterations": "finalize",
            "tools": "human_approval",
            "end": "finalize",
        },
    )

    # human_approval 后的条件分支
    graph.add_conditional_edges(
        "human_approval",
        after_approval,
        {
            "approved": "tools",
            "rejected": "agent",
        },
    )

    # 工具执行后回到 agent_node 继续推理
    graph.add_edge("tools", "agent")

    # finalize 后结束
    graph.add_edge("finalize", END)

    # 使用 MemorySaver 作为 checkpointer，支持 HITL interrupt 恢复
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
