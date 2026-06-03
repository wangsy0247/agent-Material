"""
LangGraph Agent 状态机定义

核心特性：
1. 显式节点编排（memory → summarization → agent → human_approval → tools → agent）
2. 条件边控制循环与终止（should_continue / should_summarize）
3. Human-in-the-loop（human_approval_node 敏感操作审批）
4. 迭代次数上限防死循环（iteration_count > 10）
5. 长对话自动 summarization（消息数超阈值时压缩历史）
6. 持久化 checkpointer（PostgresSaver / MemorySaver 降级）
7. 固定 thread_id = session_id，LangGraph 管理对话轮次内记忆

图结构：
    entry ──▶ memory_node ──▶ should_summarize
                                    │
                        need_sum? ──┼──▶ summarization_node ──▶ agent_node
                                    │                                │
                        no_sum?  ───┘                                │
                                                                     ▼
                                                              should_continue
                                                                     │
                    ┌────────────────┬───────────────────────────────┼───────────────────────────────┐
                    │                │                               │                               │
              iteration>10     has tool_calls?                 无 tool_calls
                    │                │                               │                               │
                    ▼                ▼                               ▼                               ▼
               finalize      human_approval_node                 finalize
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

import asyncio
import time
from functools import partial
from typing import Annotated, Any

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from agent_material.agents.graph.checkpointer_factory import create_checkpointer
from agent_material.utils.config_handler import agent_config
from agent_material.utils.logger import logger

# ---- 敏感工具判定（可扩展） ----
_SENSITIVE_TOOLS: set[str] = {"mp_download"}

# ---- Summarization 配置 ----
_SUMMARIZATION_CFG: dict[str, Any] = agent_config.get("summarization", {})
_SUMMARIZATION_ENABLED: bool = _SUMMARIZATION_CFG.get("enabled", True)
_SUMMARIZATION_THRESHOLD: int = _SUMMARIZATION_CFG.get("message_threshold", 20)
_SUMMARIZATION_PRESERVE_LAST_N: int = _SUMMARIZATION_CFG.get("preserve_last_n", 6)


def _get_sensitive_tools() -> set[str]:
    """读取敏感工具列表（支持运行时热更新）。"""
    try:
        keywords = agent_config.get("sensitive_keywords", [])
        if keywords:
            return set(keywords)
    except Exception:
        pass
    return _SENSITIVE_TOOLS


def _is_sensitive_tool(tool_name: str) -> bool:
    """
    判断工具是否为敏感操作。

    支持两种匹配模式：
    1. 精确匹配：tool_name 在敏感列表中
    2. 后缀匹配：去掉 server 前缀后的工具名在敏感列表中
    """
    sensitive = _get_sensitive_tools()
    if not sensitive:
        return False
    if tool_name in sensitive:
        return True
    if "__" in tool_name:
        raw_name = tool_name.rsplit("__", 1)[-1]
        return raw_name in sensitive
    return False


# ================================================================
# AgentState
# ================================================================


class AgentState(TypedDict):
    """LangGraph 全局状态。"""

    messages: Annotated[list[AnyMessage], add_messages]
    memory_context: str  # 长期记忆注入的上下文文本（用于状态追踪）
    tool_calls: list[dict]  # 当前待执行的工具调用列表
    iteration_count: int  # 已执行推理轮次（防死循环）
    session_id: str  # 当前会话 ID
    pending_approval: dict | None  # HITL 挂起状态
    query: str  # 当前查询，用于记忆召回
    final_answer: str  # 最终答案（finalize_node 写入）
    user_id: str  # 当前用户 ID（用于记忆系统按用户隔离）


# ================================================================
# 工具日志包装
# ================================================================


def _make_tool_node_with_logging(tools: list[BaseTool]):
    """创建带日志的 ToolNode 闭包函数。"""
    _tn = ToolNode(tools)

    async def _tool_node(state: AgentState) -> dict:
        messages: list[AnyMessage] = state.get("messages", [])
        if messages and isinstance(messages[-1], AIMessage):
            for tc in messages[-1].tool_calls or []:
                logger.info(f"[调用工具]：{tc['name']}，参数：{tc.get('args', {})}")

        result: dict = await _tn.ainvoke(state)

        for msg in result.get("messages", []):
            if isinstance(msg, ToolMessage):
                _maybe_offload_tool_result(msg)
                logger.info(f"[工具{msg.name}]调用成功")
        return result

    return _tool_node


def _get_offload_threshold() -> int:
    """读取工具结果卸载阈值（字符数），默认 10 万字符。"""
    try:
        from agent_material.agents.tools.mcp_config import get_mcp_global_defaults
        return get_mcp_global_defaults().get("max_inline_result_chars", 100_000)
    except Exception:
        return 100_000


def _maybe_offload_tool_result(msg: ToolMessage) -> None:
    """如果 ToolMessage 内容过长，将其卸载到磁盘文件。"""
    import json
    import time as _time
    from pathlib import Path

    content = msg.content
    if not content:
        return

    if isinstance(content, (list, dict)):
        try:
            text = json.dumps(content, ensure_ascii=False, indent=2)
        except Exception:
            text = str(content)
    else:
        text = str(content)

    threshold = _get_offload_threshold()
    if len(text) <= threshold:
        return

    offload_dir = Path("workspace/mcp_results")
    offload_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(_time.time())
    safe_name = msg.name.replace("/", "_").replace("\\", "_")[:50]
    filename = f"{safe_name}_{timestamp}.txt"
    filepath = offload_dir / filename

    try:
        filepath.write_text(text, encoding="utf-8")
        logger.info(
            f"[ToolOffload] 工具 '{msg.name}' 结果过大 ({len(text)} 字符)，"
            f"已卸载到 {filepath}"
        )
        msg.content = (
            f"[系统提示：工具返回结果过大，已保存到文件]\n"
            f"文件路径: {filepath}\n"
            f"文件大小: {len(text)} 字符\n"
            f"请使用 FileReadTool 读取该文件获取完整内容。"
        )
    except Exception as e:
        logger.error(f"[ToolOffload] 卸载工具 '{msg.name}' 结果失败: {e}")
        msg.content = (
            f"{text[:threshold]}\n\n"
            f"[系统提示：结果共 {len(text)} 字符，超过阈值 {threshold}，"
            f"但自动落盘失败: {e}]"
        )


# ================================================================
# 节点函数
# ================================================================


def memory_node(state: AgentState, memory_manager: Any | None) -> dict:
    """记忆注入节点：只负责长期记忆 + 系统提示注入。

    对话历史由 LangGraph checkpointer 管理，此节点不再重复注入历史。
    使用 RemoveMessage 替换旧的 system message，避免重复累积。
    """
    messages: list[AnyMessage] = state.get("messages", [])
    query: str = state.get("query", "")

    # 1. 收集要移除的旧 system messages
    updates: list[AnyMessage] = []
    for msg in messages:
        if isinstance(msg, SystemMessage) and msg.id:
            updates.append(RemoveMessage(id=msg.id))

    if memory_manager is None:
        # 无记忆模式：仅将 query 作为 HumanMessage 传入
        if query:
            updates.append(HumanMessage(content=query))
        return {"messages": updates, "memory_context": ""}

    # 2. 构建新的 system context（系统提示 + 长期记忆）
    user_id: str = state.get("user_id", "default")
    system_content: str = memory_manager.build_system_context(
        user_id=user_id,
        query=query,
        include_memory=True,
    )

    if system_content:
        updates.append(SystemMessage(content=system_content))

    logger.info(
        f"[memory_node] 记忆注入 | session={state.get('session_id')} | "
        f"removed_systems={len([m for m in messages if isinstance(m, SystemMessage)])} | "
        f"has_memory={bool(system_content)}"
    )

    return {
        "messages": updates,
        "memory_context": system_content,
    }


# ---- Message 转换辅助函数（LangChain <-> MessageRecord） ----


def _lc_messages_to_records(messages: list[AnyMessage]) -> list:
    """LangChain Message 列表 -> MessageRecord 列表（供 ContextCompressor 使用）。"""
    from agent_material.memory.models import MessageRecord
    from agent_material.memory.utils import estimate_tokens, normalize_message_content

    records: list[MessageRecord] = []
    for msg in messages:
        role = "user"
        if isinstance(msg, SystemMessage):
            role = "system"
        elif isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, ToolMessage):
            role = "tool"

        tool_calls = None
        tool_call_id = None
        name = None
        if isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None)
        elif isinstance(msg, ToolMessage):
            tool_call_id = getattr(msg, "tool_call_id", None)
            name = getattr(msg, "name", None)

        content = normalize_message_content(msg.content or "")
        records.append(
            MessageRecord(
                role=role,  # type: ignore[arg-type]
                content=content,
                tool_call_id=tool_call_id,
                tool_calls=tool_calls,
                name=name,
                timestamp=time.time(),
                token_count=estimate_tokens(content),
            )
        )
    return records


def _record_to_lc_message(msg) -> AnyMessage:
    """MessageRecord -> LangChain Message。"""
    if msg.role == "system":
        return SystemMessage(content=msg.content)
    if msg.role == "user":
        return HumanMessage(content=msg.content)
    if msg.role == "assistant":
        return AIMessage(content=msg.content, tool_calls=msg.tool_calls)
    if msg.role == "tool":
        return ToolMessage(
            content=msg.content,
            tool_call_id=msg.tool_call_id or "",
            name=msg.name or "",
        )
    return HumanMessage(content=msg.content)


async def summarization_node(
    state: AgentState,
    compressor: Any,
) -> dict:
    """长对话压缩节点：复用 ContextCompressor 四阶段算法压缩历史。

    策略：
    1. 分离对话消息（不含 system message，避免与 memory_node 冲突）
    2. 调用 ContextCompressor 执行四阶段压缩（Smart Collapse + 保护区 + LLM 摘要 + 清洗）
    3. 用 RemoveMessage 删除被压缩的旧消息
    4. 插入压缩结果（保护区 + 摘要），摘要转为 AIMessage 避免 system message 重复
    """
    messages: list[AnyMessage] = state.get("messages", [])

    # 只压缩非 system 的对话消息
    chat_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

    if len(chat_msgs) <= _SUMMARIZATION_PRESERVE_LAST_N + 2:
        return {}

    # 转换为 MessageRecord 供 ContextCompressor 使用
    records = _lc_messages_to_records(chat_msgs)

    logger.info(
        f"[summarization_node] 触发压缩 | "
        f"chat_messages={len(chat_msgs)} | records={len(records)}"
    )

    # 调用 ContextCompressor 四阶段压缩（在线程池中执行同步 LLM 调用）
    try:
        from agent_material.memory.context_compressor import ContextCompressor

        if compressor is None:
            compressor = ContextCompressor()

        result = await asyncio.to_thread(
            compressor.compress,
            messages=records,
            max_tokens=_SUMMARIZATION_CFG.get("max_tokens", 4000),
            protect_first_n=0,
            protect_last_n=_SUMMARIZATION_PRESERVE_LAST_N,
        )
    except Exception as e:
        logger.error(f"[summarization_node] ContextCompressor 压缩失败: {e}，跳过")
        return {}

    # 删除所有旧的对话消息
    updates: list[AnyMessage] = []
    for msg in chat_msgs:
        if msg.id:
            updates.append(RemoveMessage(id=msg.id))

    # 插入压缩后的消息
    # 注意：ContextCompressor 返回的摘要是 system role，为避免与 memory_node 的
    # system message 冲突，将摘要转为 AIMessage（LLM 仍能识别为历史上下文）
    for msg in result.messages:
        lc_msg = _record_to_lc_message(msg)
        if isinstance(lc_msg, SystemMessage):
            # 摘要以 AIMessage 形式插入，保留 system message 语义标记
            lc_msg = AIMessage(
                content=f"【上文摘要】\n{msg.content}\n【摘要结束，继续对话】"
            )
        updates.append(lc_msg)

    saved = len(chat_msgs) - len(result.messages)
    logger.info(
        f"[summarization_node] 压缩完成 | "
        f"original={len(chat_msgs)} -> compressed={len(result.messages)} | "
        f"saved={saved} | tokens={result.total_tokens}"
    )

    return {"messages": updates}


def _get_message_role_label(msg: AnyMessage) -> str:
    """获取消息的角色标签（用于 summarization 格式化）。"""
    if isinstance(msg, SystemMessage):
        return "系统"
    if isinstance(msg, HumanMessage):
        return "用户"
    if isinstance(msg, AIMessage):
        label = "助手"
        if msg.tool_calls:
            tool_names = [tc.get("name", "?") for tc in msg.tool_calls]
            label += f" [调用: {', '.join(tool_names)}]"
        return label
    if isinstance(msg, ToolMessage):
        return f"工具:{msg.name}"
    return "未知"


def _normalize_content_block(block: dict) -> str:
    """将多模态 content block 转换为纯文本描述。"""
    block_type = block.get("type", "")
    if block_type == "text":
        return block.get("text", "")
    if block_type == "image_url":
        image_url = block.get("image_url", {})
        url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
        return f"[图片: {url}]"
    if block_type == "image":
        return "[图片]"
    if block_type == "file":
        file_info = block.get("file", {})
        if isinstance(file_info, dict):
            url = file_info.get("url", "")
            mime = file_info.get("mime_type", "")
            return f"[文件: {url or mime or 'unknown'}]"
        return f"[文件: {file_info}]"
    return str(block)


def _sanitize_messages_for_llm(messages: list[AnyMessage]) -> list[AnyMessage]:
    """清洗消息列表，移除孤立的 ToolMessage。

    Anthropic 等模型要求每个 tool_result (ToolMessage) 必须有对应的
    tool_use (AIMessage.tool_calls) 在其之前。如果对话历史被压缩或
    checkpointer 中残留了损坏状态，可能出现 ToolMessage 失去 parent 的情况，
    导致 API 报 400。此函数作为防御性清洗，在调用 LLM 前执行。
    """
    valid_tool_call_ids: set[str] = set()
    cleaned: list[AnyMessage] = []
    removed_count = 0

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tc_id = tc.get("id")
                if tc_id:
                    valid_tool_call_ids.add(tc_id)

        if isinstance(msg, ToolMessage):
            if msg.tool_call_id and msg.tool_call_id not in valid_tool_call_ids:
                removed_count += 1
                continue  # 跳过孤立的 ToolMessage

        cleaned.append(msg)

    if removed_count:
        logger.warning(
            f"[sanitize_messages] 移除 {removed_count} 条孤立 ToolMessage，"
            f"剩余 {len(cleaned)} 条消息"
        )
    return cleaned


async def agent_node(
    state: AgentState,
    chat_model: Any,
    tools: list[BaseTool],
) -> dict:
    """Agent 推理节点：调用 LLM（已绑定工具）。"""
    iteration: int = state["iteration_count"]
    messages: list[AnyMessage] = state["messages"]

    # 防御性清洗：移除孤立的 ToolMessage（可能来自损坏的 checkpoint）
    messages = _sanitize_messages_for_llm(messages)

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

    pending: list[dict] = [tc for tc in tool_calls if _is_sensitive_tool(tc["name"])]

    if not pending:
        return {}

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

    final_answer: str = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            final_answer = msg.content
            break

    return {"final_answer": final_answer}


# ================================================================
# 条件边
# ================================================================


def should_summarize(state: AgentState) -> str:
    """条件边：判断是否需要先压缩历史再进入 agent_node。"""
    if not _SUMMARIZATION_ENABLED:
        return "agent"

    messages: list[AnyMessage] = state.get("messages", [])
    # 只统计非 system 的对话消息
    chat_count = len([m for m in messages if not isinstance(m, SystemMessage)])

    if chat_count > _SUMMARIZATION_THRESHOLD:
        return "summarize"
    return "agent"


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
        CompiledStateGraph：已编译的状态机，支持 .stream / .ainvoke
    """
    graph = StateGraph(AgentState)

    # 节点注册
    graph.add_node("memory", partial(memory_node, memory_manager=memory_manager))
    graph.add_node("agent", partial(agent_node, chat_model=chat_model, tools=tools))
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("tools", _make_tool_node_with_logging(tools))
    graph.add_node("finalize", finalize_node)

    if _SUMMARIZATION_ENABLED:
        from agent_material.memory.context_compressor import ContextCompressor

        compressor = ContextCompressor(model=chat_model)
        graph.add_node(
            "summarize",
            partial(summarization_node, compressor=compressor),
        )

    # 入口与边
    graph.set_entry_point("memory")

    # memory_node 后的条件分支：是否需要 summarization
    if _SUMMARIZATION_ENABLED:
        graph.add_conditional_edges(
            "memory",
            should_summarize,
            {
                "summarize": "summarize",
                "agent": "agent",
            },
        )
        graph.add_edge("summarize", "agent")
    else:
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

    # 使用可配置的 checkpointer（PostgresSaver / MemorySaver 降级）
    checkpointer = create_checkpointer()
    return graph.compile(checkpointer=checkpointer)
