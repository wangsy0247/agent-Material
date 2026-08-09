"""
ReAct Agent - 基于 LangGraph StateGraph 的状态机实现

核心改进：
1. 替换 create_agent 为显式 StateGraph（memory -> summarization -> agent -> human_approval -> tools）
2. 条件边控制循环与终止（should_continue + iteration_count 上限）
3. Human-in-the-loop：敏感操作前挂起等待人工确认
4. 长期记忆系统集成（memory_node 注入 USER.md / 项目知识）
5. LangGraph checkpointer 管理对话轮次内记忆（固定 thread_id = session_id）
6. 支持 stream / invoke 双模式

工具来源：
    - MCP 工具：通过 MCP 动态加载（包含已解耦的 RAG：rag_search, rag_query 等）
"""

from __future__ import annotations

import time

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

from agent_material.agents.graph.agent_graph import build_agent_graph
from agent_material.agents.tools.agent_tools import get_all_tools
from agent_material.models.model_factory import chat_model
from agent_material.utils.config_handler import prompt_config
from agent_material.utils.langfuse_setup import get_run_config
from agent_material.observability.structured_logger import get_logger
from agent_material.observability.metrics import (
    agent_iteration_count,
    agent_duration,
)
from agent_material.utils.path_tools import get_abs_path
from agent_material.utils.prompt_loader import get_prompts

logger = get_logger("React_agent")


class ReactAgent:
    def __init__(self, memory_manager=None, tools: list[BaseTool] | None = None):
        """
        Args:
            memory_manager: MemoryManager 实例，为 None 时不启用记忆功能
            tools: 自定义工具列表。为 None 时自动加载原生工具 + MCP 工具
        """
        self.memory_manager = memory_manager

        # 加载原始系统提示
        self._base_system_prompt = get_prompts(
            get_abs_path(prompt_config["sys_prompt_path"])
        )

        # 如果启用了记忆，将系统提示设置到 MemoryManager
        if self.memory_manager:
            self.memory_manager.set_system_prompt(self._base_system_prompt)
            logger.info("[ReactAgent] 记忆系统已集成")

        # 加载工具列表
        if tools is None:
            tools = get_all_tools()
        self._tools = tools

        logger.info(
            f"[ReactAgent] Agent 初始化 | 工具数={len(tools)} | "
            f"工具名={[t.name for t in tools]}"
        )

        # 编译 LangGraph 状态机
        self.graph = build_agent_graph(
            chat_model=chat_model,
            tools=tools,
            memory_manager=memory_manager,
        )

    def _ensure_session(self, session_id: str | None = None, user_id: str = "default") -> str:
        """确保有活跃的会话，自动创建或加载。"""
        if not self.memory_manager:
            return session_id or "default"

        if session_id:
            pool_key = (user_id, session_id)
            if pool_key not in self.memory_manager._session_pool:
                if not self.memory_manager.load_session(session_id, user_id=user_id):
                    session_id = self.memory_manager.create_session(
                        user_id=user_id, title=f"会话 {session_id[:8]}"
                    )
            else:
                self.memory_manager.session_store.update_session(
                    session_id, user_id=user_id, is_active=True, end_reason=None
                )
            return session_id

        return self.memory_manager.create_session(user_id=user_id, title="Agent 对话")

    @staticmethod
    def _extract_text_from_message(msg: AIMessage) -> str:
        """从 AIMessage 中提取可展示的文本内容。"""
        text = ""
        content = msg.content
        if isinstance(content, str) and content.strip():
            text = content.strip()
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "thinking" and item.get("thinking"):
                        text += f"[思考] {item['thinking']}\n"
                    elif item.get("type") == "text" and item.get("text"):
                        text += item["text"]
        return text

    async def _stream_events(self, astream_iter, session_id: str, full_answer: list[str]):
        """将 astream(stream_mode=["messages", "updates"]) 的 (mode, payload) 二元组
        转换为结构化事件 dict。

        - messages 模式（仅 agent 节点）：文本增量 → token；reasoning_content /
          thinking block → thinking（仅用于流式展示，不参与 full_answer 拼接）
        - updates 模式：agent 节点 AIMessage 带 tool_calls → tool_call；
          tools 节点 ToolMessage → tool_result；__interrupt__ → approval_required
        - full_answer 只累积"无 tool_calls 的最终轮 AIMessage 文本"（从 updates 取，
          不从 token 流拼），保证存入记忆的是干净最终答案（修复中间推理混入记忆）

        Args:
            astream_iter: self.graph.astream(..., stream_mode=["messages", "updates"]) 迭代器
            session_id: 会话ID（用于 approval_required 事件）
            full_answer: 单元素 list，用于向调用方回传累积的最终答案
        """
        async for mode, payload in astream_iter:
            if mode == "messages":
                chunk, metadata = payload
                if not isinstance(chunk, AIMessageChunk):
                    continue
                if metadata.get("langgraph_node") != "agent":
                    continue
                # 模型 reasoning 增量（qwen 等可能走 additional_kwargs.reasoning_content）
                reasoning = chunk.additional_kwargs.get("reasoning_content")
                if reasoning:
                    yield {"type": "thinking", "content": reasoning}
                content = chunk.content
                if isinstance(content, str):
                    if content:
                        yield {"type": "token", "content": content}
                elif isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") == "text" and block.get("text"):
                            yield {"type": "token", "content": block["text"]}
                        elif block.get("type") == "thinking" and block.get("thinking"):
                            yield {"type": "thinking", "content": block["thinking"]}
            elif mode == "updates":
                if not isinstance(payload, dict):
                    continue

                # 检测 HITL interrupt 事件
                if "__interrupt__" in payload:
                    interrupt_tuple = payload["__interrupt__"]
                    interrupt_value = interrupt_tuple[0].value if interrupt_tuple else {}
                    yield {
                        "type": "approval_required",
                        "session_id": session_id,
                        "tool_calls": interrupt_value.get("tool_calls", []),
                        "message": interrupt_value.get("message", ""),
                    }
                    return

                # agent 节点：tool_calls → tool_call 事件；最终轮文本 → 累积 full_answer
                agent_update = payload.get("agent")
                if agent_update:
                    for msg in agent_update.get("messages", []):
                        if not isinstance(msg, AIMessage):
                            continue
                        if msg.tool_calls:
                            for tc in msg.tool_calls:
                                yield {
                                    "type": "tool_call",
                                    "name": tc.get("name"),
                                    "args": tc.get("args"),
                                }
                        else:
                            text = self._extract_text_from_message(msg)
                            if text:
                                full_answer[0] += text

                # tools 节点：ToolMessage → tool_result 事件（截断 500 字，带截断标记）
                tools_update = payload.get("tools")
                if tools_update:
                    for msg in tools_update.get("messages", []):
                        if isinstance(msg, ToolMessage):
                            content = (
                                msg.content if isinstance(msg.content, str) else str(msg.content)
                            )
                            yield {
                                "type": "tool_result",
                                "name": msg.name,
                                "content": content[:500],
                                "truncated": len(content) > 500,
                            }

    async def excutor_stream(self, query, session_id: str | None = None, user_id: str = "default"):
        """执行 Agent 流式推理，集成记忆系统与 HITL。

        设计变更（固定 thread_id）：
        - thread_id 固定为 session_id，同一 session 的所有请求共享同一个 LangGraph thread
        - LangGraph checkpointer 自动管理对话轮次内的消息历史
        - 每次调用只传入当前 query 的 HumanMessage，不再组装完整历史
        - HITL 中断状态由持久化 checkpointer 保存，恢复时直接使用 session_id

        Args:
            query: 用户问题
            session_id: 会话ID，为 None 时自动创建/复用当前会话
            user_id: 用户ID（多租户隔离）

        Yields:
            结构化事件 dict：{"type": "token"|"thinking"|"tool_call"|"tool_result"|"approval_required", ...}
        """
        active_session_id = self._ensure_session(session_id, user_id=user_id)

        # 将用户消息保存到记忆系统（长期/中期记忆）
        if self.memory_manager:
            self.memory_manager.add_user_message(active_session_id, query, user_id=user_id)

        # 初始状态：只包含当前 query（HumanMessage）
        # LangGraph checkpointer 会自动加载该 thread 的历史状态
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "memory_context": "",
            "tool_calls": [],
            "iteration_count": 0,
            "session_id": active_session_id,
            "pending_approval": None,
            "query": query,
            "final_answer": "",
            "user_id": user_id,
        }

        # 固定 thread_id = session_id
        thread_id = active_session_id

        config = {
            "configurable": {"thread_id": thread_id},
            **get_run_config(
                run_name="agent_react",
                query=query,
                tags=["agent", "react", "with_memory" if self.memory_manager else "no_memory"],
            ),
        }

        full_answer = [""]
        t_start = time.time()
        final_state = initial_state

        astream_iter = self.graph.astream(
            initial_state, config, stream_mode=["messages", "updates"]
        )
        async for event in self._stream_events(astream_iter, active_session_id, full_answer):
            if event["type"] == "approval_required":
                agent_iteration_count.observe(final_state.get("iteration_count", 0))
                logger.warning(
                    "agent_hitl_interrupt",
                    extra={
                        "session_id": active_session_id,
                        "tools": event.get("tool_calls", []),
                    },
                )
                yield event
                return
            yield event

        # 记录指标
        agent_iteration_count.observe(final_state.get("iteration_count", 0))
        agent_duration.labels(mode="stream").observe(time.time() - t_start)

        # 保存助手回复到记忆系统（干净最终答案，不含中间推理）
        answer = full_answer[0].strip()
        if self.memory_manager and answer:
            self.memory_manager.add_assistant_message(
                active_session_id, answer, user_id=user_id
            )
            logger.info(
                "agent_stream_done",
                extra={
                    "session_id": active_session_id,
                    "user_id": user_id,
                    "iterations": final_state.get("iteration_count", 0),
                    "duration_s": time.time() - t_start,
                },
            )

    async def excutor(self, query, session_id: str | None = None, user_id: str = "default") -> str | dict:
        """执行 Agent 非流式推理，返回完整回答或 HITL 事件。"""
        chunks = []
        async for event in self.excutor_stream(query, session_id=session_id, user_id=user_id):
            if event.get("type") == "approval_required":
                return event
            if event.get("type") == "token":
                chunks.append(event["content"])
        return "".join(chunks)

    # ------------------------------------------------------------------
    # HITL 恢复接口
    # ------------------------------------------------------------------

    async def resume_stream(
        self,
        session_id: str,
        approval_result: dict,
        user_id: str = "default",
    ):
        """恢复被 HITL 挂起的 Agent 执行（流式）。

        设计变更：
        - 不再依赖内存中的 _hitl_threads 映射
        - 直接使用 session_id 作为 thread_id，从持久化 checkpointer 恢复状态

        Args:
            session_id: 会话ID（与中断时一致）
            approval_result: {"action": "approve" | "reject", "comment": str}
            user_id: 用户ID（多租户隔离）

        Yields:
            结构化事件 dict：{"type": "token"|"thinking"|"tool_call"|"tool_result"|"approval_required", ...}
        """
        thread_id = session_id

        config = {
            "configurable": {"thread_id": thread_id},
            **get_run_config(
                run_name="agent_react_resume",
                query=f"HITL resume: {approval_result.get('action')}",
                tags=["agent", "react", "hitl_resume"],
            ),
        }

        full_answer = [""]

        astream_iter = self.graph.astream(
            Command(resume=approval_result),
            config,
            stream_mode=["messages", "updates"],
        )
        async for event in self._stream_events(astream_iter, session_id, full_answer):
            if event["type"] == "approval_required":
                yield event
                return
            yield event

        # 保存助手回复（干净最终答案，不含中间推理）
        answer = full_answer[0].strip()
        if self.memory_manager and answer:
            self.memory_manager.add_assistant_message(
                session_id, answer, user_id=user_id
            )
            logger.info(f"[ReactAgent] HITL 恢复后对话已保存 | session={session_id} | user={user_id}")

    async def resume(self, session_id: str, approval_result: dict, user_id: str = "default") -> str | dict:
        """恢复被 HITL 挂起的 Agent 执行（非流式）。"""
        chunks = []
        async for event in self.resume_stream(session_id, approval_result, user_id=user_id):
            if event.get("type") == "approval_required":
                return event
            if event.get("type") == "token":
                chunks.append(event["content"])
        return "".join(chunks)


if __name__ == "__main__":
    import asyncio

    agent = ReactAgent()

    async def main():
        async for event in agent.excutor_stream("帮我下载一种MOF材料"):
            event_type = event.get("type")
            if event_type == "approval_required":
                print(f"\n[HITL] {event}")
            elif event_type == "token":
                print(event["content"], end="", flush=True)
            else:
                print(f"\n[{event_type}] {event}")

    asyncio.run(main())
