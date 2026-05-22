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
    - 本地工具：rag_summary, mp_download, mp_query, mp_list
    - MCP 工具：通过 mcp_adapter 动态加载
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

from agent.graph.agent_graph import build_agent_graph
from agent.tools.agent_tools import get_all_tools
from model.model_factory import chat_model
from utils.config_handler import prompt_config
from utils.langsmith_setup import get_run_config
from utils.logger import logger
from utils.path_tools import get_abs_path
from utils.prompt_loader import get_prompts


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

    def _ensure_session(self, session_id: str | None = None) -> str:
        """确保有活跃的会话，自动创建或加载。"""
        if not self.memory_manager:
            return session_id or "default"

        if session_id:
            if session_id not in self.memory_manager._session_pool:
                if not self.memory_manager.load_session(session_id):
                    session_id = self.memory_manager.create_session(
                        title=f"会话 {session_id[:8]}"
                    )
            else:
                self.memory_manager.session_store.update_session(
                    session_id, is_active=True, end_reason=None
                )
            return session_id

        return self.memory_manager.create_session(title="Agent 对话")

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

    async def excutor_stream(self, query, session_id: str | None = None):
        """执行 Agent 流式推理，集成记忆系统与 HITL。

        设计变更（固定 thread_id）：
        - thread_id 固定为 session_id，同一 session 的所有请求共享同一个 LangGraph thread
        - LangGraph checkpointer 自动管理对话轮次内的消息历史
        - 每次调用只传入当前 query 的 HumanMessage，不再组装完整历史
        - HITL 中断状态由持久化 checkpointer 保存，恢复时直接使用 session_id

        Args:
            query: 用户问题
            session_id: 会话ID，为 None 时自动创建/复用当前会话

        Yields:
            文本片段（逐 token 输出）或 HITL 事件字典
        """
        active_session_id = self._ensure_session(session_id)

        # 将用户消息保存到记忆系统（长期/中期记忆）
        if self.memory_manager:
            self.memory_manager.add_user_message(active_session_id, query)

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

        full_answer = ""

        async for chunk in self.graph.astream(
            initial_state, config, stream_mode="updates"
        ):
            if not isinstance(chunk, dict):
                continue

            # 检测 HITL interrupt 事件
            if "__interrupt__" in chunk:
                interrupt_tuple = chunk["__interrupt__"]
                interrupt_value = interrupt_tuple[0].value if interrupt_tuple else {}
                logger.warning(
                    f"[ReactAgent] HITL 中断 | session={active_session_id} | "
                    f"tools={interrupt_value.get('tool_calls', [])}"
                )
                yield {
                    "type": "approval_required",
                    "session_id": active_session_id,
                    "tool_calls": interrupt_value.get("tool_calls", []),
                    "message": interrupt_value.get("message", ""),
                }
                return

            # 只处理 agent 节点生成的新消息
            agent_update = chunk.get("agent")
            if agent_update:
                for msg in agent_update.get("messages", []):
                    if isinstance(msg, AIMessage):
                        text = self._extract_text_from_message(msg)
                        if text:
                            full_answer += text
                            yield text.strip() + "\n"

        # 保存助手回复到记忆系统
        if self.memory_manager and full_answer:
            self.memory_manager.add_assistant_message(
                active_session_id, full_answer.strip()
            )
            logger.info(f"[ReactAgent] 对话已保存 | session={active_session_id}")

    async def excutor(self, query, session_id: str | None = None) -> str | dict:
        """执行 Agent 非流式推理，返回完整回答或 HITL 事件。"""
        chunks = []
        async for chunk in self.excutor_stream(query, session_id):
            if isinstance(chunk, dict):
                return chunk
            chunks.append(chunk)
        return "".join(chunks)

    # ------------------------------------------------------------------
    # HITL 恢复接口
    # ------------------------------------------------------------------

    async def resume_stream(
        self,
        session_id: str,
        approval_result: dict,
    ):
        """恢复被 HITL 挂起的 Agent 执行（流式）。

        设计变更：
        - 不再依赖内存中的 _hitl_threads 映射
        - 直接使用 session_id 作为 thread_id，从持久化 checkpointer 恢复状态

        Args:
            session_id: 会话ID（与中断时一致）
            approval_result: {"action": "approve" | "reject", "comment": str}

        Yields:
            文本片段或新的 HITL 事件字典
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

        full_answer = ""

        async for chunk in self.graph.astream(
            Command(resume=approval_result),
            config,
            stream_mode="updates",
        ):
            if not isinstance(chunk, dict):
                continue

            # 检测 HITL interrupt 事件
            if "__interrupt__" in chunk:
                interrupt_tuple = chunk["__interrupt__"]
                interrupt_value = interrupt_tuple[0].value if interrupt_tuple else {}
                yield {
                    "type": "approval_required",
                    "session_id": session_id,
                    "tool_calls": interrupt_value.get("tool_calls", []),
                    "message": interrupt_value.get("message", ""),
                }
                return

            # 只处理 agent 节点生成的新消息
            agent_update = chunk.get("agent")
            if agent_update:
                for msg in agent_update.get("messages", []):
                    if isinstance(msg, AIMessage):
                        text = self._extract_text_from_message(msg)
                        if text:
                            full_answer += text
                            yield text.strip() + "\n"

        # 保存助手回复
        if self.memory_manager and full_answer:
            self.memory_manager.add_assistant_message(
                session_id, full_answer.strip()
            )
            logger.info(f"[ReactAgent] HITL 恢复后对话已保存 | session={session_id}")

    async def resume(self, session_id: str, approval_result: dict) -> str | dict:
        """恢复被 HITL 挂起的 Agent 执行（非流式）。"""
        chunks = []
        async for chunk in self.resume_stream(session_id, approval_result):
            if isinstance(chunk, dict):
                return chunk
            chunks.append(chunk)
        return "".join(chunks)


if __name__ == "__main__":
    import asyncio

    agent = ReactAgent()

    async def main():
        async for chunk in agent.excutor_stream("帮我下载一种MOF材料"):
            if isinstance(chunk, dict):
                print(f"\n[HITL] {chunk}")
            else:
                print(chunk, end="", flush=True)

    asyncio.run(main())
