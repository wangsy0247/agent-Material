"""
ReAct Agent - 集成三层记忆系统 + MCP 工具

核心改进：
1. 支持注入 MemoryManager，自动管理对话上下文
2. 系统提示自动注入长期记忆（USER.md / MEMORY.md）
3. 对话历史自动保存到 SQLite 会话存储
4. Token 超限时自动触发上下文压缩
5. 支持 MCP 工具（原生工具 + 外部 MCP Server 工具混合）

工具来源：
    - 原生工具：rag_summary, mp_download_tool, mp_query_tool, mp_list_tool
    - MCP 工具：从 config/mcp_config.yaml 配置的 MCP Server 动态加载
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from model.model_factory import chat_model
from utils.prompt_loader import get_prompts
from agent.tools.agent_tools import get_all_tools
from agent.tools.middleware import monitor_tool, log_before_model
from utils.path_tools import get_abs_path
from utils.config_handler import prompt_config
from utils.langsmith_setup import get_run_config
from utils.logger import logger


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

        # 创建底层 Agent（系统提示会在每次调用时动态组装）
        self.agent = create_agent(
            model=chat_model,
            system_prompt=self._base_system_prompt,
            tools=tools,
            middleware=[monitor_tool, log_before_model],
        )

    def _ensure_session(self, session_id: str | None = None) -> str:
        """确保有活跃的会话，自动创建或加载"""
        if not self.memory_manager:
            return session_id or "default"

        # 如果指定了 session_id 且与当前不同，尝试加载
        if session_id and session_id != self.memory_manager._current_session_id:
            if not self.memory_manager.load_session(session_id):
                # 加载失败，创建新会话
                session_id = self.memory_manager.create_session(title=f"会话 {session_id[:8]}")

        # 如果没有当前会话，自动创建
        if not self.memory_manager._current_session_id:
            session_id = self.memory_manager.create_session(title="Agent 对话")

        return self.memory_manager._current_session_id

    def excutor_stream(self, query, session_id: str | None = None):
        """执行 Agent 流式推理，集成记忆系统

        Args:
            query: 用户问题
            session_id: 会话ID，为 None 时自动创建/复用当前会话

        Yields:
            文本片段（逐 token 输出）
        """
        # 确保会话
        active_session_id = self._ensure_session(session_id)

        if self.memory_manager:
            # 添加用户消息到记忆系统
            self.memory_manager.add_user_message(query)

            # 构建带记忆的完整上下文
            context_messages = self.memory_manager.build_context(
                query=query,
                include_memory=True,
            )
            logger.info(
                f"[ReactAgent] 上下文组装完成 | "
                f"session={active_session_id} | "
                f"messages={len(context_messages)} | "
                f"tokens={self.memory_manager.short_term.total_tokens}"
            )
        else:
            # 无记忆模式：单次请求
            context_messages = [{"role": "user", "content": query}]

        # 构造 Agent 输入
        input_data = {"messages": context_messages}

        full_answer = ""

        for chunk in self.agent.stream(
            input_data,
            stream_mode="values",
            config=get_run_config(
                run_name="agent_react",
                query=query,
                tags=["agent", "react", "with_memory" if self.memory_manager else "no_memory"],
            ),
        ):
            messages = chunk.get("messages")
            if not messages:
                continue

            latest_message = messages[-1]

            # 只输出 AI 生成的文本内容，跳过工具调用和工具结果
            if isinstance(latest_message, AIMessage):
                text = ""
                content = latest_message.content
                if isinstance(content, str) and content.strip():
                    text = content.strip()
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "thinking" and item.get("thinking"):
                                text += f"[思考] {item['thinking']}\n"
                            elif item.get("type") == "text" and item.get("text"):
                                text += item["text"]

                if text:
                    full_answer += text
                    yield text.strip() + "\n"

        # 保存助手回复到记忆系统
        if self.memory_manager and full_answer:
            self.memory_manager.add_assistant_message(full_answer.strip())
            logger.info(f"[ReactAgent] 对话已保存 | session={active_session_id}")

    def excutor(self, query, session_id: str | None = None) -> str:
        """执行 Agent 非流式推理，返回完整回答

        Args:
            query: 用户问题
            session_id: 会话ID

        Returns:
            完整回答文本
        """
        return "".join(self.excutor_stream(query, session_id))


if __name__ == "__main__":
    agent = ReactAgent()
    for chunk in agent.excutor_stream("帮我下载一种MOF材料"):
        print(chunk, end="", flush=True)
