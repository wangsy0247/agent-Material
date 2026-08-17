"""
短期记忆 - 对话上下文窗口管理

仅维护当前会话的消息列表和 Token 计数。
对话历史的压缩由 LangGraph summarization_node（复用 ContextCompressor）负责，
避免双重压缩和压缩结果不一致的问题。

配置来源：config/memory_config.yaml -> manager.max_context_tokens
"""

from agent_material.memory.models import MessageRecord, ContextWindow
from agent_material.memory.utils import estimate_messages_tokens
from agent_material.utils.config_handler import memory_config
from agent_material.utils.logger import logger

# ---- 配置读取 ----
DEFAULT_MAX_TOKENS: int = memory_config.get("manager", {}).get("max_context_tokens", 8000)


class ShortTermMemory:
    """短期记忆管理器

    核心职责：
    1. 维护当前对话的 message 列表
    2. 追踪 token 使用量
    """

    def __init__(self, max_tokens: int | None = None):
        """初始化短期记忆

        Args:
            max_tokens: 最大 Token 预算，None 时读取配置
        """
        self.max_tokens: int = max_tokens or DEFAULT_MAX_TOKENS
        self._context: ContextWindow = ContextWindow(max_tokens=self.max_tokens)

    @property
    def messages(self) -> list[MessageRecord]:
        """当前消息列表"""
        return self._context.messages

    @property
    def total_tokens(self) -> int:
        """当前总 token 数"""
        return self._context.total_tokens

    def add_messages(self, messages: list[MessageRecord]) -> None:
        """批量追加消息并重算 Token 计数

        用于加载历史会话时批量恢复消息，不做压缩检查。
        """
        if not messages:
            return

        self._context.messages.extend(messages)
        self._context.total_tokens = estimate_messages_tokens(
            [m.to_lc_message() for m in self._context.messages]
        )

    def clear(self) -> None:
        """清空短期记忆"""
        self._context = ContextWindow(max_tokens=self.max_tokens)
        logger.info("[ShortTermMemory] 上下文已清空")
