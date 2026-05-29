"""
短期记忆 - 对话上下文窗口管理

参考 Claude-Code + Hermes-Agent 设计：
- Token 预算管理（滑动窗口）
- 智能保护头尾消息
- 触发压缩时调用 ContextCompressor

配置来源：config/memory_config.yaml -> short_term
"""

import time
from agent_material.memory.models import MessageRecord, ContextWindow
from agent_material.memory.utils import estimate_messages_tokens
from agent_material.utils.config_handler import memory_config
from agent_material.utils.logger import logger

# ---- 配置读取 ----
_SHORT_TERM_CFG: dict = memory_config.get("short_term", {})

DEFAULT_MAX_TOKENS: int = memory_config.get("manager", {}).get("max_context_tokens")
DEFAULT_COMPRESS_THRESHOLD: float = _SHORT_TERM_CFG.get("compress_threshold")
DEFAULT_PROTECT_FIRST_N: int = _SHORT_TERM_CFG.get("protect_first_n")
DEFAULT_PROTECT_LAST_N: int = _SHORT_TERM_CFG.get("protect_last_n")

# 可压缩区域最小消息数
MIN_COMPRESSIBLE_MESSAGES: int = 3


class ShortTermMemory:
    """短期记忆管理器

    核心职责：
    1. 维护当前对话的 message 列表
    2. 监控 token 使用量，接近预算时触发压缩
    3. 保护系统提示和最近消息不被截断
    """

    def __init__(
        self,
        max_tokens: int | None = None,
        compress_threshold: float | None = None,
        protect_first_n: int | None = None,
        protect_last_n: int | None = None,
    ):
        """初始化短期记忆

        Args:
            max_tokens: 最大 Token 预算，None 时读取配置
            compress_threshold: 触发压缩阈值（比例），None 时读取配置
            protect_first_n: 保护头部消息数，None 时读取配置
            protect_last_n: 保护尾部消息数，None 时读取配置
        """
        self.max_tokens: int = max_tokens or DEFAULT_MAX_TOKENS
        self.compress_threshold: float = compress_threshold if compress_threshold is not None else DEFAULT_COMPRESS_THRESHOLD
        self.protect_first_n: int = protect_first_n if protect_first_n is not None else DEFAULT_PROTECT_FIRST_N
        self.protect_last_n: int = protect_last_n if protect_last_n is not None else DEFAULT_PROTECT_LAST_N

        self._context: ContextWindow = ContextWindow(max_tokens=self.max_tokens)
        self._compressor: "ContextCompressor | None" = None  # 懒加载

    @property
    def messages(self) -> list[MessageRecord]:
        """当前消息列表"""
        return self._context.messages

    @property
    def total_tokens(self) -> int:
        """当前总 token 数"""
        return self._context.total_tokens

    def _ensure_compressor(self) -> None:
        """懒加载压缩器"""
        if self._compressor is None:
            from agent_material.memory.context_compressor import ContextCompressor
            self._compressor = ContextCompressor()

    def add_message(self, msg: MessageRecord) -> bool:
        """添加消息到短期记忆

        Args:
            msg: 消息记录

        Returns:
            是否触发了压缩
        """
        self._context.messages.append(msg)
        self._context.total_tokens = estimate_messages_tokens(
            [m.to_lc_message() for m in self._context.messages]
        )

        # 检查是否需要压缩
        if self._should_compress():
            self._compress_context()
            return True
        return False

    def add_messages(self, messages: list[MessageRecord]) -> bool:
        """批量添加消息

        与 add_message 的区别：
        - add_message: 逐条添加，每条都检查压缩（适合实时流式对话）
        - add_messages: 批量追加后统一检查压缩（适合加载历史会话），最多触发一次压缩

        Args:
            messages: 消息列表

        Returns:
            是否触发了压缩
        """
        if not messages:
            return False

        # 批量追加所有消息
        self._context.messages.extend(messages)

        # 统一计算总 token（比逐条估算更高效）
        self._context.total_tokens = estimate_messages_tokens(
            [m.to_lc_message() for m in self._context.messages]
        )

        # 统一检查压缩（最多触发一次）
        if self._should_compress():
            self._compress_context()
            return True
        return False

    def _should_compress(self) -> bool:
        """判断是否需要压缩上下文"""
        if self._context.total_tokens < self.max_tokens * self.compress_threshold:
            return False

        # 可压缩区域是否足够大
        compressible: int = len(self._context.messages) - self.protect_first_n - self.protect_last_n
        if compressible < MIN_COMPRESSIBLE_MESSAGES:
            return False

        # 委托压缩器进行反颠簸和冷却检查
        self._ensure_compressor()
        threshold_tokens: int = int(self.max_tokens * self.compress_threshold)
        return self._compressor.should_compress(self._context.total_tokens, threshold_tokens)

    def _compress_context(self) -> None:
        """执行上下文压缩"""
        self._ensure_compressor()

        logger.info(
            f"[ShortTermMemory] 触发压缩 | "
            f"messages={len(self._context.messages)} | "
            f"tokens={self._context.total_tokens}/{self.max_tokens}"
        )

        result: ContextWindow = self._compressor.compress(
            messages=self._context.messages,
            max_tokens=self.max_tokens,
            protect_first_n=self.protect_first_n,
            protect_last_n=self.protect_last_n,
        )

        # 替换为压缩后的消息
        self._context.messages = result.messages
        self._context.total_tokens = result.total_tokens
        self._context.compression_count += 1
        self._context.last_compression_at = time.time()

        logger.info(
            f"[ShortTermMemory] 压缩完成 | "
            f"messages={len(self._context.messages)} | "
            f"tokens={self._context.total_tokens} | "
            f"compressions={self._context.compression_count}"
        )

    def get_context_for_model(self) -> list[dict]:
        """获取用于 LLM 调用的消息列表（LangChain 格式）"""
        return [m.to_lc_message() for m in self._context.messages]

    def clear(self) -> None:
        """清空短期记忆"""
        self._context = ContextWindow(max_tokens=self.max_tokens)
        if self._compressor is not None:
            self._compressor.reset()
        logger.info("[ShortTermMemory] 上下文已清空")

    def fork(self) -> "ShortTermMemory":
        """创建当前上下文的副本（用于子代理 / 压缩会话）"""
        new_stm: ShortTermMemory = ShortTermMemory(
            max_tokens=self.max_tokens,
            compress_threshold=self.compress_threshold,
            protect_first_n=self.protect_first_n,
            protect_last_n=self.protect_last_n,
        )
        new_stm._context.messages = self._context.messages.copy()
        new_stm._context.total_tokens = self._context.total_tokens
        new_stm._context.compression_count = self._context.compression_count
        # 复制压缩器状态（如有）
        if self._compressor is not None:
            from agent_material.memory.context_compressor import ContextCompressor
            new_stm._compressor = ContextCompressor(model=self._compressor.model)
            new_stm._compressor._previous_summary = self._compressor._previous_summary
            new_stm._compressor._ineffective_count = self._compressor._ineffective_count
        return new_stm

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "messages": [m.model_dump() for m in self._context.messages],
            "total_tokens": self._context.total_tokens,
            "max_tokens": self._context.max_tokens,
            "compression_count": self._context.compression_count,
        }
