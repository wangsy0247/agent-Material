"""
上下文压缩器 - LLM 驱动的对话摘要

参考 Claude-Code 设计：
- 保护头部（系统提示）和尾部（最近对话）消息
- 对中间消息生成结构化摘要
- 使用轻量级模型或主模型进行压缩

配置来源：config/memory_config.yaml -> compression
"""

import time
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from memory.models import MessageRecord, ContextWindow
from memory.utils import estimate_tokens, estimate_messages_tokens, truncate_text
from model.model_factory import chat_model
from utils.config_handler import memory_config
from utils.logger import logger

# ---- 配置读取 ----
_COMPRESSION_CFG: dict = memory_config.get("compression", {})

# 压缩提示词模板（留空则使用内置默认值）
_COMPRESS_PROMPT_TEMPLATE: str = _COMPRESSION_CFG.get(
    "prompt_template",
    """你是一个对话压缩助手。请将以下对话历史压缩成简洁的摘要，保留所有关键信息。

【压缩要求】
1. 保留所有事实性信息、决策、代码修改、错误修复
2. 保留用户明确表达的需求和偏好
3. 保留工具调用的关键结果
4. 去除冗余的寒暄、重复内容
5. 使用第三人称客观描述

【原始对话】
{conversation}

【摘要】（用中文输出，尽量详细但简洁）:""",
)

# 单条消息压缩输入的最大字符数
MAX_MESSAGE_COMPRESS_CHARS: int = 2000

# 摘要消息前缀标记
SUMMARY_MARKER: str = "【上文摘要】\n"
SUMMARY_END_MARKER: str = "\n【摘要结束，继续对话】"


class ContextCompressor:
    """上下文压缩器

    压缩策略：
    1. 分离保护区（头尾）和压缩区（中间）
    2. 对压缩区生成结构化摘要
    3. 将摘要作为 system 消息插入
    """

    def __init__(self, model=None):
        """初始化压缩器

        Args:
            model: 用于压缩的 LLM 模型，None 时复用主模型
        """
        self.model = model or chat_model
        self._chain = None

    def _get_chain(self):
        """懒加载压缩链"""
        if self._chain is None:
            prompt = PromptTemplate.from_template(_COMPRESS_PROMPT_TEMPLATE)
            self._chain = prompt | self.model | StrOutputParser()
        return self._chain

    def compress(
        self,
        messages: list[MessageRecord],
        max_tokens: int = 8000,
        protect_first_n: int = 1,
        protect_last_n: int = 4,
    ) -> ContextWindow:
        """压缩消息列表

        Args:
            messages: 原始消息列表
            max_tokens: 压缩后目标最大 token 数
            protect_first_n: 保留前 N 条消息（通常是系统提示）
            protect_last_n: 保留后 N 条消息（最近对话）

        Returns:
            压缩后的 ContextWindow
        """
        n: int = len(messages)

        if n <= protect_first_n + protect_last_n + 2:
            # 消息太少，不值得压缩，直接截断工具结果
            return self._truncate_tool_results(messages, max_tokens)

        # 分离区域
        head: list[MessageRecord] = messages[:protect_first_n]
        tail: list[MessageRecord] = messages[-protect_last_n:] if protect_last_n > 0 else []
        middle: list[MessageRecord] = messages[protect_first_n : n - protect_last_n]

        original_tokens: int = estimate_messages_tokens([m.to_lc_message() for m in messages])

        logger.info(
            f"[ContextCompressor] 开始压缩 | "
            f"total={n} | head={len(head)} | middle={len(middle)} | tail={len(tail)}"
        )

        # 构建压缩区对话文本
        conversation_text: str = self._format_messages_for_compression(middle)

        # 生成摘要
        try:
            summary: str = self._get_chain().invoke({"conversation": conversation_text})
            summary = summary.strip()
        except Exception as e:
            logger.error(f"[ContextCompressor] LLM 摘要失败: {e}，回退到简单截断")
            return self._truncate_tool_results(messages, max_tokens)

        # 构建压缩后的消息列表
        compressed_messages: list[MessageRecord] = list(head)

        # 插入摘要消息
        summary_msg: MessageRecord = MessageRecord(
            role="system",
            content=f"{SUMMARY_MARKER}{summary}{SUMMARY_END_MARKER}",
            timestamp=time.time(),
            token_count=estimate_tokens(summary) + len(SUMMARY_MARKER) + len(SUMMARY_END_MARKER),
        )
        compressed_messages.append(summary_msg)

        # 添加尾部
        compressed_messages.extend(tail)

        # 构建结果
        result: ContextWindow = ContextWindow(max_tokens=max_tokens)
        result.messages = compressed_messages
        result.total_tokens = estimate_messages_tokens(
            [m.to_lc_message() for m in compressed_messages]
        )
        result.compression_count = 1
        result.last_compression_at = time.time()

        saved_tokens: int = original_tokens - result.total_tokens
        logger.info(
            f"[ContextCompressor] 压缩完成 | "
            f"original_tokens={original_tokens} -> new_tokens={result.total_tokens} | "
            f"saved={saved_tokens} ({saved_tokens / max(original_tokens, 1) * 100:.1f}%)"
        )

        return result

    def _format_messages_for_compression(self, messages: list[MessageRecord]) -> str:
        """将消息格式化为压缩输入文本"""
        lines: list[str] = []
        for msg in messages:
            role_label: str = {
                "user": "用户",
                "assistant": "助手",
                "tool": "工具",
                "system": "系统",
            }.get(msg.role, msg.role)
            content: str = msg.content or ""
            # 截断过长的单条消息
            if len(content) > MAX_MESSAGE_COMPRESS_CHARS:
                content = content[:MAX_MESSAGE_COMPRESS_CHARS] + "... [内容过长，已截断]"
            lines.append(f"[{role_label}] {content}")
        return "\n\n".join(lines)

    def _truncate_tool_results(
        self, messages: list[MessageRecord], max_tokens: int
    ) -> ContextWindow:
        """回退策略：截断工具结果"""
        result: ContextWindow = ContextWindow(max_tokens=max_tokens)
        result.messages = []

        for msg in messages:
            new_msg: MessageRecord = MessageRecord(
                role=msg.role,
                content=msg.content,
                tool_call_id=msg.tool_call_id,
                tool_calls=msg.tool_calls,
                name=msg.name,
                timestamp=msg.timestamp,
            )

            # 工具结果过长时截断
            if msg.role == "tool" and msg.content and len(msg.content) > MAX_MESSAGE_COMPRESS_CHARS:
                head_len: int = MAX_MESSAGE_COMPRESS_CHARS // 2
                tail_len: int = MAX_MESSAGE_COMPRESS_CHARS // 4
                new_msg.content = (
                    msg.content[:head_len]
                    + "\n... [中间内容省略] ...\n"
                    + msg.content[-tail_len:]
                )
                new_msg.token_count = estimate_tokens(new_msg.content)

            result.messages.append(new_msg)

        result.total_tokens = estimate_messages_tokens(
            [m.to_lc_message() for m in result.messages]
        )
        logger.info(f"[ContextCompressor] 使用工具截断回退 | tokens={result.total_tokens}")
        return result
