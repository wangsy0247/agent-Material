"""
上下文压缩器 - LLM 驱动的对话摘要

参考 Hermes-Agent v3 设计：
- Phase 1: 预处理（Smart Collapse + 参数截断）
- Phase 2: 保护头部和尾部
- Phase 3: 结构化 LLM 摘要（首次 / 迭代更新）
- Phase 4: 组装与清洗

配置来源：config/memory_config.yaml -> compression
"""

import json
import time
from typing import Any

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from memory.models import MessageRecord, ContextWindow
from memory.utils import estimate_tokens, estimate_messages_tokens
from model.model_factory import chat_model
from utils.config_handler import memory_config
from utils.logger import logger

# ---- 配置读取 ----
_COMPRESSION_CFG: dict = memory_config.get("compression", {})

# 工具输出替换阈值：超过此长度的工具结果才进行 Smart Collapse
_SMART_COLLAPSE_THRESHOLD: int = 200
# tool_call 参数截断阈值
_TOOL_CALL_ARG_TRUNCATE: int = 500
_TOOL_CALL_ARG_KEEP: int = 200
# 反颠簸：连续低效压缩阈值
_ANTITHRASHING_THRESHOLD: float = 0.10
_ANTITHRASHING_MAX_COUNT: int = 2
# 失败冷却（秒）
_FAILURE_COOLDOWN_CONFIG: int = 600   # 配置类错误
_FAILURE_COOLDOWN_TRANSIENT: int = 60  # 瞬态错误
# 序列化时单条消息最大字符数
_MAX_MESSAGE_COMPRESS_CHARS: int = 3000


# ---- 提示词模板（完全从 memory_config.yaml -> compression 读取） ----
# 请在 config/memory_config.yaml 的 compression 节中配置 prompt_template / prompt_template_update

# 首次压缩模板：优先读配置，缺失时启用极简 fallback 并告警
_PROMPT_FIRST_CFG: str | None = _COMPRESSION_CFG.get("prompt_template")
if _PROMPT_FIRST_CFG and _PROMPT_FIRST_CFG.strip():
    _COMPRESS_PROMPT_TEMPLATE_FIRST = _PROMPT_FIRST_CFG
else:
    logger.warning(
        "[ContextCompressor] memory_config.yaml 中未配置 compression.prompt_template，"
        "将使用内置极简兜底模板。建议在 config/memory_config.yaml 中配置完整提示词。"
    )
    _COMPRESS_PROMPT_TEMPLATE_FIRST = (
        "请对以下对话历史进行结构化摘要，保留关键信息：\n\n"
        "{conversation}\n\n"
        "【摘要】（用中文输出，严格按章节组织，禁止输出前缀）："
    )

# 迭代更新模板：优先读配置，缺失时启用极简 fallback 并告警
_PROMPT_UPDATE_CFG: str | None = _COMPRESSION_CFG.get("prompt_template_update")
if _PROMPT_UPDATE_CFG and _PROMPT_UPDATE_CFG.strip():
    _COMPRESS_PROMPT_TEMPLATE_UPDATE = _PROMPT_UPDATE_CFG
else:
    logger.warning(
        "[ContextCompressor] memory_config.yaml 中未配置 compression.prompt_template_update，"
        "将使用内置极简兜底模板。建议在 config/memory_config.yaml 中配置完整提示词。"
    )
    _COMPRESS_PROMPT_TEMPLATE_UPDATE = (
        "请将以下新对话合并到旧摘要中，保留所有有用信息。\n\n"
        "旧摘要：\n{previous_summary}\n\n"
        "新对话：\n{new_turns}\n\n"
        "【更新后的摘要】（用中文输出，严格按章节组织，禁止输出前缀）："
    )

# 摘要消息标记
SUMMARY_MARKER: str = "【上文摘要】\n"
SUMMARY_END_MARKER: str = "\n【摘要结束，继续对话】"


# ---- Smart Collapse 辅助函数 ----

def _get_tool_collapse_info(tool_name: str | None, args: dict, content_len: int) -> str:
    """根据工具名生成信息化单行摘要"""
    name = tool_name or "unknown"
    
    # RAG 检索
    if name == "rag_summary":
        q = str(args.get("query", ""))[:50] if args else ""
        if q:
            return f"[rag_summary] query='{q}' -> {content_len} chars"
        return f"[rag_summary] -> {content_len} chars"

    # Materials Project 查询
    if name == "mp_query":
        fml = str(args.get("formula", "")) if args else ""
        if fml:
            return f"[mp_query] formula='{fml}' -> {content_len} chars"
        return f"[mp_query] -> {content_len} chars"

    # Materials Project 下载
    if name == "mp_download":
        fml = str(args.get("formula", "")) if args else ""
        if fml:
            return f"[mp_download] formula='{fml}' -> saved, {content_len} chars"
        return f"[mp_download] -> saved, {content_len} chars"

    # Materials Project 列表
    if name == "mp_list":
        return f"[mp_list] -> {content_len} chars"
    
    # MCP 工具通用 fallback
    arg_preview = ""
    if args:
        try:
            arg_preview = json.dumps(args, ensure_ascii=False)[:60]
        except Exception:
            arg_preview = str(args)[:60]
    return f"[{name}] args={arg_preview}... -> {content_len} chars"


class ContextCompressor:
    """上下文压缩器（Hermes v3 风格）

    压缩策略：
    1. Phase 1: 预处理（Smart Collapse + 参数截断）
    2. Phase 2: 分离保护区（头尾）和压缩区（中间）
    3. Phase 3: 结构化 LLM 摘要（首次 / 迭代更新）
    4. Phase 4: 组装与清洗（确保工具对完整性）
    """

    def __init__(self, model=None):
        """初始化压缩器

        Args:
            model: 用于压缩的 LLM 模型，None 时复用主模型
        """
        self.model = model or chat_model
        self._chain_first = None
        self._chain_update = None
        self._previous_summary: str | None = None
        self._ineffective_count: int = 0
        self._cooldown_until: float = 0.0
        self._last_original_tokens: int = 0

    def _get_chain_first(self):
        """懒加载首次压缩链"""
        if self._chain_first is None:
            prompt = PromptTemplate.from_template(_COMPRESS_PROMPT_TEMPLATE_FIRST)
            self._chain_first = prompt | self.model | StrOutputParser()
        return self._chain_first

    def _get_chain_update(self):
        """懒加载迭代更新链"""
        if self._chain_update is None:
            prompt = PromptTemplate.from_template(_COMPRESS_PROMPT_TEMPLATE_UPDATE)
            self._chain_update = prompt | self.model | StrOutputParser()
        return self._chain_update

    def should_compress(self, current_tokens: int, threshold_tokens: int) -> bool:
        """判断是否需要压缩（含反颠簸和冷却检查）

        Args:
            current_tokens: 当前上下文 token 数
            threshold_tokens: 触发压缩的 token 阈值

        Returns:
            是否应当执行压缩
        """
        if current_tokens < threshold_tokens:
            return False

        # 冷却检查
        if time.monotonic() < self._cooldown_until:
            logger.debug("[ContextCompressor] 处于冷却期，跳过压缩")
            return False

        # 反颠簸检查
        if self._ineffective_count >= _ANTITHRASHING_MAX_COUNT:
            logger.warning(
                f"[ContextCompressor] 反颠簸锁定 — 最近 {_ANTITHRASHING_MAX_COUNT} 次压缩节省"
                f" <{_ANTITHRASHING_THRESHOLD * 100:.0f}%。建议清空会话重新开始。"
            )
            return False

        return True

    def compress(
        self,
        messages: list[MessageRecord],
        max_tokens: int = 8000,
        protect_first_n: int = 1,
        protect_last_n: int = 4,
    ) -> ContextWindow:
        """压缩消息列表

        执行四阶段压缩算法：
        Phase 1: 预处理（Smart Collapse + 参数截断）
        Phase 2: 分离保护区（头尾）和压缩区（中间）
        Phase 3: 结构化 LLM 摘要（首次 / 迭代更新）
        Phase 4: 组装与清洗

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

        # Phase 1: 预处理
        processed_messages: list[MessageRecord] = self._preprocess_messages(messages)

        # Phase 2: 分离区域 — 确保最后一条用户消息被保护在尾部
        last_user_idx: int | None = self._find_last_user_message_index(processed_messages)
        if (
            last_user_idx is not None
            and protect_first_n <= last_user_idx < n - protect_last_n
        ):
            # 最后一条 user 在中间区域，扩大尾部保护
            protect_last_n = n - last_user_idx
            logger.info(
                f"[ContextCompressor] 扩大尾部保护至 {protect_last_n} 条以保留最后一条用户消息"
            )

        head: list[MessageRecord] = processed_messages[:protect_first_n]
        tail: list[MessageRecord] = (
            processed_messages[-protect_last_n:] if protect_last_n > 0 else []
        )
        middle: list[MessageRecord] = processed_messages[protect_first_n : n - protect_last_n]

        original_tokens: int = estimate_messages_tokens(
            [m.to_lc_message() for m in processed_messages]
        )

        logger.info(
            f"[ContextCompressor] 开始压缩 | "
            f"total={n} | head={len(head)} | middle={len(middle)} | tail={len(tail)}"
        )

        # Phase 3: 生成结构化摘要
        try:
            if self._previous_summary:
                # 迭代更新：传入旧摘要 + 新轮次
                new_turns_text: str = self._format_messages_for_compression(middle)
                summary: str = self._get_chain_update().invoke(
                    {
                        "previous_summary": self._previous_summary,
                        "new_turns": new_turns_text,
                    }
                )
            else:
                # 首次压缩
                conversation_text: str = self._format_messages_for_compression(middle)
                summary = self._get_chain_first().invoke(
                    {"conversation": conversation_text}
                )

            summary = summary.strip()
            self._previous_summary = summary  # 保存用于下次迭代
            self._cooldown_until = 0.0  # 成功则重置冷却

        except Exception as e:
            logger.error(f"[ContextCompressor] LLM 摘要失败: {e}，回退到简单截断")
            self._set_cooldown(e)
            return self._truncate_tool_results(messages, max_tokens)

        # Phase 4: 组装与清洗
        compressed_messages: list[MessageRecord] = list(head)

        # 插入摘要消息
        summary_msg: MessageRecord = MessageRecord(
            role="system",
            content=f"{SUMMARY_MARKER}{summary}{SUMMARY_END_MARKER}",
            timestamp=time.time(),
            token_count=estimate_tokens(summary)
            + len(SUMMARY_MARKER)
            + len(SUMMARY_END_MARKER),
        )
        compressed_messages.append(summary_msg)
        compressed_messages.extend(tail)

        # 构建结果
        result: ContextWindow = ContextWindow(max_tokens=max_tokens)
        result.messages = compressed_messages
        result.total_tokens = estimate_messages_tokens(
            [m.to_lc_message() for m in compressed_messages]
        )
        result.compression_count = 1
        result.last_compression_at = time.time()

        # 计算节省并更新反颠簸计数
        saved_tokens: int = original_tokens - result.total_tokens
        saved_ratio: float = saved_tokens / max(original_tokens, 1)

        if saved_ratio < _ANTITHRASHING_THRESHOLD:
            self._ineffective_count += 1
            logger.warning(
                f"[ContextCompressor] 压缩效果不佳 ({saved_ratio * 100:.1f}% < "
                f"{_ANTITHRASHING_THRESHOLD * 100:.0f}%)，"
                f"反颠簸计数: {self._ineffective_count}/{_ANTITHRASHING_MAX_COUNT}"
            )
        else:
            self._ineffective_count = 0

        logger.info(
            f"[ContextCompressor] 压缩完成 | "
            f"original_tokens={original_tokens} -> new_tokens={result.total_tokens} | "
            f"saved={saved_tokens} ({saved_ratio * 100:.1f}%) | "
            f"ineffective_count={self._ineffective_count}"
        )

        self._last_original_tokens = original_tokens
        return result

    # ==================== Phase 1: 预处理 ====================

    def _preprocess_messages(
        self, messages: list[MessageRecord]
    ) -> list[MessageRecord]:
        """Phase 1: 预处理（零 LLM Token 成本）

        Pass 1: Smart Collapse — 大型工具输出替换为信息化单行摘要
        Pass 2: 参数截断 — 过长的 tool_call 参数截断
        """
        processed: list[MessageRecord] = []

        for msg in messages:
            new_msg: MessageRecord = MessageRecord(
                role=msg.role,
                content=msg.content,
                tool_call_id=msg.tool_call_id,
                tool_calls=msg.tool_calls,
                name=msg.name,
                timestamp=msg.timestamp,
                token_count=msg.token_count,
            )

            if msg.role == "tool" and msg.content and len(msg.content) > _SMART_COLLAPSE_THRESHOLD:
                # Pass 1: Smart Collapse 工具结果
                new_msg.content = _get_tool_collapse_info(
                    msg.name, {}, len(msg.content)
                )
                new_msg.token_count = estimate_tokens(new_msg.content)

            elif msg.role == "assistant" and msg.tool_calls:
                # Pass 2: 截断过长的 tool_call 参数
                new_msg.tool_calls = self._truncate_tool_call_args(msg.tool_calls)

            processed.append(new_msg)

        return processed

    def _truncate_tool_call_args(self, tool_calls: list[dict] | None) -> list[dict] | None:
        """截断 tool_call 参数（>500 字符截到 200 + 标记）

        Args:
            tool_calls: 原始工具调用列表

        Returns:
            截断后的工具调用列表
        """
        if not tool_calls:
            return tool_calls

        truncated: list[dict] = []
        for tc in tool_calls:
            new_tc: dict = dict(tc)
            func: dict | None = new_tc.get("function")
            if func:
                func = dict(func)
                args = func.get("arguments", "")
                if isinstance(args, str) and len(args) > _TOOL_CALL_ARG_TRUNCATE:
                    func["arguments"] = args[:_TOOL_CALL_ARG_KEEP] + "...[truncated]"
                new_tc["function"] = func
            truncated.append(new_tc)
        return truncated

    # ==================== Phase 2-3: 边界与格式化 ====================

    @staticmethod
    def _find_last_user_message_index(messages: list[MessageRecord]) -> int | None:
        """找到最后一条用户消息的索引"""
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "user":
                return i
        return None

    def _format_messages_for_compression(
        self, messages: list[MessageRecord]
    ) -> str:
        """将消息序列化为带标签的压缩输入文本

        格式：
            [用户] 内容
            [助手] 内容 [Tool calls: tool_name(...)]
            [工具] 内容
        """
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
            if len(content) > _MAX_MESSAGE_COMPRESS_CHARS:
                head_len: int = _MAX_MESSAGE_COMPRESS_CHARS * 2 // 3
                tail_len: int = _MAX_MESSAGE_COMPRESS_CHARS // 4
                content = (
                    content[:head_len]
                    + "\n... [内容过长，已截断] ...\n"
                    + content[-tail_len:]
                )

            # 助手消息附加工具调用信息
            extra: str = ""
            if msg.role == "assistant" and msg.tool_calls:
                tool_names: list[str] = [
                    tc.get("function", {}).get("name", "unknown")
                    for tc in msg.tool_calls
                ]
                extra = f" [Tool calls: {', '.join(tool_names)}]"

            lines.append(f"[{role_label}] {content}{extra}")

        return "\n\n".join(lines)

    # ==================== 容错与回退 ====================

    def _set_cooldown(self, error: Exception) -> None:
        """根据错误类型设置冷却时间

        区分配置类错误（长冷却）和瞬态错误（短冷却），
        让压缩能更快从短暂故障中恢复。
        """
        err_name: str = type(error).__name__
        if err_name in ("RuntimeError", "ConfigurationError", "ValueError"):
            self._cooldown_until = time.monotonic() + _FAILURE_COOLDOWN_CONFIG
            logger.warning(
                f"[ContextCompressor] 配置类错误({err_name})，"
                f"冷却 {_FAILURE_COOLDOWN_CONFIG} 秒"
            )
        else:
            self._cooldown_until = time.monotonic() + _FAILURE_COOLDOWN_TRANSIENT
            logger.warning(
                f"[ContextCompressor] 瞬态错误({err_name})，"
                f"冷却 {_FAILURE_COOLDOWN_TRANSIENT} 秒"
            )

    def reset(self) -> None:
        """重置压缩器状态（用于新会话或 /reset）"""
        self._previous_summary = None
        self._ineffective_count = 0
        self._cooldown_until = 0.0
        self._last_original_tokens = 0
        logger.info("[ContextCompressor] 状态已重置")

    def _truncate_tool_results(
        self, messages: list[MessageRecord], max_tokens: int
    ) -> ContextWindow:
        """回退策略：截断工具结果

        当消息太少不值得压缩，或 LLM 摘要失败时使用。
        """
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

            # 工具结果过长时截断（保留头尾）
            if (
                msg.role == "tool"
                and msg.content
                and len(msg.content) > _MAX_MESSAGE_COMPRESS_CHARS
            ):
                head_len: int = _MAX_MESSAGE_COMPRESS_CHARS // 2
                tail_len: int = _MAX_MESSAGE_COMPRESS_CHARS // 4
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
