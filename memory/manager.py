"""
记忆管理器 - 三层记忆的协调中枢

参考 Hermes-Agent MemoryManager 设计：
- 协调短期 / 中期 / 长期记忆
- 提供统一的上下文组装接口
- 会话生命周期管理（按 session_id 隔离短期记忆，支持并发安全）

配置来源：config/memory_config.yaml
"""

import threading
import time
import uuid
from typing import Any

from memory.models import MessageRecord, SessionInfo, MemoryEntry
from memory.short_term import ShortTermMemory
from memory.session_store import SessionStore
from memory.long_term import LongTermMemory
from memory.utils import estimate_messages_tokens
from utils.config_handler import memory_config
from utils.logger import logger

# ---- 配置读取 ----
_MANAGER_CFG: dict[str, Any] = memory_config.get("manager", {})
_LONG_TERM_CFG: dict[str, Any] = memory_config.get("long_term", {})

DEFAULT_MAX_CONTEXT_TOKENS: int = _MANAGER_CFG.get("max_context_tokens", 8000)
DEFAULT_ENABLE_LONG_TERM: bool = _MANAGER_CFG.get("enable_long_term", True)
DEFAULT_ENABLE_COMPRESSION: bool = _MANAGER_CFG.get("enable_compression", True)


class MemoryManager:
    """记忆管理器

    核心职责：
    1. 管理会话生命周期（创建 / 恢复 / 结束）
    2. 协调三层记忆（短期 -> 中期 -> 长期）
    3. 组装 LLM 上下文（系统提示 + 记忆 + 对话历史）
    4. 持久化对话历史

    并发安全设计：
    - 短期记忆按 session_id 隔离（每个会话独立的 ShortTermMemory 实例）
    - session_pool 的操作受 threading.Lock 保护
    - SessionStore(SQLite) 本身支持 WAL 模式多读者单写者
    """

    def __init__(
        self,
        max_context_tokens: int | None = None,
        enable_long_term: bool | None = None,
        enable_compression: bool | None = None,
    ):
        """初始化记忆管理器

        Args:
            max_context_tokens: 上下文窗口最大 Token 预算，None 时读取配置
            enable_long_term: 是否启用长期记忆，None 时读取配置
            enable_compression: 是否启用上下文压缩，None 时读取配置
        """
        self.max_context_tokens: int = max_context_tokens or DEFAULT_MAX_CONTEXT_TOKENS
        self.enable_long_term: bool = enable_long_term if enable_long_term is not None else DEFAULT_ENABLE_LONG_TERM
        self.enable_compression: bool = enable_compression if enable_compression is not None else DEFAULT_ENABLE_COMPRESSION

        # 中期记忆（SQLite 持久化）- 全局共享，线程安全
        self.session_store: SessionStore = SessionStore()
        # 长期记忆（文件 + 向量）- 全局共享
        self.long_term: LongTermMemory | None = LongTermMemory() if self.enable_long_term else None

        # 短期记忆池：session_id → ShortTermMemory（按会话隔离）
        self._session_pool: dict[str, ShortTermMemory] = {}
        self._lock: threading.Lock = threading.Lock()

        # 系统提示（全局共享）
        self._system_prompt: str = ""

    def _get_or_create_stm(self, session_id: str) -> ShortTermMemory:
        """获取或创建指定会话的短期记忆实例"""
        if session_id not in self._session_pool:
            self._session_pool[session_id] = ShortTermMemory(max_tokens=self.max_context_tokens)
        return self._session_pool[session_id]

    # ==================== 会话生命周期 ====================

    def create_session(self, title: str | None = None, source: str = "api") -> str:
        """创建新会话"""
        session_id: str = str(uuid.uuid4())
        self.session_store.create_session(session_id, title=title, source=source)

        with self._lock:
            self._session_pool[session_id] = ShortTermMemory(max_tokens=self.max_context_tokens)

        logger.info(f"[MemoryManager] 创建会话: {session_id}")
        return session_id

    def load_session(self, session_id: str) -> bool:
        """加载已有会话"""
        info: SessionInfo | None = self.session_store.get_session(session_id)
        if not info:
            logger.warning(f"[MemoryManager] 会话不存在: {session_id}")
            return False

        # 恢复消息历史到该会话的短期记忆
        messages: list[MessageRecord] = self.session_store.get_messages(session_id)
        with self._lock:
            stm: ShortTermMemory = ShortTermMemory(max_tokens=self.max_context_tokens)
            if messages:
                stm.add_messages(messages)
            self._session_pool[session_id] = stm

        # 更新会话状态为活跃
        self.session_store.update_session(session_id, is_active=True, end_reason=None)

        logger.info(
            f"[MemoryManager] 恢复会话: {session_id} | "
            f"messages={len(messages)} | tokens={stm.total_tokens}"
        )
        return True

    async def end_session(self, session_id: str, reason: str = "user_close") -> None:
        """结束指定会话"""
        if not session_id:
            return

        # 自动提取长期记忆（从 SQLite 读取，不依赖内存中的 STM）
        if self.enable_long_term and self.long_term:
            await self._auto_extract_memory(session_id)

        # 标记 SQLite 中的会话结束
        self.session_store.end_session(session_id, reason)

        # 从内存池中移除该会话的短期记忆
        with self._lock:
            self._session_pool.pop(session_id, None)

        logger.info(f"[MemoryManager] 结束会话: {session_id}, 原因: {reason}")

    def list_sessions(self, limit: int = 20) -> list[SessionInfo]:
        """列出会话列表"""
        return self.session_store.list_sessions(limit=limit)

    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        # 从内存池中移除
        with self._lock:
            self._session_pool.pop(session_id, None)
        return self.session_store.delete_session(session_id)

    # ==================== 消息管理 ====================

    def add_user_message(self, session_id: str, content: str) -> bool:
        """添加用户消息

        Args:
            session_id: 目标会话 ID
            content: 消息内容

        Returns:
            是否触发了压缩
        """
        msg: MessageRecord = MessageRecord(role="user", content=content, timestamp=time.time())
        return self._add_message(session_id, msg)

    def add_assistant_message(self, session_id: str, content: str) -> bool:
        """添加助手消息

        Args:
            session_id: 目标会话 ID
            content: 消息内容

        Returns:
            是否触发了压缩
        """
        msg: MessageRecord = MessageRecord(role="assistant", content=content, timestamp=time.time())
        return self._add_message(session_id, msg)

    def add_tool_result(self, session_id: str, tool_name: str, result: str) -> bool:
        """添加工具结果

        Args:
            session_id: 目标会话 ID
            tool_name: 工具名称
            result: 工具返回结果

        Returns:
            是否触发了压缩
        """
        msg: MessageRecord = MessageRecord(
            role="tool",
            content=result,
            name=tool_name,
            timestamp=time.time(),
        )
        return self._add_message(session_id, msg)

    def _add_message(self, session_id: str, msg: MessageRecord) -> bool:
        """内部添加消息"""
        stm: ShortTermMemory = self._get_or_create_stm(session_id)

        # 添加到该会话的短期记忆
        compressed: bool = stm.add_message(msg)

        # 持久化到中期记忆
        self.session_store.append_message(session_id, msg)

        return compressed

    # ==================== 上下文组装 ====================

    def build_context(self, session_id: str, query: str | None = None, include_memory: bool = True) -> list[dict]:
        """构建 LLM 上下文

        组装顺序：
        1. 系统提示（原始系统提示 + 记忆上下文）
        2. 对话历史（指定会话的短期记忆中的消息）

        Args:
            session_id: 目标会话 ID
            query: 当前查询，用于召回相关记忆
            include_memory: 是否注入长期记忆

        Returns:
            LangChain 格式的消息列表
        """
        messages: list[dict] = []

        # 1. 系统提示 + 记忆注入
        system_content: str = self._build_system_prompt(query, include_memory)
        if system_content:
            messages.append({"role": "system", "content": system_content})

        # 2. 对话历史（从指定会话的短期记忆中获取）
        stm: ShortTermMemory = self._get_or_create_stm(session_id)
        for msg in stm.messages:
            # 跳过系统消息（已单独处理）
            if msg.role == "system":
                continue
            messages.append(msg.to_lc_message())

        total_tokens: int = estimate_messages_tokens(messages)
        logger.debug(
            f"[MemoryManager] 上下文组装完成 | session={session_id} | "
            f"messages={len(messages)} | tokens={total_tokens}"
        )

        return messages

    def _build_system_prompt(self, query: str | None = None, include_memory: bool = True) -> str:
        """构建系统提示（含记忆注入）"""
        parts: list[str] = []

        if self._system_prompt:
            parts.append(self._system_prompt)

        # 注入长期记忆上下文
        if include_memory and self.enable_long_term and self.long_term and query:
            memory_ctx: str = self.long_term.build_memory_context(query)
            if memory_ctx:
                parts.append(memory_ctx)

        return "\n\n".join(parts) if parts else ""

    def set_system_prompt(self, prompt: str) -> None:
        """设置系统提示"""
        self._system_prompt = prompt

    # ==================== 长期记忆操作 ====================

    def recall_memory(self, query: str, top_k: int | None = None) -> list[MemoryEntry]:
        """召回相关记忆"""
        if not self.long_term:
            return []

        top_k = top_k or _LONG_TERM_CFG.get("recall_top_k", 5)
        return self.long_term.recall(query, top_k=top_k)

    def add_memory(self, entry: MemoryEntry) -> bool:
        """添加长期记忆

        通过 LongTermMemory.store() 统一写入，自动保证：
        - 向量存储与文件存储的一致性
        - 同类别相似记忆的自动去重

        Args:
            entry: 记忆条目

        Returns:
            是否成功写入
        """
        if not self.long_term:
            return False
        return self.long_term.store(entry)

    async def _auto_extract_memory(self, session_id: str) -> None:
        """自动从指定会话提取长期记忆并持久化

        Args:
            session_id: 要提取记忆的会话 ID
        """
        if not (self.long_term and session_id):
            return

        # 获取指定会话的完整消息历史
        messages: list[MessageRecord] = self.session_store.get_messages(session_id)
        if len(messages) < 3:
            logger.debug("[MemoryManager] 对话过短，跳过自动提取")
            return

        # 转换为 dict 格式供提取器使用
        msg_dicts: list[dict] = [m.to_lc_message() for m in messages]

        try:
            entries: list[MemoryEntry] = await self.long_term.extract_from_conversation(
                msg_dicts, source_session_id=session_id
            )
            stored_count: int = 0
            for entry in entries:
                if self.add_memory(entry):
                    stored_count += 1

            # 同时写入每日日志
            info: SessionInfo | None = self.session_store.get_session(session_id)
            if info:
                self.long_term.write_daily_note(
                    session_id,
                    f"会话结束: {info.title or '未命名'} | "
                    f"消息数: {info.message_count} | Token: {info.total_tokens} | "
                    f"提取记忆: {len(entries)} 条, 实际写入: {stored_count} 条",
                )

            logger.info(
                f"[MemoryManager] 自动提取完成: "
                f"提取 {len(entries)} 条, 写入 {stored_count} 条"
            )
        except Exception as e:
            logger.error(f"[MemoryManager] 自动提取记忆失败: {e}")

    # ==================== 工具方法 ====================

    def get_session_stats(self, session_id: str) -> dict:
        """获取指定会话的统计信息

        Args:
            session_id: 目标会话 ID

        Returns:
            会话统计字典
        """
        stm: ShortTermMemory = self._get_or_create_stm(session_id)
        return {
            "messages_in_history": len(stm.messages),
            "total_tokens": stm.total_tokens,
            "max_tokens": stm.max_tokens,
            "compression_count": stm._context.compression_count,
        }

    def get_stats(self) -> dict:
        """获取记忆系统整体统计"""
        stats: dict = {
            "active_sessions_in_memory": len(self._session_pool),
            "session_store": {
                "db_path": self.session_store.db_path,
            },
        }

        if self.long_term:
            stats["long_term"] = {
                "memory_dir": str(self.long_term.memory_dir),
                "user_file_exists": self.long_term.user_file.exists(),
                "project_file_exists": self.long_term.project_file.exists(),
            }

        return stats

    def clear(self) -> None:
        """清空所有记忆（慎用）"""
        with self._lock:
            self._session_pool.clear()
        logger.info("[MemoryManager] 记忆已清空")
