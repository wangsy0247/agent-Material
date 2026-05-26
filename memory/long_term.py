"""
长期记忆 - 跨会话知识存储与召回（按 user_id 隔离）

参考 OpenClaw + Hermes-Agent 设计：
- 显式文件分层：{user_id}/USER.md / {user_id}/MEMORY.md
- 复用现有 ChromaDB 做向量索引（按 user_id metadata 过滤）
- 自动记忆提取（对话结束后）

配置来源：config/memory_config.yaml -> long_term
"""

import os
import time
from pathlib import Path

from langchain_core.documents import Document

from memory.models import MemoryEntry
from memory.utils import estimate_tokens, format_timestamp, truncate_text
from utils.config_handler import memory_config
from utils.path_tools import get_abs_path
from utils.logger import logger

# ---- 配置读取 ----
_LONG_TERM_CFG: dict = memory_config.get("long_term", {})

DEFAULT_MEMORY_DIR: str = _LONG_TERM_CFG.get("memory_dir", "workspace/memory")
DEFAULT_ENABLE_VECTOR: bool = _LONG_TERM_CFG.get("enable_vector_memory", True)
DEFAULT_VECTOR_COLLECTION: str = _LONG_TERM_CFG.get("vector_collection", "long_term_memory")
DEFAULT_RECALL_TOP_K: int = _LONG_TERM_CFG.get("recall_top_k", 5)
DEFAULT_RECALL_THRESHOLD: float = _LONG_TERM_CFG.get("recall_score_threshold", 0.1)
DEFAULT_MAX_CONTEXT_CHARS: int = _LONG_TERM_CFG.get("max_memory_context_chars", 1000)
DEFAULT_DEDUP_THRESHOLD: float = _LONG_TERM_CFG.get("dedup_threshold", 0.92)

# 文件记忆配置
USER_FILE_NAME: str = _LONG_TERM_CFG.get("user_file", "USER.md")
PROJECT_FILE_NAME: str = _LONG_TERM_CFG.get("project_file", "MEMORY.md")
DAILY_DIR_NAME: str = _LONG_TERM_CFG.get("daily_dir", "daily")

# 向量检索：Chroma cosine distance -> similarity 转换
COSINE_DISTANCE_TO_SIMILARITY: float = 0.5


class LongTermMemory:
    """长期记忆管理器（按 user_id 隔离）

    存储方式：
    1. 文件记忆：{memory_dir}/{user_id}/USER.md（用户画像）、MEMORY.md（项目知识）
    2. 向量记忆：通过 ChromaDB 存储结构化记忆条目（带 user_id metadata）
    """

    def __init__(self, user_id: str = "default", memory_dir: str | None = None):
        """初始化长期记忆

        Args:
            user_id: 用户ID，用于数据隔离
            memory_dir: 记忆根目录，None 时读取配置
        """
        self.user_id: str = user_id
        self.memory_dir: Path = Path(memory_dir) if memory_dir else Path(get_abs_path(DEFAULT_MEMORY_DIR))
        self.user_memory_dir: Path = self.memory_dir / user_id
        self.user_memory_dir.mkdir(parents=True, exist_ok=True)

        self.user_file: Path = self.user_memory_dir / USER_FILE_NAME
        self.project_file: Path = self.user_memory_dir / PROJECT_FILE_NAME
        self.daily_dir: Path = self.user_memory_dir / DAILY_DIR_NAME
        self.daily_dir.mkdir(exist_ok=True)

        # 向量存储（复用现有 ChromaDB，使用独立 collection）
        self._vector_store = None
        self._enable_vector: bool = DEFAULT_ENABLE_VECTOR

        logger.info(f"[LongTermMemory] 初始化 | user={user_id} | 目录: {self.user_memory_dir}")

    # ==================== 文件记忆读写 ====================

    def _read_file(self, path: Path) -> str:
        """读取记忆文件"""
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"[LongTermMemory] 读取文件失败 {path}: {e}")
            return ""

    def _write_file(self, path: Path, content: str) -> None:
        """写入记忆文件"""
        try:
            path.write_text(content, encoding="utf-8")
        except Exception as e:
            logger.error(f"[LongTermMemory] 写入文件失败 {path}: {e}")

    def get_user_profile(self) -> str:
        """获取用户画像"""
        return self._read_file(self.user_file)

    def get_project_memory(self) -> str:
        """获取项目记忆"""
        return self._read_file(self.project_file)

    def _fact_exists_in_file(self, path: Path, fact: str) -> bool:
        """检查事实是否已存在于文件中（忽略时间戳和格式标记）"""
        if not path.exists():
            return False
        content: str = self._read_file(path)
        # 提取纯文本内容（去掉 markdown 标记和时间戳）进行比较
        stripped_fact: str = fact.strip()
        return stripped_fact in content

    def add_user_fact(self, fact: str) -> None:
        """添加用户相关事实（自动去重）"""
        if self._fact_exists_in_file(self.user_file, fact):
            logger.debug(f"[LongTermMemory] 用户事实已存在，跳过: {truncate_text(fact, 50)}")
            return

        current: str = self.get_user_profile()
        timestamp: str = format_timestamp()
        new_entry: str = f"\n- [{timestamp}] {fact}"

        if not current.strip():
            new_content: str = f"# 用户画像\n{new_entry}"
        else:
            new_content = current + new_entry

        self._write_file(self.user_file, new_content)
        logger.info(f"[LongTermMemory] 添加用户事实: {truncate_text(fact, 50)}")

    def add_project_fact(self, fact: str) -> None:
        """添加项目相关事实（自动去重）"""
        if self._fact_exists_in_file(self.project_file, fact):
            logger.debug(f"[LongTermMemory] 项目事实已存在，跳过: {truncate_text(fact, 50)}")
            return

        current: str = self.get_project_memory()
        timestamp: str = format_timestamp()
        new_entry: str = f"\n- [{timestamp}] {fact}"

        if not current.strip():
            new_content: str = f"# 项目记忆\n{new_entry}"
        else:
            new_content = current + new_entry

        self._write_file(self.project_file, new_content)
        logger.info(f"[LongTermMemory] 添加项目事实: {truncate_text(fact, 50)}")

    def write_daily_note(self, session_id: str, summary: str) -> None:
        """写入每日会话摘要"""
        from datetime import datetime

        date_str: str = datetime.now().strftime("%Y-%m-%d")
        file_path: Path = self.daily_dir / f"{date_str}.md"

        timestamp: str = format_timestamp()
        entry: str = f"\n## 会话 {session_id[:8]} ({timestamp})\n{summary}\n"

        if file_path.exists():
            content: str = file_path.read_text(encoding="utf-8")
        else:
            content = f"# 记忆日志 {date_str}\n"

        file_path.write_text(content + entry, encoding="utf-8")

    # ==================== 向量记忆 ====================

    def _ensure_vector_store(self) -> None:
        """确保向量存储初始化"""
        if self._vector_store is not None:
            return

        from langchain_chroma import Chroma
        from model.model_factory import embedding_model
        from utils.config_handler import chroma_config

        persist_dir: str = chroma_config.get("persist_directory", "chroma_db")
        self._vector_store = Chroma(
            collection_name=DEFAULT_VECTOR_COLLECTION,
            embedding_function=embedding_model,
            persist_directory=persist_dir,
        )
        logger.info("[LongTermMemory] 向量存储初始化完成")

    def add_memory_entry(self, entry: MemoryEntry, dedup_threshold: float | None = None) -> bool:
        """添加记忆条目到向量存储（带自动去重，按 user_id 隔离）

        Args:
            entry: 记忆条目
            dedup_threshold: 去重相似度阈值，None 时使用配置默认值

        Returns:
            是否成功写入（False 表示被去重跳过）
        """
        if not self._enable_vector:
            return False

        threshold: float = dedup_threshold if dedup_threshold is not None else DEFAULT_DEDUP_THRESHOLD
        self._ensure_vector_store()

        # ---- 去重检查：用内容本身查询同类别、同用户记忆 ----
        try:
            dup_results: list[tuple[Document, float]] = self._vector_store.similarity_search_with_score(
                entry.content,
                k=1,
                filter={"$and": [{"category": entry.category}, {"user_id": self.user_id}]},
            )
            if dup_results:
                _, score = dup_results[0]
                similarity: float = 1.0 - score * COSINE_DISTANCE_TO_SIMILARITY
                if similarity >= threshold:
                    logger.info(
                        f"[LongTermMemory] 跳过重复向量记忆 "
                        f"(sim={similarity:.3f}): {truncate_text(entry.content, 40)}"
                    )
                    return False
        except Exception as e:
            logger.debug(f"[LongTermMemory] 去重检查异常，继续写入: {e}")

        doc: Document = Document(
            page_content=entry.content,
            metadata={
                "user_id": self.user_id,
                "category": entry.category,
                "source_session_id": entry.source_session_id,
                "created_at": entry.created_at,
                "confidence": entry.confidence,
                "tags": ",".join(entry.tags),
            },
        )

        self._vector_store.add_documents([doc])
        logger.info(f"[LongTermMemory] 向量记忆条目添加: {truncate_text(entry.content, 50)}")
        return True

    def recall(
        self,
        query: str,
        category: str | None = None,
        top_k: int | None = None,
    ) -> list[MemoryEntry]:
        """召回相关记忆（按 user_id 过滤）

        混合检索：
        1. 向量相似度搜索（带 user_id 过滤）
        2. 关键词过滤（category）

        Args:
            query: 查询内容
            category: 记忆类别过滤
            top_k: 召回数量，None 时读取配置

        Returns:
            记忆条目列表，按置信度降序排列
        """
        if not self._enable_vector:
            return []

        self._ensure_vector_store()
        top_k = top_k or DEFAULT_RECALL_TOP_K

        # 构建过滤条件：必须匹配当前 user_id
        filter_dict = {"user_id": self.user_id}
        try:
            results: list[tuple[Document, float]] = self._vector_store.similarity_search_with_score(
                query,
                k=top_k * 4,  # 多取一些，过滤后可能不够
                filter=filter_dict,
            )
        except Exception as e:
            logger.warning(f"[LongTermMemory] 向量检索失败: {e}")
            return []

        entries: list[MemoryEntry] = []
        for doc, score in results:
            # 校验 user_id（双重保险）
            if doc.metadata.get("user_id") != self.user_id:
                continue

            # 如果有 category 过滤
            if category and doc.metadata.get("category") != category:
                continue

            # 转换 distance 到 similarity（Chroma cosine distance）
            similarity: float = 1.0 - score * COSINE_DISTANCE_TO_SIMILARITY

            # 阈值过滤
            if similarity < DEFAULT_RECALL_THRESHOLD:
                continue

            entries.append(
                MemoryEntry(
                    content=doc.page_content,
                    category=doc.metadata.get("category", "fact"),
                    source_session_id=doc.metadata.get("source_session_id"),
                    created_at=doc.metadata.get("created_at", 0),
                    confidence=similarity,
                    tags=(
                        doc.metadata.get("tags", "").split(",")
                        if doc.metadata.get("tags")
                        else []
                    ),
                )
            )

        # 按置信度排序，取 top_k
        entries.sort(key=lambda x: x.confidence, reverse=True)
        return entries[:top_k]

    def store(self, entry: MemoryEntry) -> bool:
        """统一存储记忆条目（向量 + 文件双轨，带自动去重）

        这是写入长期记忆的推荐入口，自动保证：
        1. 向量存储与文件存储的一致性
        2. 同类别相似记忆的自动去重

        Args:
            entry: 记忆条目

        Returns:
            是否成功写入（任一渠道写入成功即返回 True）
        """
        vector_stored: bool = False
        file_stored: bool = False

        # 1. 写入向量存储（内部自带去重）
        if self._enable_vector:
            vector_stored = self.add_memory_entry(entry)

        # 2. 写入分类文件（内部自带去重）
        if entry.category == "user":
            self.add_user_fact(entry.content)
            file_stored = True
        elif entry.category == "project":
            self.add_project_fact(entry.content)
            file_stored = True

        # fact / task 类不强制写入文件，仅保留在向量库中便于检索
        return vector_stored or file_stored

    # ==================== 记忆注入系统提示 ====================

    def build_memory_context(self, query: str | None = None) -> str:
        """构建记忆上下文块，用于注入系统提示

        返回格式化的记忆文本，包含：
        - 用户画像摘要
        - 项目知识摘要
        - 相关记忆召回
        """
        sections: list[str] = []

        # 1. 用户画像
        user_profile: str = self.get_user_profile()
        if user_profile:
            sections.append(f"【用户画像】\n{self._summarize_md(user_profile, DEFAULT_MAX_CONTEXT_CHARS // 2)}")

        # 2. 项目知识
        project_mem: str = self.get_project_memory()
        if project_mem:
            sections.append(f"【项目知识】\n{self._summarize_md(project_mem, DEFAULT_MAX_CONTEXT_CHARS // 2)}")

        # 3. 相关记忆召回（如果有查询）
        if query:
            recalled: list[MemoryEntry] = self.recall(query, top_k=3)
            if recalled:
                mem_lines: list[str] = []
                for e in recalled:
                    mem_lines.append(f"- [{e.category}] {e.content}")
                sections.append("【相关记忆】\n" + "\n".join(mem_lines))

        if not sections:
            return ""

        return "\n\n".join(sections)

    def _summarize_md(self, text: str, max_chars: int) -> str:
        """简单截断 Markdown 内容"""
        lines: list[str] = text.strip().split("\n")
        result: list[str] = []
        current_len: int = 0

        for line in lines:
            if current_len + len(line) > max_chars:
                result.append("... [更多内容省略]")
                break
            result.append(line)
            current_len += len(line) + 1

        return "\n".join(result)

    # ==================== 自动提取 ====================

    async def extract_from_conversation(
        self, messages: list[dict], source_session_id: str | None = None
    ) -> list[MemoryEntry]:
        """从对话历史自动提取长期记忆（异步）

        设计参考 Claude-Code 的 extractMemories：
        - 提取用户偏好、关键决策、重要事实
        - 返回结构化记忆条目
        """
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from model.model_factory import chat_model

        extract_prompt: str = """请从以下对话中提取值得长期记忆的信息。
只提取事实、用户偏好、关键决策，不提取临时性内容。

【对话历史】
{conversation}

【提取格式】每行一条，格式：类别 | 内容
类别可选：user(用户偏好), project(项目知识), fact(事实), task(任务)

【提取结果】:"""

        # 构建对话文本（只取最近 20 条）
        recent: list[dict] = messages[-20:] if len(messages) > 20 else messages
        conversation: str = "\n".join(
            [f"[{m.get('role')}] {str(m.get('content', ''))[:200]}" for m in recent]
        )

        try:
            chain = (
                PromptTemplate.from_template(extract_prompt)
                | chat_model
                | StrOutputParser()
            )
            result: str = await chain.ainvoke({"conversation": conversation})

            entries: list[MemoryEntry] = []
            for line in result.strip().split("\n"):
                line = line.strip()
                if "|" in line:
                    parts: list[str] = line.split("|", 1)
                    category: str = parts[0].strip().lower()
                    content: str = parts[1].strip()
                    if content and category in ("user", "project", "fact", "task"):
                        entries.append(
                            MemoryEntry(
                                category=category,
                                content=content,
                                source_session_id=source_session_id,
                            )
                        )

            return entries
        except Exception as e:
            logger.error(f"[LongTermMemory] 自动提取失败: {e}")
            return []
