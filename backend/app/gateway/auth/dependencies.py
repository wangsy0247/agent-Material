"""
依赖注入模块
使用懒加载模式管理 Agent、RAG 和记忆系统实例，避免启动时初始化重量级对象

重要：本模块不在顶层导入任何重量级依赖（如 PyTorch、PGVector、Embedding 模型等），
所有导入均在函数内部延迟执行，防止 WSL / 多进程环境下的 Segmentation Fault。
"""

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_material.agents.React_agent import ReactAgent
    from agent_material.rag.rag_service import RagService
    from agent_material.memory.manager import MemoryManager
    from .schemas import UserInfo


@lru_cache(maxsize=1)
def get_agent() -> "ReactAgent":
    """获取 Agent 实例（单例，懒加载）

    LRU 缓存确保整个应用生命周期中只有一个 Agent 实例
    首次调用时才会触发重量级对象的初始化

    注意：默认不绑定记忆管理器，避免循环依赖。
    需要记忆功能时，通过 get_agent_with_memory() 获取。
    """
    from agent_material.agents.React_agent import ReactAgent
    return ReactAgent()


@lru_cache(maxsize=1)
def get_agent_with_memory() -> "ReactAgent":
    """获取集成记忆系统的 Agent 实例（全局单例）

    自动将 MemoryManager 注入 Agent，实现：
    - 对话历史自动保存
    - 长期记忆注入系统提示
    - Token 超限自动压缩

    注意：Agent 实例本身是全局共享的，多租户隔离通过方法参数中的
    user_id 实现（如 excutor_stream(..., user_id=...)）。
    """
    from agent_material.agents.React_agent import ReactAgent
    mm = get_memory_manager()
    return ReactAgent(memory_manager=mm)


@lru_cache(maxsize=1)
def get_rag_service() -> "RagService":
    """获取 RAG 服务实例（单例，懒加载）

    LRU 缓存确保整个应用生命周期中只有一个 RAGService 实例
    首次调用时才会初始化 VectorStore、重排序模型等
    """
    from agent_material.rag.rag_service import RagService
    return RagService()


@lru_cache(maxsize=1)
def get_memory_manager() -> "MemoryManager":
    """获取记忆管理器实例（单例，懒加载）

    三层记忆系统：
    - L1 短期记忆：对话上下文窗口管理
    - L2 中期记忆：SQLite 会话持久化
    - L3 长期记忆：文件 + 向量检索

    注意：MemoryManager 内部按 user_id 隔离数据，
    但实例本身是全局共享的（通过方法参数传入 user_id 实现隔离）。
    """
    from agent_material.memory.manager import MemoryManager
    return MemoryManager()


# ---- 认证依赖 ----
from .auth import get_current_user
