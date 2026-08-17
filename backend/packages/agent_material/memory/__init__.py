"""
企业级三层记忆系统

L1 - 短期记忆 (Short-term): 当前对话上下文窗口管理
L2 - 中期记忆 (Session): SQLite 持久化会话历史
L3 - 长期记忆 (Long-term): 文件记忆 + 向量检索

参考设计:
- Claude-Code: 多级压缩、SessionMemory、记忆目录
- Hermes-Agent: 三层分离、SQLite FTS5、MemoryProvider ABC
- OpenClaw: 显式文件分层、梦境晋升、混合搜索
"""

from agent_material.memory.manager import MemoryManager
from agent_material.memory.models import SessionInfo, MessageRecord

__all__ = ["MemoryManager", "SessionInfo", "MessageRecord"]
