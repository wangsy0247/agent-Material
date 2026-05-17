"""
记忆系统数据模型 (Pydantic V2)
兼容 LangChain message 格式
"""

from pydantic import BaseModel, Field
from typing import Literal


class MessageRecord(BaseModel):
    """单条消息记录（兼容 LangChain/OpenAI 格式）"""
    role: Literal["system", "user", "assistant", "tool"] = Field(..., description="消息角色")
    content: str = Field(default="", description="消息内容")
    tool_call_id: str | None = Field(default=None, description="工具调用ID（tool角色时）")
    tool_calls: list[dict] | None = Field(default=None, description="工具调用列表（assistant角色时）")
    name: str | None = Field(default=None, description="工具名称")
    timestamp: float | None = Field(default=None, description="时间戳")
    token_count: int | None = Field(default=None, description="估算token数")
    
    def to_lc_message(self) -> dict:
        """转换为 LangChain 消息格式"""
        msg = {"role": self.role, "content": self.content}
        if self.name:
            msg["name"] = self.name
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg
    
    @classmethod
    def from_lc_message(cls, msg: dict, timestamp: float | None = None) -> "MessageRecord":
        """从 LangChain 消息格式创建"""
        from memory.utils import estimate_tokens
        return cls(
            role=msg.get("role", "user"),
            content=msg.get("content", "") or "",
            name=msg.get("name"),
            tool_call_id=msg.get("tool_call_id"),
            tool_calls=msg.get("tool_calls"),
            timestamp=timestamp,
            token_count=estimate_tokens(msg.get("content", "") or ""),
        )


class SessionInfo(BaseModel):
    """会话元数据"""
    session_id: str = Field(..., description="会话唯一ID")
    title: str | None = Field(default=None, description="会话标题")
    created_at: float = Field(default_factory=lambda: __import__("time").time(), description="创建时间")
    updated_at: float = Field(default_factory=lambda: __import__("time").time(), description="最后更新时间")
    message_count: int = Field(default=0, description="消息数量")
    total_tokens: int = Field(default=0, description="总token数")
    source: str = Field(default="api", description="来源（api/cli等）")
    is_active: bool = Field(default=True, description="是否活跃")
    end_reason: str | None = Field(default=None, description="结束原因")
    parent_session_id: str | None = Field(default=None, description="父会话ID（压缩分叉时）")
    metadata: dict = Field(default_factory=dict, description="额外元数据")


class MemoryEntry(BaseModel):
    """长期记忆条目"""
    id: str | None = Field(default=None, description="记忆ID")
    category: Literal["user", "project", "fact", "preference", "task"] = Field(..., description="记忆类别")
    content: str = Field(..., description="记忆内容")
    source_session_id: str | None = Field(default=None, description="来源会话ID")
    created_at: float = Field(default_factory=lambda: __import__("time").time(), description="创建时间")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="置信度")
    tags: list[str] = Field(default_factory=list, description="标签")


class CompressionSummary(BaseModel):
    """上下文压缩摘要结果"""
    summary: str = Field(..., description="摘要内容")
    original_messages_count: int = Field(default=0, description="原始消息数")
    original_tokens: int = Field(default=0, description="原始token数")
    preserved_head: int = Field(default=0, description="保留头部消息数")
    preserved_tail: int = Field(default=0, description="保留尾部消息数")
    compressed_at: float = Field(default_factory=lambda: __import__("time").time(), description="压缩时间")


class ContextWindow(BaseModel):
    """上下文窗口状态"""
    messages: list[MessageRecord] = Field(default_factory=list, description="当前消息列表")
    total_tokens: int = Field(default=0, description="当前总token数")
    max_tokens: int = Field(default=8000, description="最大token预算")
    compression_count: int = Field(default=0, description="已压缩次数")
    last_compression_at: float | None = Field(default=None, description="上次压缩时间")
