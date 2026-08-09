"""
API 数据模型 (Pydantic V2)
定义所有请求和响应的数据结构
"""

from pydantic import BaseModel, Field
from typing import Literal, Any


# ==================== 通用模型 ====================

class BaseResponse(BaseModel):
    """通用响应基类"""
    code: int = Field(default=200, description="状态码")
    message: str = Field(default="success", description="状态信息")
    data: Any = Field(default=None, description="响应数据")
    total: int | None = Field(default=None, description="结果总数（列表类响应可选）")


class HealthResponse(BaseResponse):
    """健康检查响应"""
    data: dict = Field(default_factory=dict, description="服务状态详情")


# ==================== Agent 相关模型 ====================

class AgentChatRequest(BaseModel):
    """Agent 对话请求"""
    question: str = Field(..., min_length=1, max_length=10000, description="用户问题")
    session_id: str = Field(default="default", description="会话ID，用于区分不同对话")
    stream: bool = Field(default=True, description="是否使用 SSE 流式输出")


class AgentChatResponse(BaseModel):
    """Agent 对话非流式响应"""
    answer: str = Field(..., description="Agent 完整回答")
    session_id: str = Field(..., description="会话ID")


class AgentApprovalRequest(BaseModel):
    """HITL 审批请求"""
    session_id: str = Field(..., description="会话ID（与中断时一致）")
    action: Literal["approve", "reject"] = Field(..., description="审批动作")
    comment: str | None = Field(default=None, description="审批备注/拒绝原因")


# ==================== RAG 相关模型 ====================

class RAGQueryRequest(BaseModel):
    """RAG 查询请求"""
    question: str = Field(..., min_length=1, max_length=10000, description="用户问题")
    top_k: int = Field(default=5, ge=1, le=20, description="检索文档数量")
    stream: bool = Field(default=True, description="是否使用 SSE 流式输出")


class RAGRetrieveRequest(BaseModel):
    """RAG 仅检索文档请求（非流式）"""
    question: str = Field(..., min_length=1, max_length=10000, description="用户问题")
    top_k: int = Field(default=5, ge=1, le=20, description="检索文档数量")


class DocumentSource(BaseModel):
    """文档来源信息"""
    index: int = Field(..., description="序号")
    source: str = Field(..., description="来源文件名")
    score: float = Field(default=0.0, description="相似度分数")
    snippet: str = Field(..., description="内容片段")


class RAGRetrieveResponse(BaseResponse):
    """RAG 检索响应"""
    data: list[DocumentSource] = Field(default_factory=list, description="检索到的文档列表")
    total: int = Field(default=0, description="文档总数")


class RAGQueryResponse(BaseModel):
    """RAG 查询非流式响应"""
    answer: str = Field(..., description="生成的回答")
    sources: list[DocumentSource] = Field(default_factory=list, description="引用来源")


# ==================== SSE 事件模型 ====================

class SSEEvent(BaseModel):
    """SSE 事件通用结构

    事件分三类（对齐 docs/FASTAPI 分类法）：
    - 状态类：start / status / thinking
    - 内容类：token / tool_call / tool_result / sources
    - 终结类：done / error / cancelled / approval_required（互斥为终态）
    """
    type: Literal[
        "start",              # 开始处理
        "status",             # 状态更新（记忆/上下文/阶段 phase）
        "thinking",           # Agent 思考过程
        "tool_call",          # 工具调用
        "tool_result",        # 工具返回结果
        "token",              # 大模型生成 token（带 index 序号）
        "answer",             # 完整答案片段（deprecated：后端无发射点，前端仅兜底）
        "sources",            # RAG 引用来源
        "approval_required",  # HITL 敏感操作需人工确认
        "error",              # 错误
        "done",               # 正常完成
        "cancelled",          # 用户中止/断连导致的取消（终态）
    ] = Field(..., description="事件类型")
    content: str | dict | list | None = Field(default=None, description="事件内容")
    index: int | None = Field(default=None, description="token 事件序号（run 内从 1 递增，用于乱序/丢帧检测）")
    timestamp: float | None = Field(default=None, description="时间戳")


# ==================== 记忆系统模型 ====================

class SessionCreateRequest(BaseModel):
    """创建会话请求"""
    title: str | None = Field(default=None, description="会话标题")
    source: str = Field(default="api", description="来源")

class SessionInfoResponse(BaseModel):
    """会话信息"""
    session_id: str
    title: str | None
    created_at: float
    updated_at: float
    message_count: int
    total_tokens: int
    source: str
    is_active: bool
    end_reason: str | None

class SessionListResponse(BaseResponse):
    """会话列表响应"""
    data: list[SessionInfoResponse] = Field(default_factory=list)
    total: int = Field(default=0)

class MemoryRecallRequest(BaseModel):
    """记忆召回请求"""
    query: str = Field(..., min_length=1, description="查询内容")
    category: str | None = Field(default=None, description="记忆类别过滤")
    top_k: int = Field(default=5, ge=1, le=20)

class MemoryEntryResponse(BaseModel):
    """记忆条目响应"""
    category: str
    content: str
    confidence: float
    created_at: float | None
    tags: list[str]

class MemoryRecallResponse(BaseResponse):
    """记忆召回响应"""
    data: list[MemoryEntryResponse] = Field(default_factory=list)

class MemoryStatsResponse(BaseResponse):
    """记忆系统统计"""
    data: dict = Field(default_factory=dict)

# ==================== 系统统计模型 ====================

class RAGStatsResponse(BaseResponse):
    """RAG 统计信息响应"""
    data: dict = Field(default_factory=dict, description="统计详情")


# ==================== MCP 配置模型 ====================

class MCPServerConfig(BaseModel):
    """单个 MCP Server 配置（与 extensions_config.json 对齐）"""
    enabled: bool = Field(default=True, description="是否启用")
    type: str = Field(default="stdio", description="传输类型：stdio / sse / http")
    command: str | None = Field(default=None, description="stdio 模式主命令")
    args: list[str] | None = Field(default=None, description="stdio 模式参数列表")
    url: str | None = Field(default=None, description="sse / http 模式 URL")
    env: dict[str, str] | None = Field(default=None, description="环境变量")
    cwd: str | None = Field(default=None, description="工作目录")
    timeout: int = Field(default=60, description="工具调用超时（秒）")
    auth: dict | None = Field(default=None, description="认证配置（Bearer / OAuth）")


class MCPConfigResponse(BaseResponse):
    """MCP 配置查询响应"""
    data: dict = Field(default_factory=dict, description="mcpServers 配置字典")


class MCPConfigUpdateRequest(BaseModel):
    """MCP 配置更新请求"""
    mcpServers: dict[str, MCPServerConfig] = Field(default_factory=dict)
    mcpInterceptors: list[str] = Field(default_factory=list)
    skills: dict = Field(default_factory=dict)


class MCPStatusResponse(BaseResponse):
    """MCP 连接状态响应"""
    data: dict = Field(default_factory=dict, description="各 Server 健康状态")
