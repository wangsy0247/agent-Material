"""测试进程的安全、确定性环境配置。"""

import os


# 测试密钥只用于本地临时数据库和短生命周期 token，不用于部署。
os.environ["AUTH_SECRET_KEY"] = (
    "test-only-auth-secret-key-with-at-least-thirty-two-characters"
)
os.environ.setdefault("AGENT_ALLOW_RUNTIME_MCP_CONFIG", "false")
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""
# 模型客户端会在构造阶段校验密钥；这里使用不可用的测试占位值，
# 使请求校验类测试无需真实凭据，也不会因此访问外部模型服务。
os.environ.setdefault("OPENAI_API_KEY", "test-only-openai-key")
os.environ.setdefault("DASHSCOPE_API_KEY", "test-only-dashscope-key")
os.environ.setdefault(
    "POSTGRES_CHECKPOINTER_URL",
    "postgresql://postgres:postgres@127.0.0.1:5433/langgraph",
)
os.environ.setdefault(
    "PGVECTOR_URL",
    "postgresql+psycopg://postgres:postgres@127.0.0.1:5433/postgres",
)
