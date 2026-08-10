"""PostgreSQL 控制面（Spec V2 第 5 章）。

PostgreSQL 是唯一持久化真相源，统一承载：
- 任务队列（rag_ingest_tasks）
- 文件状态（rag_file_states）
- 文档 / 版本状态（rag_documents / rag_document_versions）
- 知识库版本号（rag_kb_state）
"""

from agent_material.rag_engine.db.pg.repositories import (
    FileStateRepository,
    KbStateRepository,
    TaskRepository,
    VersionRepository,
)
from agent_material.rag_engine.db.pg.schema import (
    BASE_TABLES,
    drop_schema,
    ensure_schema,
    t,
)

__all__ = [
    "BASE_TABLES",
    "FileStateRepository",
    "KbStateRepository",
    "TaskRepository",
    "VersionRepository",
    "drop_schema",
    "ensure_schema",
    "t",
]
