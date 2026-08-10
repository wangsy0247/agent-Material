"""确定性 ID 生成模块（OFF-001）。

为文档、版本、父块、子块生成稳定的确定性标识：
- document_id：由规范化 source + 知识库标识生成，内容变化时保持不变。
- version_id：由完整内容 hash 生成，内容变化时必须变化。
- parent_id / chunk_id：链路包含 document_id、version_id、位置与内容 hash，
  保证不同版本、不同位置的块 ID 不冲突。

所有函数均为纯函数 — 相同输入永远生成相同 ID，无随机数、无时间依赖。
"""

import hashlib
import os

# 各级 ID 统一取 sha256 hex 的前 32 位，兼顾冲突概率与存储长度
_ID_LEN = 32


def compute_content_hash(data: bytes) -> str:
    """计算完整内容的 sha256 hex（64 位）。"""
    return hashlib.sha256(data).hexdigest()


def normalize_source_path(source: str) -> str:
    """规范化 source 路径（V2 Spec 6.4）。

    - 转为绝对路径，避免相对/绝对路径生成两个 document_id；
    - os.path.normcase：Windows 下大小写归一化（Linux 下为 no-op）；
    - 统一路径分隔符为 '/'。
    """
    abs_path = os.path.abspath(os.path.normpath(source))
    return os.path.normcase(abs_path).replace(os.sep, "/")


def compute_document_id(source: str, kb_id: str = "agent") -> str:
    """由知识库标识 + 规范化 source 生成 document_id（前 32 位 hex）。

    source 经 normalize_source_path 规范化（绝对路径 + 大小写归一 +
    统一分隔符），避免路径写法差异导致 ID 漂移；
    同一 source 内容变化时 document_id 不变，不同 source 即使内容相同也不同。
    """
    normalized = normalize_source_path(source)
    return hashlib.sha256(f"{kb_id}:{normalized}".encode()).hexdigest()[:_ID_LEN]


def compute_version_id(content_hash: str) -> str:
    """由内容 hash 生成 version_id（前 32 位），内容变化时必须变化。"""
    return content_hash[:_ID_LEN]


def compute_parent_id(
    document_id: str,
    version_id: str,
    position: int,
    content: str,
) -> str:
    """生成父块 ID — 链路包含 document_id、version_id、位置与内容 hash。"""
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    raw = f"{document_id}:{version_id}:{position}:{content_hash}"
    return hashlib.sha256(raw.encode()).hexdigest()[:_ID_LEN]


def compute_chunk_id(
    document_id: str,
    version_id: str,
    parent_id: str,
    position: int,
    content: str,
) -> str:
    """生成子块 ID — 链路在父块基础上额外包含 parent_id。"""
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    raw = f"{document_id}:{version_id}:{parent_id}:{position}:{content_hash}"
    return hashlib.sha256(raw.encode()).hexdigest()[:_ID_LEN]
