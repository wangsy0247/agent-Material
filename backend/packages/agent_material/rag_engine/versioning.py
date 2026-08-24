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
import re

# 各级 ID 统一取 sha256 hex 的前 32 位，兼顾冲突概率与存储长度
_ID_LEN = 32

# 手工文本来源的逻辑标识前缀（V3 ENT-002）
MANUAL_SOURCE_PREFIX = "manual://"

# 带 scheme 的逻辑标识（如 manual://note_1）—— 非文件路径，不做路径展开
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")


def compute_content_hash(data: bytes) -> str:
    """计算完整内容的 sha256 hex（64 位）。"""
    return hashlib.sha256(data).hexdigest()


def normalize_source_path(source: str) -> str:
    """规范化 source 路径（V2 Spec 6.4）。

    - 带 scheme 的逻辑标识（如 manual://note_1）仅 normcase 原样保留，
      不做路径展开 —— 逻辑标识不得依赖当前工作目录（V3 ENT-002）；
    - 转为绝对路径，避免相对/绝对路径生成两个 document_id；
    - os.path.normcase：Windows 下大小写归一化（Linux 下为 no-op）；
    - 统一路径分隔符为 '/'。
    """
    if _SCHEME_RE.match(source):
        return os.path.normcase(source)
    abs_path = os.path.abspath(os.path.normpath(source))
    return os.path.normcase(abs_path).replace(os.sep, "/")


def _has_path_features(source: str) -> bool:
    """判断 source 是否具有文件路径特征（否则按手工文本逻辑标识处理）。"""
    return (
        os.path.isabs(source)
        or "/" in source
        or "\\" in source
        or source.startswith(".")
        or bool(os.path.splitext(source)[1])
    )


def canonicalize_source(source: str) -> str:
    """统一 source 契约（V3 ENT-002）：入库与删除必须使用同一 canonical source。

    - 已带 scheme 的逻辑标识：normcase 后原样返回；
    - 有路径特征（绝对/相对路径、含分隔符、带扩展名）：规范化绝对路径；
    - 其余视为手工文本来源：加 manual:// 前缀作为显式逻辑标识。
    """
    s = (source or "").strip()
    if not s:
        raise ValueError("source 不能为空")
    if _SCHEME_RE.match(s):
        return os.path.normcase(s)
    if _has_path_features(s):
        return normalize_source_path(s)
    return f"{MANUAL_SOURCE_PREFIX}{s}"


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


def compute_artifact_id(
    document_id: str,
    version_id: str,
    artifact_type: str,
    ordinal: int,
    content_hash: str,
) -> str:
    """生成稳定的图表实体 ID。

    ID 不依赖普通文本切分位置，因此调整父子 chunk 参数不会让同一文档版本中的
    figure/table 身份漂移。
    """
    raw = (
        f"{document_id}:{version_id}:artifact:{artifact_type}:"
        f"{ordinal}:{content_hash}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:_ID_LEN]
