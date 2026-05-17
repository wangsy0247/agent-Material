"""Pytest 共享 fixtures。"""

import os
import sys
from unittest.mock import MagicMock

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 若当前环境未安装 langchain 等依赖，先注入 mock 模块避免 ImportError
for _mod_name in [
    "langchain_core",
    "langchain_core.tools",
    "langchain",
    "langchain_anthropic",
    "mp_api",
    "mp_api.client",
    "pymatgen",
    "pymatgen.symmetry",
    "pymatgen.symmetry.analyzer",
    "langchain_chroma",
    "langchain_text_splitters",
    "langchain_community",
    "langchain_core.prompts",
    "langchain_core.output_parsers",
]:
    try:
        __import__(_mod_name)
    except ImportError:
        # 只有当真实导入失败时才注入 mock
        parts = _mod_name.split(".")
        for i in range(1, len(parts) + 1):
            parent = ".".join(parts[:i])
            if parent not in sys.modules:
                sys.modules[parent] = MagicMock()

# 为 langchain_core.tools.tool 提供可用装饰器
_mock_tool_module = sys.modules.setdefault("langchain_core.tools", MagicMock())


def _fake_tool(*args, description=None, args_schema=None, **kwargs):
    """模拟的 tool 装饰器：支持无参或有参调用。"""

    def decorator(func):
        func.description = description or func.__doc__
        func.args_schema = args_schema
        return func

    if len(args) == 1 and callable(args[0]) and not kwargs:
        return decorator(args[0])
    return decorator


_mock_tool_module.tool = _fake_tool

# 为 pydantic 提供最小可用的 BaseModel/Field
try:
    from pydantic import BaseModel, Field  # noqa: F401
except Exception:
    _mock_pydantic = MagicMock()
    _mock_pydantic.BaseModel = MagicMock
    _mock_pydantic.Field = lambda *a, **k: None
    sys.modules.setdefault("pydantic", _mock_pydantic)

# 为 yaml 提供最小可用模块
try:
    import yaml  # noqa: F401
except Exception:
    _mock_yaml = MagicMock()
    _mock_yaml.safe_load = lambda s: s if isinstance(s, dict) else {}
    sys.modules.setdefault("yaml", _mock_yaml)
