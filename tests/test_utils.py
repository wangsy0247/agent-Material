"""测试 utils 模块中的通用工具函数。"""

import os
import pytest
from utils.path_tools import get_project_path, get_abs_path


class TestPathTools:
    """测试路径工具函数。"""

    def test_get_project_path_returns_existing_dir(self):
        """get_project_path 应返回项目根目录，且该目录存在。"""
        path = get_project_path()
        assert os.path.isdir(path)
        # 项目根目录下应有 config、agent、rag 等子目录
        assert os.path.isdir(os.path.join(path, "config"))
        assert os.path.isdir(os.path.join(path, "agent"))

    def test_get_abs_path_with_relative_path(self):
        """get_abs_path 应将相对路径正确转换为绝对路径。"""
        rel_path = "config/agent_config.yaml"
        abs_path = get_abs_path(rel_path)

        assert os.path.isabs(abs_path)
        assert abs_path.endswith("config" + os.sep + "agent_config.yaml")
        assert os.path.exists(abs_path)

    def test_get_abs_path_with_nested_relative_path(self):
        """测试多层嵌套相对路径。"""
        rel_path = "agent/tools/mp_downloader.py"
        abs_path = get_abs_path(rel_path)

        assert os.path.isabs(abs_path)
        assert os.path.exists(abs_path)

    def test_get_abs_path_with_os_specific_separator(self):
        """测试使用操作系统路径分隔符。"""
        # 使用 os.path.join 构造跨平台路径
        rel_path = os.path.join("data", "领域分类.md")
        abs_path = get_abs_path(rel_path)

        assert os.path.isabs(abs_path)
        assert os.path.exists(abs_path)
