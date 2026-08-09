"""会话标题与删除的单元测试。

覆盖：_make_session_title 截断规则、_is_default_title 判定、
_ensure_session 的首问标题替换逻辑（新建/默认标题/已有标题三态）。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.gateway.routers.agent import (  # noqa: E402
    _ensure_session,
    _is_default_title,
    _make_session_title,
)


class TestMakeSessionTitle:
    def test_plain(self):
        assert _make_session_title("什么是深度势能？") == "什么是深度势能？"

    def test_whitespace_compressed(self):
        assert _make_session_title("你好\n  世界\t hello") == "你好 世界 hello"

    def test_truncation(self):
        q = "这是一个非常非常长的问题，超过了二十个字符的限制会被截断"
        title = _make_session_title(q)
        assert len(title) == 21  # 20 字 + 省略号
        assert title.endswith("…")
        assert title == q[:20] + "…"

    def test_exact_limit_no_ellipsis(self):
        q = "a" * 20
        assert _make_session_title(q) == q

    def test_empty(self):
        assert _make_session_title("   ") == "新会话"


class TestIsDefaultTitle:
    def test_defaults(self):
        assert _is_default_title(None)
        assert _is_default_title("")
        assert _is_default_title("会话 abcd1234")
        assert _is_default_title("Agent 对话")
        assert _is_default_title("新会话")

    def test_custom_title_kept(self):
        assert not _is_default_title("深度势能讨论")
        assert not _is_default_title("会话记录整理")  # 用户自命名，前缀不同


# ------------------------------------------------------------------
# _ensure_session 首问标题替换
# ------------------------------------------------------------------

class FakeSessionInfo:
    def __init__(self, title, message_count):
        self.title = title
        self.message_count = message_count


class FakeSessionStore:
    def __init__(self, info=None):
        self.info = info
        self.updates = []

    def get_session(self, session_id, user_id=None):
        return self.info

    def update_session(self, session_id, user_id=None, **kwargs):
        self.updates.append(kwargs)
        return True


class FakeMM:
    def __init__(self, in_pool=False, store_info=None, load_ok=False):
        self._session_pool = {("u1", "s-exists"): object()} if in_pool else {}
        self.session_store = FakeSessionStore(store_info)
        self._load_ok = load_ok
        self.created_with = None

    def load_session(self, session_id, user_id=None):
        return self._load_ok

    def create_session(self, user_id, title):
        self.created_with = title
        return "s-new"


class TestEnsureSessionTitle:
    def test_new_session_uses_title_hint(self):
        mm = FakeMM()
        sid = _ensure_session(mm, None, "u1", title_hint="什么是深度势能？")
        assert sid == "s-new"
        assert mm.created_with == "什么是深度势能？"

    def test_new_session_fallback_title(self):
        mm = FakeMM()
        _ensure_session(mm, None, "u1")
        assert mm.created_with == "Agent 对话"

    def test_pool_default_title_first_message_renamed(self):
        mm = FakeMM(in_pool=True, store_info=FakeSessionInfo("新会话", 0))
        sid = _ensure_session(mm, "s-exists", "u1", title_hint="首问标题")
        assert sid == "s-exists"
        assert mm.session_store.updates[-1].get("title") == "首问标题"

    def test_pool_default_title_not_first_message_kept(self):
        mm = FakeMM(in_pool=True, store_info=FakeSessionInfo("新会话", 4))
        _ensure_session(mm, "s-exists", "u1", title_hint="不应采用")
        assert "title" not in mm.session_store.updates[-1]

    def test_pool_custom_title_kept(self):
        mm = FakeMM(in_pool=True, store_info=FakeSessionInfo("深度势能讨论", 0))
        _ensure_session(mm, "s-exists", "u1", title_hint="不应采用")
        assert "title" not in mm.session_store.updates[-1]

    def test_loaded_session_first_message_renamed(self):
        mm = FakeMM(store_info=FakeSessionInfo("会话 abc12345", 0), load_ok=True)
        _ensure_session(mm, "s-loaded", "u1", title_hint="加载后首问")
        assert mm.session_store.updates[-1].get("title") == "加载后首问"

    def test_unknown_session_created_with_hint(self):
        mm = FakeMM(load_ok=False)
        sid = _ensure_session(mm, "s-unknown", "u1", title_hint="新标题")
        assert sid == "s-new"
        assert mm.created_with == "新标题"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
