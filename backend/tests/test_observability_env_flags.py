from agent_material.utils import langfuse_setup


def test_langfuse_explicit_false_skips_initialization(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "should-not-be-used")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "should-not-be-used")

    assert langfuse_setup.setup_langfuse() is False
    assert langfuse_setup.is_tracing_enabled() is False
