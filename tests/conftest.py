import pytest

from app.core.config import settings


@pytest.fixture(autouse=True)
def llm_log(tmp_path, monkeypatch):
    """Tests write LLM logs to a temp file, not logs/llm_calls.jsonl."""
    path = tmp_path / "llm_calls.jsonl"
    monkeypatch.setattr(settings, "llm_log_file", str(path))
    return path
