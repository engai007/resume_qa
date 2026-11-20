import asyncio
import json

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError
from langchain_core.messages import AIMessage, HumanMessage

from app.agents import base
from app.agents.qa_agent import QAAgent
from app.agents.resume_parser import ResumeParserAgent


def http_error(status: int) -> HfHubHTTPError:
    response = httpx.Response(status, request=httpx.Request("POST", "https://router.huggingface.co/x"))
    return HfHubHTTPError(f"{status} error", response=response)


class FlakyChat:
    """Raises the given errors in order, then answers. Counts the calls."""

    def __init__(self, *errors):
        self.errors = list(errors)
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return AIMessage(content='{"ok": true}', usage_metadata={"input_tokens": 3, "output_tokens": 2,
                                                                  "total_tokens": 5})


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    monkeypatch.setattr(base, "RETRY_WAITS", [0, 0])


def call(agent):
    return asyncio.run(agent.call_llm([HumanMessage(content="hi")]))


def log_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_timeouts_are_retried_on_the_fallback_provider(llm_log):
    chat = FlakyChat(httpx.ReadTimeout("hung"), http_error(504))
    agent = ResumeParserAgent(chat=chat)  # deepinfra, fallback featherless-ai

    raw, stats = call(agent)

    assert raw == '{"ok": true}' and chat.calls == 3
    lines = log_lines(llm_log)
    assert [(l["attempt"], l["provider"], l.get("error")) for l in lines] == [
        (1, "deepinfra", "ReadTimeout"),
        (2, "featherless-ai", "HfHubHTTPError 504"),
        (3, "deepinfra", None),
    ]
    assert stats["attempt"] == 3 and lines[-1]["output_tokens"] == 2


def test_real_errors_fail_at_once(llm_log):
    chat = FlakyChat(http_error(401))
    with pytest.raises(HfHubHTTPError):
        call(QAAgent(chat=chat))
    assert chat.calls == 1
    [line] = log_lines(llm_log)
    assert line["error"] == "HfHubHTTPError 401"


def test_gives_up_after_three_tries(llm_log):
    chat = FlakyChat(*[http_error(503)] * 3)
    with pytest.raises(HfHubHTTPError):
        call(QAAgent(chat=chat))
    assert chat.calls == 3
    assert len(log_lines(llm_log)) == 3


def test_no_fallback_provider_means_same_provider(llm_log):
    agent = QAAgent(chat=FlakyChat(httpx.ConnectError("down")))
    agent.fallback_provider = ""
    call(agent)
    assert [l["provider"] for l in log_lines(llm_log)] == ["deepinfra", "deepinfra"]


def test_is_temporary():
    assert base.is_temporary(httpx.ReadTimeout("x"))
    assert base.is_temporary(httpx.ConnectError("x"))
    assert all(base.is_temporary(http_error(s)) for s in (429, 500, 502, 503, 504))
    assert not any(base.is_temporary(http_error(s)) for s in (400, 401, 403, 404, 422))
    assert not base.is_temporary(ValueError("bad JSON"))
