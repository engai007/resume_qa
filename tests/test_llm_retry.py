import asyncio
import json

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agents import base
from app.agents.planner_agent import PlannerAgent
from app.agents.qa_agent import QAAgent

REQUEST = httpx.Request("POST", "https://router.huggingface.co/v1/chat/completions")


def http_error(status: int, message: str = "error") -> openai.APIStatusError:
    """The error the router's client (openai) raises for an HTTP status, as the real one would."""
    response = httpx.Response(status, request=REQUEST)
    cls = {400: openai.BadRequestError, 401: openai.AuthenticationError, 429: openai.RateLimitError}.get(
        status, openai.InternalServerError if status >= 500 else openai.APIStatusError)
    return cls(message, response=response, body=None)


def timeout() -> openai.APITimeoutError:
    return openai.APITimeoutError(request=REQUEST)


NOT_SERVED = "The requested model 'meta-llama/Llama-3.1-8B-Instruct' is not supported by provider 'novita'."


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
    chat = FlakyChat(timeout(), http_error(504))
    agent = PlannerAgent(chat=chat)  # groq, fallback deepinfra

    raw, stats = call(agent)

    assert raw == '{"ok": true}' and chat.calls == 3
    lines = log_lines(llm_log)
    assert [(l["attempt"], l["provider"], l.get("error")) for l in lines] == [
        (1, "groq", "APITimeoutError"),
        (2, "deepinfra", "InternalServerError 504"),
        (3, "groq", None),
    ]
    assert stats["attempt"] == 3 and lines[-1]["output_tokens"] == 2


def test_model_not_served_moves_to_the_other_provider(llm_log):
    chat = FlakyChat(http_error(400, NOT_SERVED))
    call(PlannerAgent(chat=chat))
    assert chat.calls == 2
    assert [(l["provider"], l.get("error")) for l in log_lines(llm_log)] == [
        ("groq", "BadRequestError 400"), ("deepinfra", None)]


def test_model_not_served_without_another_provider_fails_at_once(llm_log):
    chat = FlakyChat(http_error(400, NOT_SERVED))
    with pytest.raises(openai.BadRequestError):
        call(QAAgent(chat=chat))  # Qwen 72B: no fallback provider, so retrying can't help
    assert chat.calls == 1


def test_real_errors_fail_at_once(llm_log):
    for err in (http_error(401), http_error(400, "bad request: max_tokens too large")):
        chat = FlakyChat(err)
        with pytest.raises(openai.APIStatusError):
            call(PlannerAgent(chat=chat))
        assert chat.calls == 1


def test_gives_up_after_three_tries(llm_log):
    chat = FlakyChat(*[http_error(503)] * 3)
    with pytest.raises(openai.InternalServerError):
        call(QAAgent(chat=chat))
    assert chat.calls == 3
    assert len(log_lines(llm_log)) == 3


def test_no_fallback_provider_means_same_provider(llm_log):
    call(QAAgent(chat=FlakyChat(openai.APIConnectionError(request=REQUEST))))
    assert [l["provider"] for l in log_lines(llm_log)] == ["deepinfra", "deepinfra"]


def test_is_temporary():
    assert base.is_temporary(timeout())
    assert base.is_temporary(openai.APIConnectionError(request=REQUEST))
    assert base.is_temporary(httpx.ReadTimeout("x"))
    assert all(base.is_temporary(http_error(s)) for s in (429, 500, 502, 503, 504))
    assert not any(base.is_temporary(http_error(s)) for s in (400, 401, 403, 404, 422))
    assert not base.is_temporary(ValueError("bad JSON"))


def test_chat_model_goes_to_the_router(monkeypatch):
    from app.llm.huggingface import ROUTER_URL, build_hf_chat
    monkeypatch.setattr(base.settings, "hf_token", "hf_test_not_real")
    chat = build_hf_chat("Qwen/Qwen2.5-72B-Instruct", "deepinfra", 100, 0.01, 30)
    assert chat.model_name == "Qwen/Qwen2.5-72B-Instruct:deepinfra"
    assert str(chat.openai_api_base).rstrip("/") == ROUTER_URL and chat.max_retries == 0
