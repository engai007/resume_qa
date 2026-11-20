import asyncio
import json
import logging
import time
from datetime import datetime

import httpx
from huggingface_hub.errors import HfHubHTTPError, InferenceTimeoutError
from langchain_core.messages import SystemMessage

from app.core.config import BASE_DIR, settings
from app.llm.huggingface import build_hf_chat
from app.prompts import load_prompt

logger = logging.getLogger(__name__)

RETRY_WAITS = [2, 5]  # seconds to wait before try 2 and try 3, so at most 3 tries
TEMPORARY_STATUS = {429, 500, 502, 503, 504}  # busy / gateway errors: worth trying again


def is_temporary(e: Exception) -> bool:
    """True for errors that may work on the next try: timeouts, dropped connections, 429/5xx.
    False for real errors (401 bad token, 400 bad request, ...): retrying won't help."""
    if isinstance(e, (httpx.TransportError, InferenceTimeoutError, asyncio.TimeoutError)):  # timeouts, connection
        return True
    if isinstance(e, HfHubHTTPError):
        return e.response is not None and e.response.status_code in TEMPORARY_STATUS
    return False


class BaseAgent:
    """Every agent sets `name` and `prompt`, and can override the model settings."""

    name: str
    prompt: str  # file in app/prompts, e.g. "resume_parser/system"
    model_id: str = settings.default_model_id
    provider: str = settings.default_provider
    fallback_provider: str = settings.default_fallback_provider  # "" = none; same model on another provider
    temperature: float = settings.default_temperature
    max_tokens: int = settings.default_max_tokens
    timeout: int = settings.default_timeout

    def __init__(self, chat=None):
        # `chat` lets tests pass a fake model (then used for every try)
        self._fake_chat = chat
        self._chats = {}

    def providers_for_tries(self) -> list[str]:
        """Provider of each try: main, fallback, main. Without a fallback: main 3 times.
        Switching on try 2 means a provider that hangs is skipped right away."""
        n = len(RETRY_WAITS) + 1
        if not self.fallback_provider or self.fallback_provider == self.provider:
            return [self.provider] * n
        return [self.provider if i % 2 == 0 else self.fallback_provider for i in range(n)]

    def chat_for(self, provider: str):
        if self._fake_chat is not None:
            return self._fake_chat
        if provider not in self._chats:  # built on first use, so an unused fallback costs nothing
            self._chats[provider] = build_hf_chat(self.model_id, provider, self.max_tokens,
                                                  self.temperature, self.timeout)
        return self._chats[provider]

    async def call_llm(self, messages: list) -> tuple[str, dict]:
        """Call the model with this agent's system prompt and log the call.
        Temporary errors (timeout, 502/503/504, ...) are tried again, see RETRY_WAITS and providers_for_tries.
        Every try is logged, failed ones with "error"; the last error is raised."""
        messages = [SystemMessage(content=load_prompt(self.prompt)), *messages]
        providers = self.providers_for_tries()
        for attempt, provider in enumerate(providers, start=1):
            start = time.perf_counter()
            stats = {"agent": self.name, "model": self.model_id, "provider": provider, "attempt": attempt}
            try:
                resp = await self.chat_for(provider).ainvoke(messages)
            except Exception as e:
                stats.update(seconds=round(time.perf_counter() - start, 1), error=error_name(e))
                log_llm_call(stats)
                if attempt == len(providers) or not is_temporary(e):
                    raise
                await asyncio.sleep(RETRY_WAITS[attempt - 1])
                continue
            usage = resp.usage_metadata or {}
            stats.update(input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
                         seconds=round(time.perf_counter() - start, 1))
            log_llm_call(stats)
            return resp.content, stats


def error_name(e: Exception) -> str:
    """"ReadTimeout", "HfHubHTTPError 504": short, and never the message (it can echo the request)."""
    status = getattr(getattr(e, "response", None), "status_code", None)
    return f"{type(e).__name__} {status}" if status else type(e).__name__


def log_llm_call(stats: dict) -> None:
    """One line per LLM try in logs/llm_calls.jsonl (no resume text, only numbers)."""
    logger.info("llm call %s", stats)
    path = BASE_DIR / settings.llm_log_file
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), **stats}) + "\n")
