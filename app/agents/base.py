import json
import logging
import time
from datetime import datetime

from langchain_core.messages import SystemMessage

from app.core.config import BASE_DIR, settings
from app.llm.huggingface import build_hf_chat
from app.prompts import load_prompt

logger = logging.getLogger(__name__)


class BaseAgent:
    """Every agent sets `name` and `prompt`, and can override the model settings."""

    name: str
    prompt: str  # file in app/prompts, e.g. "qa_agent/system"
    model_id: str = settings.default_model_id
    provider: str = settings.default_provider
    temperature: float = settings.default_temperature
    max_tokens: int = settings.default_max_tokens
    timeout: int = settings.default_timeout

    def __init__(self, chat=None):
        # `chat` lets tests pass a fake model
        self.chat = chat or build_hf_chat(self.model_id, self.provider, self.max_tokens,
                                          self.temperature, self.timeout)

    async def call_llm(self, messages: list) -> tuple[str, dict]:
        """Call the model with this agent's system prompt and log the call."""
        start = time.perf_counter()
        resp = await self.chat.ainvoke([SystemMessage(content=load_prompt(self.prompt)), *messages])
        usage = resp.usage_metadata or {}
        stats = {
            "agent": self.name,
            "model": self.model_id,
            "provider": self.provider,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "seconds": round(time.perf_counter() - start, 1),
        }
        log_llm_call(stats)
        return resp.content, stats


def log_llm_call(stats: dict) -> None:
    """One line per LLM call in logs/llm_calls.jsonl (no resume text, only numbers)."""
    logger.info("llm call %s", stats)
    path = BASE_DIR / settings.llm_log_file
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), **stats}) + "\n")
