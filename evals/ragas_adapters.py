"""Our judge LLM and bge embeddings, in the shapes RAGAS 0.4 metrics accept. See docs/ragas_eval_plan.md §4, §12.

- judge:      RAGAS wants an *instructor* LLM (structured output into its own pydantic models), made by llm_factory
              from an OpenAI client. The HF router is OpenAI-compatible, so: same URL, same HF_TOKEN as the agents.
- embeddings: RAGAS wants a BaseRagasEmbedding (embed_text / aembed_text). Ours are plain async functions.
"""
import asyncio
import os

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")  # before ragas is imported: no usage data sent to RAGAS

from openai import AsyncOpenAI  # noqa: E402
from ragas.embeddings.base import BaseRagasEmbedding  # noqa: E402
from ragas.llms.base import llm_factory  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.embeddings import embed_texts  # noqa: E402
from app.llm.huggingface import ROUTER_URL  # noqa: E402


def judge_model_name(provider: str | None = None) -> str:
    """"openai/gpt-oss-120b:cerebras": the router picks the provider from the suffix, like the agents."""
    return f"{settings.judge_model_id}:{provider or settings.judge_provider}"


def make_judge(provider: str | None = None):
    if not settings.hf_token:
        raise RuntimeError("HF_TOKEN is not set. Add it to .env")
    client = AsyncOpenAI(base_url=ROUTER_URL, api_key=settings.hf_token, timeout=settings.judge_timeout,
                         max_retries=1)
    return llm_factory(judge_model_name(provider), client=client,
                       max_tokens=settings.judge_max_tokens, temperature=0.0)


class BgeEmbeddings(BaseRagasEmbedding):
    """Our bge model for answer relevancy. No bge query prefix: it compares the question with questions written
    from the answer, so both sides must be embedded the same way."""

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        [vector] = await embed_texts([text])
        return vector

    async def aembed_texts(self, texts: list[str], **kwargs) -> list[list[float]]:
        return await embed_texts(texts)  # one request for all, not one per text

    def embed_text(self, text: str, **kwargs) -> list[float]:
        return asyncio.run(self.aembed_text(text))
