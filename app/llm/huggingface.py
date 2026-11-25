"""Chat models through Hugging Face's router (its OpenAI-compatible endpoint).

Why the router directly, not langchain_huggingface / huggingface_hub (used until 2026-11-25): that client first asks
HF's model API which providers serve a model, and refuses to send the request if the provider isn't listed. On
2026-11-25 that list came back EMPTY for every model, so every LLM call failed with
"Model ... is not supported by provider ...", although the router itself still served them.
The router decides on its own, so a provider that really serves the model always works.

Same token (HF_TOKEN), same billing, same models and providers: "<model>:<provider>" picks the provider.
"""
from langchain_openai import ChatOpenAI

from app.core.config import settings

ROUTER_URL = "https://router.huggingface.co/v1"


def build_hf_chat(model_id: str, provider: str, max_tokens: int, temperature: float, timeout: int) -> ChatOpenAI:
    if not settings.hf_token:
        raise RuntimeError("HF_TOKEN is not set. Add it to .env")
    return ChatOpenAI(
        base_url=ROUTER_URL,
        api_key=settings.hf_token,
        model=f"{model_id}:{provider}" if provider else model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        max_retries=0,  # BaseAgent.call_llm retries (and switches provider) itself, and logs every try
    )
