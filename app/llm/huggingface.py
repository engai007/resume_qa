from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from app.core.config import settings


def build_hf_chat(model_id: str, provider: str, max_tokens: int,
                  temperature: float, timeout: int) -> ChatHuggingFace:
    if not settings.hf_token:
        raise RuntimeError("HF_TOKEN is not set. Add it to .env")
    endpoint = HuggingFaceEndpoint(
        repo_id=model_id,
        task="text-generation",
        provider=provider,
        max_new_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        huggingfacehub_api_token=settings.hf_token,
    )
    return ChatHuggingFace(llm=endpoint)
