"""Text -> vectors with a Hugging Face Inference embedding model (not an LLM: returns numbers, not text)."""
from huggingface_hub import AsyncInferenceClient

from app.core.config import settings


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """One vector per text, all in one request. Texts over the model's limit (512 tokens) are cut."""
    if not settings.hf_token:
        raise RuntimeError("HF_TOKEN is not set in .env")
    client = AsyncInferenceClient(provider=settings.embedding_provider, api_key=settings.hf_token,
                                  timeout=settings.embedding_timeout)
    vectors = await client.feature_extraction(texts, model=settings.embedding_model, truncate=True)
    return vectors.tolist()


# bge models want this in front of search questions (not in front of the stored chunks)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


async def embed_query(query: str) -> list[float]:
    [vector] = await embed_texts([QUERY_PREFIX + query])
    return vector
