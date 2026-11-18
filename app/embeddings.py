"""Text -> vectors.

- dense:  Hugging Face Inference model (BAAI/bge-base-en-v1.5), 768 numbers that capture *meaning*
- sparse: BM25 with fastembed (Qdrant/bm25), runs locally, a list of {word id: weight} that captures *words*
"""
import asyncio
from functools import cache

from fastembed import SparseTextEmbedding
from huggingface_hub import AsyncInferenceClient
from qdrant_client import models

from app.core.config import settings


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """One dense vector per text, all in one request. Texts over the model's limit (512 tokens) are cut."""
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


# In code, not .env: changing it needs a rebuild migration.
SPARSE_MODEL = "Qdrant/bm25"


@cache
def get_sparse_model() -> SparseTextEmbedding:
    """Loaded once. The first time ever, it downloads a few small files (stop words, stemmer) to the cache."""
    return SparseTextEmbedding(SPARSE_MODEL)


def to_qdrant(e) -> models.SparseVector:
    return models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())


async def sparse_embed_texts(texts: list[str]) -> list[models.SparseVector]:
    """BM25 vector per chunk: word counts + chunk length. Qdrant adds IDF (rare words count more)."""
    embs = await asyncio.to_thread(lambda: list(get_sparse_model().embed(texts)))
    return [to_qdrant(e) for e in embs]


async def sparse_embed_query(query: str) -> models.SparseVector:
    """BM25 vector for a question: each word weight 1. No bge prefix here, or 'represent' would be a keyword."""
    [e] = await asyncio.to_thread(lambda: list(get_sparse_model().query_embed(query)))
    return to_qdrant(e)
