"""Question -> closest chunks in Qdrant.

mode:
- "dense"  : by meaning (bge vector). score = cosine similarity, 0..1
- "sparse" : by keywords (BM25). score = BM25 score, no upper limit
- "hybrid" : both, merged by Qdrant with weighted RRF (Reciprocal Rank Fusion). score = RRF score:
             it only says how well a chunk ranked in both lists, so don't compare it with a threshold
"""
from typing import Literal

from qdrant_client import models

from app.core.config import settings
from app.db.qdrant import qdrant
from app.embeddings import embed_query, sparse_embed_query
from app.services.indexing_service import DENSE, SPARSE

Mode = Literal["hybrid", "dense", "sparse"]

CANDIDATES = 20  # hybrid: each search sends its top 20 to the RRF merge, then the final top_k is kept
RRF_WEIGHTS = [2.0, 1.0]  # [dense, bm25]: trust meaning more; BM25 only lifts chunks dense also rates fairly


async def search_chunks(user_id: int, query: str, top_k: int = 5,
                        document_hashes: list[str] | None = None, mode: Mode = "hybrid") -> list[dict]:
    """Best matching chunks, only from resumes this user uploaded (optionally only the given ones)."""
    must = [models.FieldCondition(key="uploaded_by", match=models.MatchValue(value=user_id))]
    if document_hashes:
        must.append(models.FieldCondition(key="document_hash", match=models.MatchAny(any=document_hashes)))
    flt = models.Filter(must=must)

    if mode == "dense":
        response = await qdrant.query_points(settings.qdrant_collection, query=await embed_query(query),
                                             using=DENSE, query_filter=flt, limit=top_k, with_payload=True)
    elif mode == "sparse":
        response = await qdrant.query_points(settings.qdrant_collection, query=await sparse_embed_query(query),
                                             using=SPARSE, query_filter=flt, limit=top_k, with_payload=True)
    else:
        response = await qdrant.query_points(
            settings.qdrant_collection,
            prefetch=[  # run both searches first, each with the user filter
                models.Prefetch(query=await embed_query(query), using=DENSE, filter=flt, limit=CANDIDATES),
                models.Prefetch(query=await sparse_embed_query(query), using=SPARSE, filter=flt, limit=CANDIDATES),
            ],
            query=models.RrfQuery(rrf=models.Rrf(weights=RRF_WEIGHTS)),  # then Qdrant merges the lists by rank
            limit=top_k,
            with_payload=True,
        )
    return [
        {
            "score": round(point.score, 4),
            "text": point.payload["text"],
            "document_id": point.payload["document_id"],
            "document_hash": point.payload["document_hash"],
            "candidate_name": point.payload.get("candidate_name"),
            "chunk_index": point.payload["chunk_index"],
            "section": point.payload.get("section"),  # None in collections before v5
        }
        for point in response.points
    ]
