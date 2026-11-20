"""Question -> chunks from Qdrant. Two ways, both inside access_condition(user):

search_chunks: the closest chunks. mode:
- "dense"  : by meaning (bge vector). score = cosine similarity, 0..1
- "sparse" : by keywords (BM25). score = BM25 score, no upper limit
- "hybrid" : both, merged by Qdrant with weighted RRF (Reciprocal Rank Fusion). score = RRF score:
             it only says how well a chunk ranked in both lists, so don't compare it with a threshold

fetch_sections: EVERY chunk of the given sections ("all projects"), no ranking, in resume order.
"""
from typing import Literal

from qdrant_client import models

from app.core.config import settings
from app.db.qdrant import qdrant
from app.embeddings import embed_query, sparse_embed_query
from app.models import User
from app.services.access import access_condition
from app.services.indexing_service import DENSE, SPARSE

Mode = Literal["hybrid", "dense", "sparse"]

CANDIDATES = 20  # hybrid: each search sends its top 20 to the RRF merge, then the final top_k is kept
RRF_WEIGHTS = [2.0, 1.0]  # [dense, bm25]: trust meaning more; BM25 only lifts chunks dense also rates fairly
MAX_SECTION_CHUNKS = 40  # fetch_sections: at most this many chunks go to the LLM (~6k tokens)
SECTION_SCAN_LIMIT = 500  # fetch_sections reads up to this many matches, sorts them, then keeps MAX_SECTION_CHUNKS


def build_filter(user: User, document_hashes: list[str] | None = None,
                 sections: list[str] | None = None) -> models.Filter:
    """Access first (always), then the request's own narrowing. A hash of someone else's resume is ANDed
    with the access condition, so it matches nothing: filters can narrow access, never widen it."""
    must = [access_condition(user)]
    if document_hashes:
        must.append(models.FieldCondition(key="document_hash", match=models.MatchAny(any=document_hashes)))
    if sections:
        must.append(models.FieldCondition(key="section", match=models.MatchAny(any=sections)))
    return models.Filter(must=must)


def to_result(point, score: float | None) -> dict:
    return {
        "score": score,
        "text": point.payload["text"],
        "document_id": point.payload["document_id"],
        "document_hash": point.payload["document_hash"],
        "candidate_name": point.payload.get("candidate_name"),
        "chunk_index": point.payload["chunk_index"],
        "section": point.payload.get("section"),  # None in collections before v5
    }


async def search_chunks(user: User, query: str, top_k: int = 5, document_hashes: list[str] | None = None,
                        mode: Mode = "hybrid", sections: list[str] | None = None) -> list[dict]:
    """Best matching chunks the user may read (optionally only the given resumes / sections)."""
    flt = build_filter(user, document_hashes, sections)

    if mode == "dense":
        response = await qdrant.query_points(settings.qdrant_collection, query=await embed_query(query),
                                             using=DENSE, query_filter=flt, limit=top_k, with_payload=True)
    elif mode == "sparse":
        response = await qdrant.query_points(settings.qdrant_collection, query=await sparse_embed_query(query),
                                             using=SPARSE, query_filter=flt, limit=top_k, with_payload=True)
    else:
        response = await qdrant.query_points(
            settings.qdrant_collection,
            prefetch=[  # run both searches first, each with the same filter
                models.Prefetch(query=await embed_query(query), using=DENSE, filter=flt, limit=CANDIDATES),
                models.Prefetch(query=await sparse_embed_query(query), using=SPARSE, filter=flt, limit=CANDIDATES),
            ],
            query=models.RrfQuery(rrf=models.Rrf(weights=RRF_WEIGHTS)),  # then Qdrant merges the lists by rank
            limit=top_k,
            with_payload=True,
        )
    return [to_result(point, round(point.score, 4)) for point in response.points]


async def fetch_sections(user: User, sections: list[str], document_hashes: list[str] | None = None,
                         limit: int = MAX_SECTION_CHUNKS) -> tuple[list[dict], bool]:
    """Every chunk of `sections` the user may read, in resume order (document, then chunk index).
    No embedding, no ranking. -> (chunks, truncated): truncated=True when more than `limit` matched,
    then the first `limit` in resume order are kept (so whole resumes come first)."""
    if not sections:
        raise ValueError("fetch_sections needs at least one section")  # no sections = the whole resume: not this
    points, _ = await qdrant.scroll(settings.qdrant_collection, scroll_filter=build_filter(user, document_hashes, sections),
                                    limit=SECTION_SCAN_LIMIT, with_payload=True, with_vectors=False)
    points.sort(key=lambda p: (p.payload["document_id"], p.payload["chunk_index"]))
    return [to_result(p, None) for p in points[:limit]], len(points) > limit
