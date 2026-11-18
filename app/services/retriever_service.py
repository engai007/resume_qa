"""Question -> vector -> closest chunks in Qdrant."""
from qdrant_client import models

from app.core.config import settings
from app.db.qdrant import qdrant
from app.embeddings import embed_query


async def search_chunks(user_id: int, query: str, top_k: int = 5,
                        document_hashes: list[str] | None = None) -> list[dict]:
    """Best matching chunks, only from resumes this user uploaded (optionally only the given ones)."""
    must = [models.FieldCondition(key="uploaded_by", match=models.MatchValue(value=user_id))]
    if document_hashes:
        must.append(models.FieldCondition(key="document_hash", match=models.MatchAny(any=document_hashes)))

    response = await qdrant.query_points(
        settings.qdrant_collection,
        query=await embed_query(query),
        query_filter=models.Filter(must=must),
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
        }
        for point in response.points
    ]
