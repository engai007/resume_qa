"""Saved document -> chunks -> vectors -> Qdrant."""
import logging
import uuid

from qdrant_client import models

from app.core.config import settings
from app.db.qdrant import qdrant
from app.embeddings import embed_texts
from app.models import Document, User
from app.services.chunk_service import build_chunks

logger = logging.getLogger(__name__)

# Bump when chunking or the embedding model changes (together with a new vector migration).
# 1 = all-MiniLM-L6-v2 (local), 2 = BAAI/bge-base-en-v1.5 (HF Inference)
INDEX_VERSION = 2


def point_id(document_id: int, chunk_index: int) -> str:
    """Same document + chunk always gives the same id, so re-indexing overwrites instead of duplicating."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"resume-document-{document_id}-chunk-{chunk_index}"))


async def index_document(db, doc: Document, user: User, collection: str | None = None) -> bool:
    """Put one document's chunks into Qdrant and record the result on the document row.
    `collection` defaults to the alias; a rebuild migration passes the new collection (e.g. resume_chunks_v2).
    Never raises: on failure indexed_version is left as it was, so it can be retried."""
    collection = collection or settings.qdrant_collection
    try:
        chunks = build_chunks(doc, user)
        vectors = await embed_texts([c.text for c in chunks])
        points = [
            models.PointStruct(
                id=point_id(doc.id, c.metadata.chunk_index),
                vector=vector,
                payload={"text": c.text, **c.metadata.model_dump(mode="json"),
                         "embedding_model": settings.embedding_model, "index_version": INDEX_VERSION},
            )
            for c, vector in zip(chunks, vectors)
        ]
        # remove old points first, in case the document now has fewer chunks
        await qdrant.delete(collection, points_selector=models.FilterSelector(filter=models.Filter(
            must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=doc.id))])))
        await qdrant.upsert(collection, points=points)
    except Exception as e:
        logger.warning("indexing document %s failed: %s", doc.id, e)
        doc.index_attempts += 1
        doc.index_error = str(e)[:1000]
        await db.commit()
        return False

    doc.indexed_version = INDEX_VERSION
    doc.index_error = None
    await db.commit()
    return True


async def check_vector_db() -> None:
    """Called on app start. Fails if the collection is missing or has the wrong vector size.
    If Qdrant is just down, only warns: uploads still work and are indexed later."""
    name = settings.qdrant_collection
    try:
        exists = await qdrant.collection_exists(name)
    except Exception as e:
        logger.warning("Qdrant not reachable at %s (%s). Uploads will not be indexed.", settings.qdrant_url, e)
        return
    if not exists:
        raise RuntimeError(f"Qdrant collection '{name}' missing. Run: uv run python scripts/vector_migrate.py upgrade")
    size = (await qdrant.get_collection(name)).config.params.vectors.size
    if size != settings.embedding_size:
        raise RuntimeError(f"Qdrant '{name}' has vector size {size}, but EMBEDDING_SIZE is {settings.embedding_size}. "
                           "A model change needs a new vector migration.")
