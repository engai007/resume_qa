"""Rebuild for hybrid search (INDEX_VERSION 4): each point gets two named vectors.

- "dense": bge-base-en-v1.5, 768, cosine (same model as v3)
- "bm25":  sparse keyword vector from fastembed Qdrant/bm25; modifier=IDF makes Qdrant
           work out "rare words count more" itself, across the whole collection

A collection's vector layout can't change in place (one unnamed vector -> two named ones), so:
1. create resume_chunks_v4 with the same payload indexes as v3
2. re-index every active document from Postgres into v4 - search keeps using v3 meanwhile
3. move the resume_chunks alias from v3 to v4 in one step

v3 is kept for rollback. If any document fails, the alias is NOT moved and this migration is not
recorded: fix the problem and run `vector_migrate.py upgrade` again.
"""
from qdrant_client import AsyncQdrantClient, models
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.models import Document, User
from app.services.indexing_service import index_document

NEW = "resume_chunks_v4"
ALIAS = "resume_chunks"


async def upgrade(client: AsyncQdrantClient, db) -> None:
    if not await client.collection_exists(NEW):
        await client.create_collection(
            NEW,
            vectors_config={"dense": models.VectorParams(size=768, distance=models.Distance.COSINE)},
            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
    await client.create_payload_index(NEW, "uploaded_by", models.PayloadSchemaType.INTEGER)
    await client.create_payload_index(NEW, "document_id", models.PayloadSchemaType.INTEGER)
    await client.create_payload_index(NEW, "document_hash", models.PayloadSchemaType.KEYWORD)

    rows = (await db.execute(
        select(Document, User).join(User, User.id == Document.generated_by)
        .where(Document.is_active).options(selectinload(Document.meta_data))
    )).all()
    failed = [doc.id for doc, user in rows if not await index_document(db, doc, user, collection=NEW)]
    if failed:
        raise RuntimeError(f"could not index documents {failed} (see documents.index_error). "
                           "Alias not moved, search still uses v3. Fix and run upgrade again.")

    await client.update_collection_aliases(change_aliases_operations=[
        models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=ALIAS)),
        models.CreateAliasOperation(create_alias=models.CreateAlias(collection_name=NEW, alias_name=ALIAS)),
    ])
