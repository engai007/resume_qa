"""Rebuild for new chunking (INDEX_VERSION 3): one chunk per company project, skills per category
in groups of 8. Same model and size as v2 (bge-base-en-v1.5, 768) - only the chunks change.

1. create resume_chunks_v3 with the same payload indexes as v2
2. re-chunk + re-embed every active document from Postgres into v3 - search keeps using v2 meanwhile
3. move the resume_chunks alias from v2 to v3 in one step

v2 is kept for rollback. If any document fails, the alias is NOT moved and this migration is not
recorded: fix the problem and run `vector_migrate.py upgrade` again.
"""
from qdrant_client import AsyncQdrantClient, models
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.models import Document, User
from app.services.indexing_service import index_document

NEW = "resume_chunks_v3"
ALIAS = "resume_chunks"


async def upgrade(client: AsyncQdrantClient, db) -> None:
    if not await client.collection_exists(NEW):
        await client.create_collection(
            NEW,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),  # bge-base-en-v1.5
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
                           "Alias not moved, search still uses v2. Fix and run upgrade again.")

    await client.update_collection_aliases(change_aliases_operations=[
        models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=ALIAS)),
        models.CreateAliasOperation(create_alias=models.CreateAlias(collection_name=NEW, alias_name=ALIAS)),
    ])
