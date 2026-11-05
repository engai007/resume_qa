"""Rebuild with BAAI/bge-base-en-v1.5: 768 numbers per vector, reads 512 tokens (MiniLM cut chunks at 256).

1. create resume_chunks_v2 (size 768)
2. re-embed every active document from Postgres into v2 - search keeps using v1 meanwhile
3. move the resume_chunks alias from v1 to v2 in one step

v1 is kept so we can roll back by moving the alias back; delete it in a later migration.
If any document fails, the alias is NOT moved and this migration is not recorded:
fix the problem and run `vector_migrate.py upgrade` again (already indexed documents are just overwritten).
"""
from qdrant_client import AsyncQdrantClient, models
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.models import Document, User
from app.services.indexing_service import index_document

NEW = "resume_chunks_v2"
ALIAS = "resume_chunks"


async def upgrade(client: AsyncQdrantClient, db) -> None:
    if not await client.collection_exists(NEW):
        await client.create_collection(
            NEW,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),  # bge-base-en-v1.5
        )
    await client.create_payload_index(NEW, "uploaded_by", models.PayloadSchemaType.INTEGER)
    await client.create_payload_index(NEW, "document_id", models.PayloadSchemaType.INTEGER)

    rows = (await db.execute(
        select(Document, User).join(User, User.id == Document.generated_by)
        .where(Document.is_active).options(selectinload(Document.meta_data))
    )).all()
    failed = [doc.id for doc, user in rows if not await index_document(db, doc, user, collection=NEW)]
    if failed:
        raise RuntimeError(f"could not index documents {failed} (see documents.index_error). "
                           "Alias not moved, search still uses v1. Fix and run upgrade again.")

    await client.update_collection_aliases(change_aliases_operations=[
        models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=ALIAS)),
        models.CreateAliasOperation(create_alias=models.CreateAlias(collection_name=NEW, alias_name=ALIAS)),
    ])
