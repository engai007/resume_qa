"""Rebuild for new chunks (INDEX_VERSION 5), see docs/experience_summary_plan.md:
- every resume gets an "Experience summary" chunk with the total work experience
- personal projects are labelled "Personal project:"
- every chunk gets a `section` payload field (profile, skills, project, ...) with a keyword index,
  so search can filter on it, e.g. all project chunks of one resume

Same vectors as v4 (dense 768 + bm25), only the chunk texts and payload change. A new collection, so search keeps
working on v4 while this runs:
1. create resume_chunks_v5 with the same vectors and payload indexes as v4, plus `section`
2. re-index every active document from Postgres into v5
3. move the resume_chunks alias from v4 to v5 in one step

v4 is kept for rollback. If any document fails, the alias is NOT moved and this migration is not
recorded: fix the problem and run `vector_migrate.py upgrade` again.
"""
from qdrant_client import AsyncQdrantClient, models
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.models import Document, User
from app.services.indexing_service import index_document

NEW = "resume_chunks_v5"
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
    await client.create_payload_index(NEW, "section", models.PayloadSchemaType.KEYWORD)

    rows = (await db.execute(
        select(Document, User).join(User, User.id == Document.generated_by)
        .where(Document.is_active).options(selectinload(Document.meta_data))
    )).all()
    failed = [doc.id for doc, user in rows if not await index_document(db, doc, user, collection=NEW)]
    if failed:
        raise RuntimeError(f"could not index documents {failed} (see documents.index_error). "
                           "Alias not moved, search still uses v4. Fix and run upgrade again.")

    await client.update_collection_aliases(change_aliases_operations=[
        models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=ALIAS)),
        models.CreateAliasOperation(create_alias=models.CreateAlias(collection_name=NEW, alias_name=ALIAS)),
    ])
