"""Create resume_chunks_v1 and point the resume_chunks alias at it.

Values are written out (not read from settings) on purpose: a migration must do
the same thing in every environment, forever - like an Alembic file.
"""
from qdrant_client import AsyncQdrantClient, models

COLLECTION = "resume_chunks_v1"
ALIAS = "resume_chunks"


async def upgrade(client: AsyncQdrantClient, db) -> None:
    if not await client.collection_exists(COLLECTION):
        await client.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),  # all-MiniLM-L6-v2
        )
    # fast filters, e.g. "only chunks uploaded by user 1"
    await client.create_payload_index(COLLECTION, "uploaded_by", models.PayloadSchemaType.INTEGER)
    await client.create_payload_index(COLLECTION, "document_id", models.PayloadSchemaType.INTEGER)
    await client.update_collection_aliases(change_aliases_operations=[
        models.CreateAliasOperation(create_alias=models.CreateAlias(collection_name=COLLECTION, alias_name=ALIAS)),
    ])
