"""Fast filter on document_hash, so search can be limited to given resumes."""
from qdrant_client import AsyncQdrantClient, models


async def upgrade(client: AsyncQdrantClient, db) -> None:
    await client.create_payload_index("resume_chunks_v2", "document_hash", models.PayloadSchemaType.KEYWORD)
