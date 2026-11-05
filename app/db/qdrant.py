from qdrant_client import AsyncQdrantClient

from app.core.config import settings

qdrant = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
