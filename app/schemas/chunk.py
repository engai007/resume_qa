from datetime import datetime

from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    """Stored next to each chunk in the vector DB, so search results can be filtered and traced back."""
    document_id: int
    document_hash: str
    uploaded_by: int          # user id
    user_email: str
    uploaded_at: datetime
    chunk_index: int          # position of the chunk inside the resume


class Chunk(BaseModel):
    text: str
    metadata: ChunkMetadata
