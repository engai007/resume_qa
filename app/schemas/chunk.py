from datetime import datetime

from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    """Stored next to each chunk in the vector DB, so search results can be filtered and traced back."""
    document_id: int
    document_hash: str
    uploaded_by: int          # user id
    user_email: str
    uploaded_at: datetime
    candidate_name: str | None  # only the profile chunk has the name in its text
    chunk_index: int          # position of the chunk inside the resume


class Chunk(BaseModel):
    text: str
    metadata: ChunkMetadata
