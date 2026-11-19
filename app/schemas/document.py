from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class UploadResult(BaseModel):
    filename: str
    status: Literal["created", "already_exists", "failed"]
    document_id: int | None = None
    indexed: bool = False  # chunks are in the vector DB
    error: str | None = None


class DocumentOut(BaseModel):
    document_id: int
    document_hash: str
    candidate_name: str | None  # from the parsed resume
    uploaded_at: datetime
    indexed: bool  # chunks are in the vector DB with the current INDEX_VERSION, so it's searchable
    indexed_version: int | None
    index_error: str | None  # why indexing last failed, if it did


class DeleteResult(BaseModel):
    document_id: int
    status: Literal["deleted"]
