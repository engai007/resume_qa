from typing import Literal

from pydantic import BaseModel


class UploadResult(BaseModel):
    filename: str
    status: Literal["created", "already_exists", "failed"]
    document_id: int | None = None
    indexed: bool = False  # chunks are in the vector DB
    error: str | None = None
