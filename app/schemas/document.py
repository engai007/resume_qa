from typing import Literal

from pydantic import BaseModel


class UploadResult(BaseModel):
    filename: str
    status: Literal["created", "already_exists", "failed"]
    document_id: int | None = None
    error: str | None = None
