from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.db.database import get_db
from app.models import User

router = APIRouter()


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class SearchResponse(BaseModel):
    query: str
    results: list[dict]


@router.post("/search", response_model=SearchResponse)
async def search(
    body: SearchRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Search the user's uploaded resumes. Not implemented yet: always returns no results."""
    return {"query": body.query, "results": []}
