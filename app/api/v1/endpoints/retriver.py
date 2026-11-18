from fastapi import APIRouter, Depends

from app.api.deps import get_current_user
from app.models import User
from app.schemas.retriever import SearchRequest, SearchResponse
from app.services.retriever_service import search_chunks

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(body: SearchRequest, user: User = Depends(get_current_user)):
    """Find the resume chunks closest to the query. Only searches resumes you uploaded;
    pass document_hashes to search only those resumes."""
    results = await search_chunks(user.id, body.query, body.top_k, body.document_hashes)
    return {"query": body.query, "results": results}
