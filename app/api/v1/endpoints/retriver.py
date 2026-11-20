from fastapi import APIRouter, Depends

from app.api.deps import get_current_user
from app.models import User
from app.schemas.retriever import SearchRequest, SearchResponse
from app.services.retriever_service import search_chunks

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(body: SearchRequest, user: User = Depends(get_current_user)):
    """Find the resume chunks closest to the query. Only searches resumes you can access;
    document_hashes = only those resumes, sections = only those sections (e.g. ["project"]).
    mode: hybrid (default), dense or sparse."""
    results = await search_chunks(user, body.query, body.top_k, body.document_hashes, body.mode, body.sections)
    return {"query": body.query, "results": results}
