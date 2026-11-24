import logging

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.resume_parser import ResumeParserAgent
from app.api.deps import get_current_user
from app.db.database import get_db
from app.models import User
from app.schemas.document import DeleteResult, DocumentOut, UploadResult
from app.services.document_service import VectorDBUnavailable, delete_document, list_documents, upload_documents

logger = logging.getLogger(__name__)
router = APIRouter()


def get_parser_agent() -> ResumeParserAgent:
    return ResumeParserAgent()


@router.post("/upload", response_model=list[UploadResult])
async def upload(
    # format "binary" makes the /docs page show a file picker for each item
    files: list[UploadFile] = File(json_schema_extra={"items": {"type": "string", "format": "binary"}}),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    agent: ResumeParserAgent = Depends(get_parser_agent),
):
    """Upload one or more resume PDFs. Already uploaded files are skipped. New ones are indexed in Qdrant."""
    contents = [(f.filename, await f.read()) for f in files]
    return await upload_documents(db, user, contents, agent)


@router.get("/", response_model=list[DocumentOut])
async def list_my_documents(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """The resumes you can read, newest first: a recruiter sees every user's, a candidate only their own.
    Use limit/offset to page through many."""
    return await list_documents(db, user, limit, offset)


@router.delete("/{document_id}", response_model=DeleteResult)
async def delete(document_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Delete one of your resumes: its chunks in Qdrant, its metadata and the document row.
    404 if it isn't yours (or doesn't exist). 503 if Qdrant is down: then nothing is deleted, try again."""
    try:
        deleted = await delete_document(db, user, document_id)
    except VectorDBUnavailable as e:
        logger.warning("deleting document %s failed in Qdrant: %s", document_id, e)
        raise HTTPException(503, "Vector DB not reachable, nothing was deleted. Try again later.")
    if not deleted:
        raise HTTPException(404, "Document not found")
    return {"document_id": document_id, "status": "deleted"}
