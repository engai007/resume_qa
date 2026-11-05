from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.resume_parser import ResumeParserAgent
from app.api.deps import get_current_user
from app.db.database import get_db
from app.models import User
from app.schemas.document import UploadResult
from app.services.document_service import upload_documents

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
    """Upload one or more resume PDFs. Already uploaded files are skipped."""
    contents = [(f.filename, await f.read()) for f in files]
    return await upload_documents(db, user.id, contents, agent)
