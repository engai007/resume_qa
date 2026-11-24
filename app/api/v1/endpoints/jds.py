import logging

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.jd_parser import InvalidJDReply, JDParserAgent
from app.agents.match_agent import MatchAgent
from app.api.deps import get_current_user
from app.core.config import settings
from app.db.database import get_db
from app.models import User
from app.schemas.jd import DeleteJDResult, JDListItem, JDOut, UploadJDResult
from app.schemas.match import MatchRequest, MatchResponse
from app.services.jd_service import (InvalidJDFile, JDParserUnavailable, delete_jd, get_jd, jd_out, list_jds,
                                     upload_jd)
from app.services.match_service import JDNotFound, MatchUnavailable, match_jd

logger = logging.getLogger(__name__)
router = APIRouter()


def get_jd_parser_agent() -> JDParserAgent:
    return JDParserAgent()


def get_match_agent() -> MatchAgent:
    return MatchAgent()


@router.post("/upload", response_model=UploadJDResult)
async def upload(file: UploadFile = File(...), user: User = Depends(get_current_user),
                 db: AsyncSession = Depends(get_db), agent: JDParserAgent = Depends(get_jd_parser_agent)):
    """Upload one job description (.pdf or .txt). Returns its structure (every field optional).
    The same JD text uploaded again, as any file type, returns the saved one (status already_exists, no LLM call)."""
    content = await file.read(settings.jd_max_file_bytes + 1)  # one byte more is enough to see "too big"
    try:
        return await upload_jd(db, user, file.filename, content, agent)
    except InvalidJDFile as e:
        raise HTTPException(422, str(e))
    except InvalidJDReply as e:
        logger.warning("jd: invalid LLM reply: %s", e)
        raise HTTPException(502, "JD parser gave an invalid reply, nothing was saved. Try again.")
    except JDParserUnavailable as e:
        logger.warning("jd: LLM unavailable: %s", e)
        raise HTTPException(503, "JD parser unavailable, nothing was saved. Try again later.")


@router.get("/", response_model=list[JDListItem])
async def list_my_jds(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
                      user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Your job descriptions, newest first."""
    return await list_jds(db, user, limit, offset)


@router.get("/{jd_id}", response_model=JDOut)
async def get_one(jd_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """One of your job descriptions with its structure. 404 if it isn't yours."""
    jd = await get_jd(db, user, jd_id)
    if not jd:
        raise HTTPException(404, "Job description not found")
    return jd_out(jd)


@router.post("/{jd_id}/match", response_model=MatchResponse)
async def match(jd_id: int, body: MatchRequest = MatchRequest(), user: User = Depends(get_current_user),
                db: AsyncSession = Depends(get_db), agent: MatchAgent = Depends(get_match_agent)):
    """Rank the resumes you can access against one of your JDs. Every resume gets a score and a breakdown:
    required / preferred skills matched and missing (with where each was found), years vs the JD's minimum, and how
    close its best work chunk is to the JD. document_hashes = only those resumes.
    explain=true: the top_n also get an LLM verdict (fit, strengths with [n] sources, gaps); the ranking itself
    is always computed in code."""
    try:
        return await match_jd(db, user, jd_id, body.document_hashes, body.limit,
                              agent if body.explain else None, body.top_n)
    except JDNotFound:
        raise HTTPException(404, "Job description not found")
    except MatchUnavailable as e:
        logger.warning("match: vector search failed: %s", e)
        raise HTTPException(503, "Search service unavailable. Try again later.")


@router.delete("/{jd_id}", response_model=DeleteJDResult)
async def delete(jd_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Delete one of your job descriptions. 404 if it isn't yours."""
    if not await delete_jd(db, user, jd_id):
        raise HTTPException(404, "Job description not found")
    return {"jd_id": jd_id, "status": "deleted"}
