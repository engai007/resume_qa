"""Job description file -> text -> LLM -> checked structure -> Postgres. See docs/jd_parser_plan.md."""
import asyncio
import hashlib
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.jd_parser import InvalidJDReply, JDParserAgent
from app.core.config import settings
from app.models import JobDescription, User
from app.schemas.jd import JDStructure
from app.services.resumer_parser.pdf_text import extract_lines, merge_wrapped

FILE_TYPES = (".pdf", ".txt")
MAX_TEXT_CHARS = 20_000  # a JD is ~3-8k characters; more is cut before the LLM


class InvalidJDFile(ValueError):
    """Wrong type, too big, no text, or not a job description. Nothing was saved."""


class JDParserUnavailable(Exception):
    """The LLM couldn't be reached (after retries). Nothing was saved."""


def normalise(text: str) -> str:
    """Whitespace (spaces, tabs, line breaks) collapsed to one space, ends trimmed; case kept."""
    return " ".join(text.split())


def text_hash(text: str) -> str:
    """Same JD text -> same hash, whether it came as a PDF, a re-saved PDF or a .txt."""
    return hashlib.sha256(normalise(text).encode("utf-8")).hexdigest()


async def file_to_text(filename: str, content: bytes) -> str:
    ext = Path(filename or "").suffix.lower()
    if ext not in FILE_TYPES:
        raise InvalidJDFile(f"use a {' or '.join(FILE_TYPES)} file")
    if len(content) > settings.jd_max_file_bytes:
        raise InvalidJDFile(f"file is larger than {settings.jd_max_file_bytes // 1_000_000} MB")
    if ext == ".pdf":
        if not content.startswith(b"%PDF"):
            raise InvalidJDFile("not a PDF file")
        lines = await asyncio.to_thread(extract_lines, content, settings.parser_gap_ratio)
        text = "\n".join(merge_wrapped(lines)) if lines else ""
    else:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise InvalidJDFile("text file is not UTF-8")
    if not normalise(text):
        raise InvalidJDFile("no text found (scanned PDF?)")
    return text.strip()[:MAX_TEXT_CHARS]


def jd_out(jd: JobDescription) -> dict:
    return {"jd_id": jd.id, "filename": jd.filename, "model_id": jd.model_id, "created_at": jd.created_at,
            "jd": jd.parsed_json}


async def find_by_hash(db: AsyncSession, user: User, h: str) -> JobDescription | None:
    return await db.scalar(select(JobDescription).where(JobDescription.created_by == user.id,
                                                        JobDescription.text_hash == h))


async def upload_jd(db: AsyncSession, user: User, filename: str, content: bytes, agent: JDParserAgent) -> dict:
    """Parse and save one JD. The same JD text uploaded again (any file type) is returned without an LLM call.
    Saved only after extraction succeeded."""
    text = await file_to_text(filename, content)
    h = text_hash(text)
    if existing := await find_by_hash(db, user, h):
        return {"jd_id": existing.id, "status": "already_exists", "filename": existing.filename,
                "jd": existing.parsed_json}

    try:
        raw = await agent.run(text)
    except InvalidJDReply:
        raise
    except Exception as e:
        raise JDParserUnavailable(type(e).__name__) from e
    structure = JDStructure.from_raw(raw)
    if structure.is_empty():
        raise InvalidJDFile("this doesn't look like a job description (no title, skills or responsibilities found)")

    jd = JobDescription(created_by=user.id, text_hash=h, filename=(filename or "jd")[:255], raw_text=text,
                        parsed_json=structure.model_dump(), model_id=agent.model_id)
    db.add(jd)
    try:
        await db.commit()
    except IntegrityError:  # the same JD uploaded twice at the same moment: keep the first one
        await db.rollback()
        jd = await find_by_hash(db, user, h)
        return {"jd_id": jd.id, "status": "already_exists", "filename": jd.filename, "jd": jd.parsed_json}
    return {"jd_id": jd.id, "status": "created", "filename": jd.filename, "jd": jd.parsed_json}


async def list_jds(db: AsyncSession, user: User, limit: int = 50, offset: int = 0) -> list[dict]:
    """The user's JDs, newest first."""
    rows = await db.scalars(select(JobDescription).where(JobDescription.created_by == user.id)
                            .order_by(JobDescription.created_at.desc(), JobDescription.id.desc())
                            .limit(limit).offset(offset))
    return [{"jd_id": jd.id, "filename": jd.filename, "title": jd.parsed_json.get("title"),
             "company": jd.parsed_json.get("company"), "created_at": jd.created_at} for jd in rows]


async def get_jd(db: AsyncSession, user: User, jd_id: int) -> JobDescription | None:
    """None if it doesn't exist or isn't this user's."""
    return await db.scalar(select(JobDescription).where(JobDescription.id == jd_id,
                                                        JobDescription.created_by == user.id))


async def delete_jd(db: AsyncSession, user: User, jd_id: int) -> bool:
    jd = await get_jd(db, user, jd_id)
    if not jd:
        return False
    await db.delete(jd)
    await db.commit()
    return True
