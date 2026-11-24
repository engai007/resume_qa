"""JD -> ranked resumes. See docs/jd_matching_plan.md.

1. the JD          the user's own (else JDNotFound)
2. scope           resumes the user can access (access.accessible_documents) ∩ document_hashes, indexed only
3. semantic        ONE grouped Qdrant call: each resume's best work chunks vs the JD's focus text
4. score           in code per resume (match_scoring.score_resume): skills, years, semantic
5. rank            score, then more required skills matched, then document id (stable)
6. explain         top_n only, LLM (MatchAgent), in parallel: fit / strengths [n] / gaps. A failed one -> null
"""
import asyncio
import logging
import re

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.agents.match_agent import MatchAgent
from app.models import Document, User
from app.services.access import accessible_documents
from app.services.indexing_service import INDEX_VERSION
from app.services.jd_service import get_jd
from app.services.match_scoring import SEMANTIC_SECTIONS, jd_focus_text, score_resume
from app.services.resume_chunker import make_chunks
from app.services.retriever_service import best_chunks_per_resume

logger = logging.getLogger(__name__)

MAX_EVIDENCE = 8  # chunks per candidate given to the LLM
EXPLAIN_CONCURRENCY = 3  # LLM calls in flight at once
FITS = ("strong", "possible", "weak")


class JDNotFound(Exception):
    pass


class MatchUnavailable(Exception):
    """The vector DB or the embedding service couldn't be reached."""


def required_matched(result: dict) -> int:
    part = result["breakdown"]["required"]
    return len(part["matched"]) if part else 0


def build_evidence(resume: dict, hits: list[dict], breakdown: dict) -> list[dict]:
    """The chunks the LLM judges from, best first, no repeats: the resume's best work chunks (semantic), the
    chunks where JD skills were found, and always the total-experience chunk. At most MAX_EVIDENCE."""
    chunks = make_chunks(resume)
    picked, seen = [], set()

    def add(index: int, section: str, text: str):
        if index not in seen and len(picked) < MAX_EVIDENCE - 1:  # keep one place for the summary
            seen.add(index)
            picked.append({"chunk_index": index, "section": section, "text": text})

    for h in hits:
        add(h["chunk_index"], h["section"], h["text"])
    for part in ("required", "preferred"):
        for m in (breakdown.get(part) or {}).get("matched", []):
            if m["chunk_index"] < len(chunks):
                add(m["chunk_index"], *chunks[m["chunk_index"]])
    for i, (section, text) in enumerate(chunks):
        if section == "experience_summary" and i not in seen:
            picked.append({"chunk_index": i, "section": section, "text": text})
    return picked


def cited(text: str, n_max: int) -> list[int]:
    """"Built APIs [1] and [2, 3]" -> [1, 2, 3]; only real chunk numbers."""
    nums = [int(x) for group in re.findall(r"\[([\d,\s]+)\]", text) for x in group.split(",") if x.strip()]
    return [n for n in nums if 1 <= n <= n_max]


def clean_note(text, limit: int = 300) -> str:
    """Gaps and summary: one line, no "[3]" or "[required_skills_missing]" (the model echoed field names)."""
    return " ".join(re.sub(r"\s*\[[^\]]*\]", "", str(text or "")).split())[:limit]


def check_verdict(reply: dict, evidence: list[dict]) -> dict:
    """Don't trust the reply blindly (like Q&A): only allowed fit values, every strength must cite a real chunk."""
    fit = str(reply.get("fit") or "").strip().lower()
    strengths, sources = [], []
    for s in reply.get("strengths") if isinstance(reply.get("strengths"), list) else []:
        nums = cited(str(s), len(evidence))
        if nums:  # a strength without evidence is dropped
            strengths.append(" ".join(str(s).split())[:300])
            sources += [n for n in nums if n not in sources]
    gaps = [clean_note(g) for g in (reply.get("gaps") if isinstance(reply.get("gaps"), list) else [])]
    gaps = [g for g in gaps if g][:6]
    summary = clean_note(reply.get("summary"), 500) or None
    return {"fit": fit if fit in FITS else None, "strengths": strengths[:6], "gaps": gaps, "summary": summary,
            "sources": [{"n": n, "chunk_index": evidence[n - 1]["chunk_index"], "section": evidence[n - 1]["section"]}
                        for n in sorted(sources)]}


async def explain(results: list[dict], resumes: dict[int, dict], groups: dict, jd: dict, agent: MatchAgent) -> None:
    """Adds "verdict" to each result (in place), in parallel. A failed call -> verdict None, ranking unchanged."""
    limit = asyncio.Semaphore(EXPLAIN_CONCURRENCY)

    async def one(r):
        evidence = build_evidence(resumes[r["document_id"]], groups.get(r["document_id"], []), r["breakdown"])
        async with limit:
            reply = await agent.run(jd, r["candidate_name"], r["breakdown"], evidence)
        return check_verdict(reply, evidence)

    verdicts = await asyncio.gather(*(one(r) for r in results), return_exceptions=True)
    for r, v in zip(results, verdicts):
        if isinstance(v, Exception):
            logger.warning("match: verdict failed for document %s: %s", r["document_id"], type(v).__name__)
            v = None
        r["verdict"] = v


async def match_jd(db: AsyncSession, user: User, jd_id: int, document_hashes: list[str] | None = None,
                   limit: int = 50, agent: MatchAgent | None = None, top_n: int = 5) -> dict:
    """agent=None or top_n=0: ranking only, no LLM call."""
    jd_row = await get_jd(db, user, jd_id)
    if not jd_row:
        raise JDNotFound(jd_id)
    jd = jd_row.parsed_json

    query = accessible_documents(user).options(selectinload(Document.meta_data))
    if document_hashes:
        query = query.where(Document.document_hash.in_(document_hashes))
    docs = (await db.scalars(query)).all()
    indexed = [d for d in docs if d.indexed_version == INDEX_VERSION and d.meta_data]

    focus = jd_focus_text(jd)
    groups = {}
    if indexed and focus:
        try:
            groups = await best_chunks_per_resume(user, focus, [d.document_hash for d in indexed],
                                                  SEMANTIC_SECTIONS, group_size=3, max_resumes=len(indexed))
        except Exception as e:
            raise MatchUnavailable(type(e).__name__) from e

    results = []
    for d in indexed:
        best = (groups.get(d.id) or [None])[0]
        scored = score_resume(d.meta_data.parsed_doc_json, jd, best and best["score"], best and best["chunk_index"],
                              semantic=bool(focus))
        results.append({"document_id": d.id, "document_hash": d.document_hash,
                        "candidate_name": d.meta_data.parsed_doc_json.get("name"), **scored})
    results.sort(key=lambda r: (-r["score"], -required_matched(r), r["document_id"]))
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank
        r["verdict"] = None
    if agent and top_n and results:
        resumes = {d.id: d.meta_data.parsed_doc_json for d in indexed}
        await explain(results[:top_n], resumes, groups, jd, agent)

    return {"jd_id": jd_row.id, "jd_title": jd.get("title"), "resumes_compared": len(indexed),
            "skipped_not_indexed": len(docs) - len(indexed), "results": results[:limit]}
