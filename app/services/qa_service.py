"""Question -> plan -> chunks -> LLM answer with sources. See docs/qa_answer_plan.md and docs/planner_plan.md.

1. plan    the planner (small LLM) decides HOW to fetch: search, or every chunk of some sections.
           planner off, or it fails -> plain search (the behaviour before the planner)
2. fetch   always inside the user's access (+ document_hashes); a section fetch that finds nothing -> plain search
3. answer  the QA agent (big LLM) answers from the chunks; its sources are checked here
"""
import logging

from app.agents.planner_agent import PlannerAgent
from app.agents.qa_agent import InvalidAnswer, QAAgent
from app.models import User
from app.schemas.plan import Plan
from app.services.retriever_service import fetch_sections, search_chunks

logger = logging.getLogger(__name__)

NO_RESUMES = "No resumes to search. Upload a resume first."
NOT_SUPPORTED = "The resumes don't clearly answer this."


class AnswerUnavailable(Exception):
    """Search or the LLM couldn't be reached. Nothing to do but try again."""


def source_numbers(raw, n_chunks: int) -> list[int]:
    """The chunk numbers the model cited, keeping only real ones (1..n_chunks), in order, no repeats."""
    out = []
    for s in raw if isinstance(raw, list) else []:
        try:
            n = int(s)
        except (TypeError, ValueError):
            continue
        if 1 <= n <= n_chunks and n not in out:
            out.append(n)
    return out


async def make_plan(question: str, planner: PlannerAgent) -> Plan:
    """The planner's plan, or plain search if it fails in any way (down, timeout, not JSON, bad values).
    The planner can make answers better, never unavailable."""
    try:
        plan = await planner.plan(question)
    except Exception as e:
        logger.warning("planner failed, using plain search: %s", type(e).__name__)
        return Plan.search_all(question)
    logger.info("plan route=%s sections=%s", plan.route, plan.sections)  # no question text in logs
    return plan


async def fetch_chunks(user: User, question: str, plan: Plan | None, top_k: int,
                       document_hashes: list[str] | None) -> tuple[list[dict], bool]:
    """-> (chunks, truncated). Search uses the original question (that's what the retrieval eval measured) and
    every section: a planner's guess of where a fact sits removed the right chunks in the answer eval
    (Spark is in skills, not projects), and plain search already found them."""
    if plan and plan.route == "section":
        chunks, truncated = await fetch_sections(user, plan.sections, document_hashes)
        if chunks:
            return chunks, truncated
        # e.g. "certifications?" but no resume has that section: maybe it's mentioned elsewhere, so search
    return await search_chunks(user, question, top_k=top_k, document_hashes=document_hashes), False


async def answer_question(user: User, question: str, agent: QAAgent, top_k: int = 5,
                          document_hashes: list[str] | None = None, planner: PlannerAgent | None = None) -> dict:
    plan = await make_plan(question, planner) if planner else None
    plan_out = {"route": plan.route, "sections": plan.sections, "fallback": plan.fallback} if plan else None

    try:
        chunks, truncated = await fetch_chunks(user, question, plan, top_k, document_hashes)
    except Exception as e:
        raise AnswerUnavailable(f"search failed: {e}") from e
    result = {"question": question, "plan": plan_out, "truncated": truncated}
    if not chunks:  # nothing uploaded (or nothing matching the given hashes): no LLM call
        return {**result, "found": False, "answer": NO_RESUMES, "sources": []}

    try:
        reply = await agent.run(question, chunks)
    except InvalidAnswer:
        raise
    except Exception as e:
        raise AnswerUnavailable(f"LLM failed: {e}") from e

    # don't trust the reply blindly: only real chunk numbers, and an answer must point at one
    numbers = source_numbers(reply.get("sources"), len(chunks))
    found = reply.get("found") is True and bool(numbers)
    if found:
        answer = str(reply.get("answer") or "").strip()
    elif reply.get("found") is True:  # claimed an answer without a real source
        answer = NOT_SUPPORTED
    else:
        answer = str(reply.get("answer") or "").strip() or NOT_SUPPORTED
    sources = [{"n": n, **{k: chunks[n - 1][k] for k in
                           ("document_id", "document_hash", "candidate_name", "chunk_index", "section", "text")}}
               for n in numbers] if found else []
    return {**result, "found": found, "answer": answer, "sources": sources}
