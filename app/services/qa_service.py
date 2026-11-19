"""Question -> hybrid search -> LLM answer with sources. See docs/qa_answer_plan.md."""
from app.agents.qa_agent import InvalidAnswer, QAAgent
from app.services.retriever_service import search_chunks

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


async def answer_question(user_id: int, question: str, agent: QAAgent, top_k: int = 5,
                          document_hashes: list[str] | None = None) -> dict:
    try:
        chunks = await search_chunks(user_id, question, top_k=top_k, document_hashes=document_hashes)
    except Exception as e:
        raise AnswerUnavailable(f"search failed: {e}") from e
    if not chunks:  # nothing uploaded (or nothing matching the given hashes): no LLM call
        return {"question": question, "found": False, "answer": NO_RESUMES, "sources": []}

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
                           ("document_id", "document_hash", "candidate_name", "chunk_index", "text")}}
               for n in numbers] if found else []
    return {"question": question, "found": found, "answer": answer, "sources": sources}
