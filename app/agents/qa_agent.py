from langchain_core.messages import HumanMessage

from app.agents.base import BaseAgent
from app.llm.json_output import extract_json


def format_chunk(n: int, c: dict) -> str:
    """One chunk as the model sees it: number, candidate, resume id, then the text.
    The RAGAS eval judges answers against exactly this (the name is only in the header)."""
    who = c.get("candidate_name") or "unknown"
    return f"[{n}] candidate: {who} | resume: {c['document_hash'][:6]}\n{c['text']}"


def build_context(question: str, chunks: list[dict]) -> str:
    """The user message: the question + chunks numbered from 1, each with its candidate and resume id."""
    parts = [f"Question: {question}", "", "Chunks:"]
    for n, c in enumerate(chunks, start=1):
        parts += [format_chunk(n, c), ""]
    return "\n".join(parts).strip()


class InvalidAnswer(Exception):
    """The model replied, but not with the JSON we asked for."""


class QAAgent(BaseAgent):
    """Answers a question from resume chunks; says found=false when they don't answer it."""
    name = "qa"
    prompt = "qa/system"
    max_tokens = 1024  # a list of all projects with [n] markers doesn't fit in 512

    async def run(self, question: str, chunks: list[dict]) -> dict:
        """-> {"found": bool, "answer": str, "sources": [chunk numbers]} as the model wrote it (not checked yet).
        Errors from the model call itself (down, timeout) are raised as they are."""
        raw, _ = await self.call_llm([HumanMessage(content=build_context(question, chunks))])
        try:
            reply = extract_json(raw)
        except ValueError as e:  # also json.JSONDecodeError
            raise InvalidAnswer(f"not JSON: {e}") from e
        if not isinstance(reply, dict):
            raise InvalidAnswer("reply is not a JSON object")
        return reply
