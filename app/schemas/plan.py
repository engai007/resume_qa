"""The planner's decision: HOW to fetch chunks for a question. See docs/planner_plan.md.
It never decides WHICH resumes: that's access + document_hashes, in code."""
from typing import Literal

from pydantic import BaseModel

from app.services.resume_chunker import SECTIONS

Route = Literal["search", "section"]
MAX_QUERY_CHARS = 200


class InvalidPlan(ValueError):
    """The planner's reply can't be used. The caller falls back to plain search."""


class Plan(BaseModel):
    route: Route  # search = find matching chunks; section = fetch every chunk of `sections`
    sections: list[str]  # only names from SECTIONS; always [] on search (search looks in every section)
    query: str  # search text (the question without filler)
    fallback: bool = False  # True = planner failed, this is plain search (today's behaviour)

    @classmethod
    def search_all(cls, question: str) -> "Plan":
        """The fallback: plain search over every section, exactly like /qa/ask without a planner."""
        return cls(route="search", sections=[], query=question, fallback=True)

    @classmethod
    def from_raw(cls, raw, question: str) -> "Plan":
        """Check and clean the planner's JSON. Only whitelisted values get through:
        unknown sections are dropped; a bad route, or a section route with no valid section, raises InvalidPlan."""
        if not isinstance(raw, dict):
            raise InvalidPlan("plan is not a JSON object")
        route = str(raw.get("route") or "").strip().lower()
        if route not in ("search", "section"):
            raise InvalidPlan(f"unknown route {route[:30]!r}")

        given = raw.get("sections") or []
        if isinstance(given, str):
            given = [given]
        sections = []
        for s in given if isinstance(given, list) else []:
            s = str(s).strip().lower()
            if s in SECTIONS and s not in sections:
                sections.append(s)
        if route == "section" and not sections:
            raise InvalidPlan("section route without a valid section")
        # Small models name only one half of a pair, whatever the prompt says, so code completes it:
        # - "project" without "experience" (a projects question) -> + "personal_project": "list all projects"
        #   means both. One way only: "personal projects" alone stays personal.
        # - "experience" -> + "project": company work is split between the two (a resume can have only project chunks).
        #   A company question (experience in the plan) never gets personal projects: they aren't company work.
        if "project" in sections and "experience" not in sections and "personal_project" not in sections:
            sections.append("personal_project")
        if "experience" in sections and "project" not in sections:
            sections.append("project")

        if route == "search":
            sections = []  # see qa_service.fetch_chunks: narrowing a search by a guessed section lost answers
        query = str(raw.get("query") or "").strip()[:MAX_QUERY_CHARS] or question
        return cls(route=route, sections=sections, query=query)
