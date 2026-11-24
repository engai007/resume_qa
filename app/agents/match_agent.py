import json

from langchain_core.messages import HumanMessage

from app.agents.base import BaseAgent
from app.llm.json_output import extract_json

JD_FIELDS = ("title", "domain", "industry", "seniority", "experience_years", "required_skills", "preferred_skills",
             "responsibilities", "education")


class InvalidVerdict(Exception):
    """The model replied, but not with a JSON object."""


def build_context(jd: dict, candidate: str | None, breakdown: dict, evidence: list[dict]) -> str:
    """The user message: the JD's key fields, what code already measured, and numbered evidence chunks."""
    jd_short = {k: jd[k] for k in JD_FIELDS if jd.get(k)}
    measured = {
        "required_skills_found": [m["skill"] for m in (breakdown.get("required") or {}).get("matched", [])],
        "required_skills_missing": (breakdown.get("required") or {}).get("missing", []),
        "preferred_skills_found": [m["skill"] for m in (breakdown.get("preferred") or {}).get("matched", [])],
        "preferred_skills_missing": (breakdown.get("preferred") or {}).get("missing", []),
        "candidate_years": (breakdown.get("experience") or {}).get("candidate_years"),
    }
    parts = ["JD:", json.dumps(jd_short, ensure_ascii=False), "",
             f"Candidate: {candidate or 'unknown'}", "Measured by code (exact skill matches):",
             json.dumps(measured, ensure_ascii=False), "", "Resume chunks:"]
    for n, c in enumerate(evidence, start=1):
        parts += [f"[{n}] section: {c['section']}", c["text"], ""]
    return "\n".join(parts).strip()


class MatchAgent(BaseAgent):
    """Explains one candidate's fit for one JD, from that resume's chunks. Ranking is done in code, not here."""
    name = "match"
    prompt = "match/system"
    max_tokens = 600

    async def run(self, jd: dict, candidate: str | None, breakdown: dict, evidence: list[dict]) -> dict:
        """-> {"fit", "strengths", "gaps", "summary"} as the model wrote it (checked in match_service)."""
        raw, _ = await self.call_llm([HumanMessage(content=build_context(jd, candidate, breakdown, evidence))])
        try:
            reply = extract_json(raw)
        except ValueError as e:
            raise InvalidVerdict(f"not JSON: {e}") from e
        if not isinstance(reply, dict):
            raise InvalidVerdict("reply is not a JSON object")
        return reply
