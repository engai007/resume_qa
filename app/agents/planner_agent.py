from langchain_core.messages import HumanMessage

from app.agents.base import BaseAgent
from app.core.config import settings
from app.llm.json_output import extract_json
from app.schemas.plan import InvalidPlan, Plan


class PlannerAgent(BaseAgent):
    """Reads ONLY the question (never resume text) and decides how to fetch: search, or whole sections.
    A small, fast model on another provider, so the big answer model's load doesn't change."""
    name = "planner"
    prompt = "planner/system"
    model_id = settings.planner_model_id
    provider = settings.planner_provider
    fallback_provider = settings.planner_fallback_provider
    max_tokens = settings.planner_max_tokens
    timeout = settings.planner_timeout

    async def plan(self, question: str) -> Plan:
        """Question -> checked Plan. Raises InvalidPlan for a reply that can't be used;
        errors from the model call itself (down, timeout) are raised as they are."""
        raw, _ = await self.call_llm([HumanMessage(content=f"Q: {question}")])
        try:
            reply = extract_json(raw)
        except ValueError as e:  # also json.JSONDecodeError
            raise InvalidPlan(f"not JSON: {e}") from e
        return Plan.from_raw(reply, question)
