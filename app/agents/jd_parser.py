from langchain_core.messages import HumanMessage

from app.agents.base import BaseAgent
from app.core.config import settings
from app.llm.json_output import extract_json


class InvalidJDReply(Exception):
    """The model replied, but not with a JSON object."""


class JDParserAgent(BaseAgent):
    """Job description text -> structured JSON (values, not line ranges: a JD is short)."""
    name = "jd_parser"
    prompt = "jd_parser/system"
    model_id = settings.jd_parser_model_id
    provider = settings.jd_parser_provider
    fallback_provider = settings.jd_parser_fallback_provider
    max_tokens = settings.jd_parser_max_tokens
    timeout = settings.jd_parser_timeout

    async def run(self, jd_text: str) -> dict:
        """-> the JSON as the model wrote it (cleaned by JDStructure.from_raw afterwards).
        Errors from the model call itself (down, timeout) are raised as they are."""
        raw, _ = await self.call_llm([HumanMessage(content=f"Job description:\n<<<\n{jd_text}\n>>>")])
        try:
            reply = extract_json(raw)
        except ValueError as e:  # also json.JSONDecodeError
            raise InvalidJDReply(f"not JSON: {e}") from e
        if not isinstance(reply, dict):
            raise InvalidJDReply("reply is not a JSON object")
        return reply
