from langchain_core.messages import HumanMessage

from app.agents.base import BaseAgent
from app.core.config import settings
from app.llm.json_output import extract_json


class ResumeParserAgent(BaseAgent):
    name = "resume_parser"
    prompt = "resume_parser/system"
    model_id = settings.parser_model_id
    provider = settings.parser_provider
    temperature = settings.parser_temperature
    max_tokens = settings.parser_max_new_tokens
    timeout = settings.parser_timeout

    async def run(self, llm_input: str) -> dict:
        """Numbered resume lines -> resume structure with line ranges."""
        raw, _ = await self.call_llm([HumanMessage(content=llm_input)])
        return extract_json(raw)
