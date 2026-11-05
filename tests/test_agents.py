import asyncio
import json

from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.agents.resume_parser import ResumeParserAgent
from app.agents.registry import AGENTS
from app.main import app
from app.prompts import load_prompt


def test_every_agent_has_a_prompt_file():
    for agent in AGENTS.values():
        assert load_prompt(agent.prompt)


def test_list_agents_endpoint():
    names = [a["name"] for a in TestClient(app).get("/api/v1/agents/").json()]
    assert names == list(AGENTS)


def test_llm_call_is_logged(llm_log):
    agent = ResumeParserAgent(chat=FakeListChatModel(responses=['{"name": "Jane"}']))
    assert asyncio.run(agent.run("0|Jane")) == {"name": "Jane"}
    [line] = llm_log.read_text().splitlines()
    assert json.loads(line)["agent"] == "resume_parser"
