"""End-to-end parser test: real PDF extraction, fake LLM (no network, no token)."""
import asyncio
import json

import pymupdf
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.agents.resume_parser import ResumeParserAgent
from app.services.resumer_parser import parse_resume, parse_resumes

RESUME_LINES = [
    "Jane Doe",                                          # 0
    "Email: jane@x.com | Phone: +91 98765 43210",        # 1
    "Experience",                                        # 2
    "Acme Corp Jan 2020 - Present",                      # 3
    "Built data pipelines in Python",                    # 4
    "Led a team of five engineers",                      # 5
    "Skills",                                            # 6
    "Languages: Python, SQL",                            # 7
]

LLM_REPLY = json.dumps({
    "name": "Jane Doe",
    "experience": [{"company": "Acme Corp", "start": "Jan 2020", "end": "Present", "lines": [3, 5]}],
    "skills": [6, 7],
})


def make_pdf(lines: list[str]) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    for i, text in enumerate(lines):
        page.insert_text((72, 72 + i * 18), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def make_agent() -> ResumeParserAgent:
    return ResumeParserAgent(chat=FakeListChatModel(responses=[LLM_REPLY]))


def test_parse_resume():
    r = asyncio.run(parse_resume(make_pdf(RESUME_LINES), make_agent()))
    assert r["name"] == "Jane Doe"
    assert r["email"] == "jane@x.com" and r["phone"] == "+91 98765 43210"
    assert r["experience"][0]["details"] == ["Built data pipelines in Python", "Led a team of five engineers"]
    assert r["skills"] == {"Languages": ["Python", "SQL"]}


def test_parse_resumes_keeps_going_when_one_fails():
    files = {"jane.pdf": make_pdf(RESUME_LINES), "broken.pdf": b"not a pdf"}
    results = asyncio.run(parse_resumes(files, make_agent()))
    assert results["jane.pdf"]["name"] == "Jane Doe"
    assert isinstance(results["broken.pdf"], Exception)
