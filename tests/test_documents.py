import json

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.agents.resume_parser import ResumeParserAgent
from app.api.v1.endpoints.documents import get_parser_agent
from app.main import app
from tests.conftest import auth_header
from tests.resumer_parser.test_pipeline import LLM_REPLY, RESUME_LINES, make_pdf

URL = "/api/v1/documents/upload"


def fake_agent():
    return ResumeParserAgent(chat=FakeListChatModel(responses=[LLM_REPLY]))


def upload(client, headers, *files):
    return client.post(URL, headers=headers, files=[("files", (name, data, "application/pdf")) for name, data in files])


def llm_calls(llm_log) -> int:
    return len(llm_log.read_text().splitlines()) if llm_log.exists() else 0


def test_upload_multiple_and_skip_existing(client, llm_log):
    app.dependency_overrides[get_parser_agent] = fake_agent
    headers = auth_header(client)
    jane = make_pdf(RESUME_LINES)
    john = make_pdf(RESUME_LINES + ["Extra line"])

    # same file twice in one upload -> parsed once
    r = upload(client, headers, ("jane.pdf", jane), ("john.pdf", john), ("jane_copy.pdf", jane)).json()
    assert [x["status"] for x in r] == ["created", "created", "created"]
    assert r[0]["document_id"] == r[2]["document_id"]
    assert llm_calls(llm_log) == 2

    # upload again -> nothing parsed
    r = upload(client, headers, ("jane.pdf", jane)).json()
    assert r == [{"filename": "jane.pdf", "status": "already_exists", "document_id": 1, "indexed": True, "error": None}]
    assert llm_calls(llm_log) == 2


def test_same_file_for_another_user_is_new(client):
    app.dependency_overrides[get_parser_agent] = fake_agent
    jane = make_pdf(RESUME_LINES)
    upload(client, auth_header(client, email="a@x.com"), ("jane.pdf", jane))
    r = upload(client, auth_header(client, email="b@x.com"), ("jane.pdf", jane)).json()
    assert r[0]["status"] == "created"


def test_non_pdf_fails_others_still_saved(client):
    app.dependency_overrides[get_parser_agent] = fake_agent
    r = upload(client, auth_header(client), ("notes.txt", b"hello"), ("jane.pdf", make_pdf(RESUME_LINES))).json()
    assert r[0] == {"filename": "notes.txt", "status": "failed", "document_id": None, "indexed": False, "error": "not a PDF file"}
    assert r[1]["status"] == "created"


def test_upload_needs_login(client):
    assert upload(client, {}, ("jane.pdf", b"%PDF")).status_code in (401, 403)
