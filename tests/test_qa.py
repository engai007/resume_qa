import json

from langchain_core.messages import AIMessage

from app.agents.qa_agent import QAAgent
from app.api.v1.endpoints.documents import get_parser_agent
from app.api.v1.endpoints.qa import get_qa_agent
from app.main import app
from tests.conftest import auth_header
from tests.resumer_parser.test_pipeline import RESUME_LINES, make_pdf
from tests.test_documents import fake_agent, upload

URL = "/api/v1/qa/ask"


class FakeChat:
    """Stands in for the LLM: returns `reply` and remembers what it was sent."""

    def __init__(self, reply):
        self.reply, self.calls = reply, []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if isinstance(self.reply, Exception):
            raise self.reply
        return AIMessage(content=self.reply if isinstance(self.reply, str) else json.dumps(self.reply))


def use_llm(reply) -> FakeChat:
    chat = FakeChat(reply)
    app.dependency_overrides[get_qa_agent] = lambda: QAAgent(chat=chat)
    return chat


def setup_resume(client, email="jane@x.com") -> dict:
    app.dependency_overrides[get_parser_agent] = fake_agent
    headers = auth_header(client, email=email)
    upload(client, headers, ("jane.pdf", make_pdf(RESUME_LINES)))
    return headers


def ask(client, headers, question="Who knows Python?", **body):
    return client.post(URL, headers=headers, json={"question": question, **body})


def test_answer_with_sources(client):
    headers = setup_resume(client)
    chat = use_llm({"found": True, "answer": "Jane Doe knows Python [1].", "sources": [1]})

    r = ask(client, headers)
    assert r.status_code == 200
    body = r.json()
    assert body["found"] is True and body["answer"] == "Jane Doe knows Python [1]."
    [src] = body["sources"]
    assert src["n"] == 1 and src["text"] and src["document_hash"]

    sent = chat.calls[0][-1].content  # the user message: question + numbered chunks with candidate names
    assert sent.startswith("Question: Who knows Python?")
    assert "[1] candidate: Jane Doe | resume:" in sent


def test_not_found_has_no_sources(client):
    headers = setup_resume(client)
    use_llm({"found": False, "answer": "None of the resumes mention COBOL.", "sources": [1]})

    body = ask(client, headers, "Who knows COBOL?").json()
    assert body == {"question": "Who knows COBOL?", "found": False,
                    "answer": "None of the resumes mention COBOL.", "sources": []}


def test_invented_sources_are_dropped(client):
    headers = setup_resume(client)
    use_llm({"found": True, "answer": "Python [1][9].", "sources": [1, 9, "x"]})
    assert [s["n"] for s in ask(client, headers).json()["sources"]] == [1]


def test_answer_without_a_real_source_becomes_not_found(client):
    headers = setup_resume(client)
    use_llm({"found": True, "answer": "Jane knows COBOL.", "sources": [9]})

    body = ask(client, headers, "Who knows COBOL?").json()
    assert body["found"] is False and body["sources"] == []
    assert "COBOL" not in body["answer"]  # the unsupported claim is not shown


def test_no_resumes_means_no_llm_call(client):
    chat = use_llm({"found": True, "answer": "x", "sources": [1]})
    body = ask(client, auth_header(client)).json()
    assert body["found"] is False and "No resumes" in body["answer"]
    assert chat.calls == []


def test_only_own_resumes_reach_the_llm(client):
    setup_resume(client, email="alice@x.com")  # alice has a resume, bob doesn't
    chat = use_llm({"found": True, "answer": "x", "sources": [1]})
    assert ask(client, auth_header(client, email="bob@x.com")).json()["found"] is False
    assert chat.calls == []


def test_llm_errors(client):
    headers = setup_resume(client)
    use_llm("Sure! Here is the answer: Jane knows Python")  # not JSON
    assert ask(client, headers).status_code == 502
    use_llm(ConnectionError("deepinfra down"))
    assert ask(client, headers).status_code == 503


def test_needs_login_and_a_question(client):
    use_llm({"found": False, "answer": "", "sources": []})
    assert ask(client, {}).status_code == 401
    assert ask(client, auth_header(client), "").status_code == 422
