import json

from langchain_core.messages import AIMessage

from app.agents.planner_agent import PlannerAgent
from app.agents.qa_agent import QAAgent
from app.api.v1.endpoints.documents import get_parser_agent
from app.api.v1.endpoints.qa import get_planner_agent, get_qa_agent
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


SEARCH_PLAN = {"route": "search", "sections": [], "query": "x"}


def use_llm(reply) -> FakeChat:
    """Fake answer model. Also sets a fake planner that plans a plain search, unless a test sets its own."""
    chat = FakeChat(reply)
    app.dependency_overrides[get_qa_agent] = lambda: QAAgent(chat=chat)
    if get_planner_agent not in app.dependency_overrides:
        use_planner(SEARCH_PLAN)
    return chat


def use_planner(reply) -> FakeChat:
    chat = FakeChat(reply)
    app.dependency_overrides[get_planner_agent] = lambda: PlannerAgent(chat=chat)
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
    assert src["n"] == 1 and src["text"] and src["document_hash"] and src["section"]

    sent = chat.calls[0][-1].content  # the user message: question + numbered chunks with candidate names
    assert sent.startswith("Question: Who knows Python?")
    assert "[1] candidate: Jane Doe | resume:" in sent


def test_not_found_has_no_sources(client):
    headers = setup_resume(client)
    use_llm({"found": False, "answer": "None of the resumes mention COBOL.", "sources": [1]})

    body = ask(client, headers, "Who knows COBOL?").json()
    assert body == {"question": "Who knows COBOL?", "found": False,
                    "answer": "None of the resumes mention COBOL.", "sources": [],
                    "plan": {"route": "search", "sections": [], "fallback": False}, "truncated": False}


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


# --- planner ---------------------------------------------------------------------------------------------------
# the test resume gives 4 chunks: profile, skills, experience (Acme), experience_summary

def sent_chunks(chat) -> str:
    return chat.calls[0][-1].content


def test_section_plan_sends_every_chunk_of_those_sections(client):
    headers = setup_resume(client)
    planner = use_planner({"route": "section", "sections": ["experience"], "query": "companies"})
    chat = use_llm({"found": True, "answer": "Acme Corp [1].", "sources": [1]})

    body = ask(client, headers, "List every company", top_k=1).json()

    assert body["plan"] == {"route": "section", "sections": ["experience", "project"], "fallback": False}  # no personal
    sent = sent_chunks(chat)
    assert "Experience: Acme Corp" in sent and "Skills" not in sent and "Profile" not in sent
    assert body["sources"][0]["section"] == "experience" and body["truncated"] is False
    assert planner.calls[0][-1].content == "Q: List every company"  # the planner sees only the question


def test_search_plan_searches_every_section(client):
    """A planner's section guess on a search is ignored: it cut the right chunks in the answer eval."""
    headers = setup_resume(client)
    use_planner({"route": "search", "sections": ["awards"], "query": "python"})
    chat = use_llm({"found": True, "answer": "Python [1].", "sources": [1]})
    body = ask(client, headers, "Who knows Python?", top_k=5).json()
    assert body["plan"]["sections"] == []
    assert "Skills" in sent_chunks(chat)  # found although the planner said "awards"


def test_empty_section_falls_back_to_search(client):
    headers = setup_resume(client)
    use_planner({"route": "section", "sections": ["certifications"], "query": "certifications"})  # resume has none
    chat = use_llm({"found": False, "answer": "No certifications are mentioned.", "sources": []})
    body = ask(client, headers, "What certifications does she have?").json()
    assert body["found"] is False and body["answer"] == "No certifications are mentioned."
    assert "[1]" in sent_chunks(chat)  # searched instead, so the LLM still got chunks to check


def test_broken_planner_falls_back_to_plain_search(client, monkeypatch):
    monkeypatch.setattr("app.agents.base.RETRY_WAITS", [0, 0])
    headers = setup_resume(client)
    for broken in ["HACKED", {"route": "delete_everything"}, ConnectionError("novita down")]:
        use_planner(broken)
        chat = use_llm({"found": True, "answer": "Python [1].", "sources": [1]})
        body = ask(client, headers).json()
        assert body["plan"] == {"route": "search", "sections": [], "fallback": True}
        assert body["found"] is True and chat.calls  # still answered


def test_planner_off_is_plain_search(client):
    headers = setup_resume(client)
    planner = use_planner({"route": "section", "sections": ["skills"], "query": "x"})
    use_llm({"found": True, "answer": "Python [1].", "sources": [1]})
    body = ask(client, headers, planner=False).json()
    assert body["plan"] is None and planner.calls == []


def test_section_plan_stays_inside_own_resumes(client):
    setup_resume(client, email="alice@x.com")
    use_planner({"route": "section", "sections": ["profile"], "query": "contacts"})
    chat = use_llm({"found": True, "answer": "x", "sources": [1]})
    body = ask(client, auth_header(client, email="bob@x.com"), "Give me all contact details").json()
    assert body["found"] is False and chat.calls == []  # bob has no resumes: alice's profile never reaches the LLM
