"""POST /api/v1/jds/{jd_id}/match: rank the user's resumes against their JD (no LLM)."""
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.agents.resume_parser import ResumeParserAgent
from app.api.v1.endpoints.documents import get_parser_agent
from app.main import app
from app.services import indexing_service
from app.services.document_service import file_hash
from tests.conftest import auth_header
from tests.resumer_parser.test_pipeline import LLM_REPLY, RESUME_LINES, make_pdf
from tests.test_documents import upload
from tests.test_jds import upload as upload_jd
from tests.test_jds import use_llm as use_jd_llm

JD = {"title": "Backend Engineer", "required_skills": ["Python", "SQL", "Kafka"], "preferred_skills": ["Docker"],
      "experience_years": {"min": 3}, "responsibilities": ["Build data pipelines"]}


def resume_pdf(skills_line: str) -> bytes:
    """The test resume (Acme Corp, Jan 2020 - Present) with its skills line replaced: same parse, other skills."""
    return make_pdf(RESUME_LINES[:7] + [skills_line])


def add_resume(client, headers, skills_line):
    app.dependency_overrides[get_parser_agent] = lambda: ResumeParserAgent(chat=FakeListChatModel(responses=[LLM_REPLY]))
    pdf = resume_pdf(skills_line)
    [r] = upload(client, headers, ("cv.pdf", pdf)).json()
    return r["document_id"], file_hash(pdf)


def add_jd(client, headers, jd=JD):
    use_jd_llm(jd)
    return upload_jd(client, headers, "jd.txt", repr(jd).encode()).json()["jd_id"]


def match(client, headers, jd_id, **body):
    body.setdefault("explain", False)  # ranking tests: no LLM (verdict tests in test_match_verdict.py)
    return client.post(f"/api/v1/jds/{jd_id}/match", headers=headers, json=body)


def test_ranks_by_skills_with_breakdown(client):
    headers = auth_header(client)
    weak, _ = add_resume(client, headers, "Languages: Java")
    strong, _ = add_resume(client, headers, "Languages: Python, SQL, Kafka, Docker")
    middle, _ = add_resume(client, headers, "Languages: Python, SQL")

    body = match(client, headers, add_jd(client, headers)).json()
    assert body["jd_title"] == "Backend Engineer" and body["resumes_compared"] == 3
    assert [r["document_id"] for r in body["results"]] == [strong, middle, weak]
    assert [r["rank"] for r in body["results"]] == [1, 2, 3]

    top = body["results"][0]["breakdown"]
    assert top["required"]["score"] == 1.0 and top["required"]["missing"] == []
    assert top["required"]["matched"][0] == {"skill": "Python", "found_as": "Python", "section": "skills",
                                             "chunk_index": 1}
    assert top["preferred"]["matched"][0]["skill"] == "Docker"
    assert top["experience"]["score"] == 1.0 and top["experience"]["required_min"] == 3  # Jan 2020 - now
    assert top["semantic"]["best_chunk_index"] is not None
    weakest = body["results"][2]["breakdown"]["required"]
    assert weakest["missing"] == ["SQL", "Kafka"]
    assert weakest["matched"][0]["section"] == "experience"  # "Built data pipelines in Python": a work bullet counts


def test_scope_document_hashes_and_limit(client):
    headers = auth_header(client)
    add_resume(client, headers, "Languages: Python")
    doc_b, hash_b = add_resume(client, headers, "Languages: SQL")
    jd_id = add_jd(client, headers)

    only = match(client, headers, jd_id, document_hashes=[hash_b]).json()
    assert only["resumes_compared"] == 1 and only["results"][0]["document_id"] == doc_b
    assert len(match(client, headers, jd_id, limit=1).json()["results"]) == 1
    assert match(client, headers, jd_id, limit=1).json()["resumes_compared"] == 2  # all scored, 1 returned


def test_only_own_resumes_and_own_jd(client):
    alice, bob = auth_header(client, email="alice@x.com"), auth_header(client, email="bob@x.com")
    _, alice_hash = add_resume(client, alice, "Languages: Python, SQL, Kafka")
    bob_doc, _ = add_resume(client, bob, "Languages: Java")
    alice_jd, bob_jd = add_jd(client, alice), add_jd(client, bob)

    assert match(client, bob, alice_jd).status_code == 404  # not bob's JD
    body = match(client, bob, bob_jd).json()
    assert [r["document_id"] for r in body["results"]] == [bob_doc]  # alice's resume never compared
    assert match(client, bob, bob_jd, document_hashes=[alice_hash]).json()["resumes_compared"] == 0


def test_not_indexed_resumes_are_skipped_and_counted(client, monkeypatch):
    headers = auth_header(client)
    add_resume(client, headers, "Languages: Python")

    async def hf_down(texts):
        raise ConnectionError("HF down")
    monkeypatch.setattr(indexing_service, "embed_texts", hf_down)
    add_resume(client, headers, "Languages: Python, SQL, Kafka")  # saved, not indexed

    body = match(client, headers, add_jd(client, headers)).json()
    assert body["resumes_compared"] == 1 and body["skipped_not_indexed"] == 1


def test_requirements_only_jd(client):
    headers = auth_header(client)
    add_resume(client, headers, "Languages: Python")
    body = match(client, headers, add_jd(client, headers, {"required_skills": ["Python"]})).json()
    b = body["results"][0]["breakdown"]
    assert b["preferred"] is None and b["experience"] is None  # left out, not 0
    assert b["semantic"] is not None  # the JD's skills are still compared with the work


def test_no_resumes_and_login(client):
    headers = auth_header(client)
    jd_id = add_jd(client, headers)
    assert match(client, headers, jd_id).json() == {"jd_id": jd_id, "jd_title": "Backend Engineer",
                                                     "resumes_compared": 0, "skipped_not_indexed": 0, "results": []}
    assert match(client, {}, jd_id).status_code == 401
