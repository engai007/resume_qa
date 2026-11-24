"""explain=true: an LLM verdict for the top_n, checked in code. The ranking never depends on it."""
from app.agents.match_agent import MatchAgent
from app.api.v1.endpoints.jds import get_match_agent
from app.main import app
from tests.conftest import auth_header
from tests.test_match import add_jd, add_resume, match
from tests.test_qa import FakeChat

GOOD = {"fit": "strong", "strengths": ["Python and SQL listed [1]", "Built data pipelines [2]"],
        "gaps": ["No Kafka shown"], "summary": "Solid backend fit."}


def use_llm(reply) -> FakeChat:
    chat = FakeChat(reply)
    app.dependency_overrides[get_match_agent] = lambda: MatchAgent(chat=chat)
    return chat


def two_resumes(client):
    headers = auth_header(client)
    strong, _ = add_resume(client, headers, "Languages: Python, SQL, Kafka, Docker")
    weak, _ = add_resume(client, headers, "Languages: Java")
    return headers, strong, weak, add_jd(client, headers)


def test_top_n_get_a_checked_verdict(client):
    headers, strong, weak, jd_id = two_resumes(client)
    chat = use_llm(GOOD)

    body = match(client, headers, jd_id, explain=True, top_n=1).json()
    first, second = body["results"]
    assert first["document_id"] == strong and second["verdict"] is None  # only the top 1
    v = first["verdict"]
    assert v["fit"] == "strong" and v["gaps"] == ["No Kafka shown"] and v["summary"] == "Solid backend fit."
    assert [s["n"] for s in v["sources"]] == [1, 2] and all("chunk_index" in s for s in v["sources"])
    assert len(chat.calls) == 1

    sent = chat.calls[0][-1].content  # JD fields, code's measurements, numbered chunks of THIS resume
    assert '"required_skills": ["Python", "SQL", "Kafka"]' in sent
    assert '"required_skills_missing": []' in sent
    assert "[1] section:" in sent and "Total experience:" in sent  # the total-experience chunk is always there
    assert "Languages: Java" not in sent  # nothing from the other resume


def test_reply_is_checked(client):
    headers, _, _, jd_id = two_resumes(client)
    use_llm({"fit": "Excellent!!", "strengths": ["Knows Python [1]", "Great leader", "Has Kafka [99]"],
             "gaps": "not a list", "summary": None})
    v = match(client, headers, jd_id, explain=True, top_n=1).json()["results"][0]["verdict"]
    assert v["fit"] is None  # not an allowed value
    assert v["strengths"] == ["Knows Python [1]"]  # no source / invented source [99] -> dropped
    assert v["gaps"] == [] and v["summary"] is None and [s["n"] for s in v["sources"]] == [1]


def test_gaps_lose_bracket_junk(client):
    headers, _, _, jd_id = two_resumes(client)
    use_llm({"fit": "possible", "strengths": ["Python [1]"],
             "gaps": ["SQS is missing [required_skills_missing]", "No Kafka [3]", "[x]"], "summary": "Ok [1]."})
    v = match(client, headers, jd_id, explain=True, top_n=1).json()["results"][0]["verdict"]
    assert v["gaps"] == ["SQS is missing", "No Kafka"] and v["summary"] == "Ok."


def test_failed_verdict_keeps_the_ranking(client):
    headers, strong, weak, jd_id = two_resumes(client)
    use_llm("Sure! This candidate is great")  # not JSON
    body = match(client, headers, jd_id, explain=True).json()
    assert [r["document_id"] for r in body["results"]] == [strong, weak]
    assert all(r["verdict"] is None for r in body["results"])

    use_llm(ConnectionError("deepinfra down"))
    assert match(client, headers, jd_id, explain=True).status_code == 200


def test_explain_off_or_top_n_zero_means_no_llm_call(client):
    headers, _, _, jd_id = two_resumes(client)
    chat = use_llm(GOOD)
    match(client, headers, jd_id, explain=False)
    match(client, headers, jd_id, explain=True, top_n=0)
    assert chat.calls == []


def test_default_explains_up_to_five(client):
    headers, _, _, jd_id = two_resumes(client)
    chat = use_llm(GOOD)
    body = client.post(f"/api/v1/jds/{jd_id}/match", headers=headers, json={}).json()  # explain defaults to true
    assert all(r["verdict"] for r in body["results"]) and len(chat.calls) == 2
