import pytest

from app.agents.jd_parser import JDParserAgent
from app.api.v1.endpoints.jds import get_jd_parser_agent
from app.main import app
from app.schemas.jd import JDStructure
from app.services.jd_service import text_hash
from tests.conftest import auth_header
from tests.resumer_parser.test_pipeline import make_pdf
from tests.test_qa import FakeChat

URL = "/api/v1/jds"
JD_LINES = ["Senior Backend Engineer - Acme", "Must have: Python, FastAPI, 5+ years", "Nice to have: Kafka"]
JD_TEXT = "\n".join(JD_LINES)
FULL = {"title": "Senior Backend Engineer", "company": "Acme", "industry": "FinTech", "domain": "Backend",
        "work_mode": "Hybrid", "employment_type": "Full-time", "seniority": "senior",
        "experience_years": {"min": "5+", "max": None},
        "required_skills": ["Python", "FastAPI", "python", " "], "preferred_skills": ["Kafka"],
        "responsibilities": ["Build APIs"], "salary": None}


def use_llm(reply) -> FakeChat:
    chat = FakeChat(reply)
    app.dependency_overrides[get_jd_parser_agent] = lambda: JDParserAgent(chat=chat)
    return chat


def upload(client, headers, name, data):
    return client.post(f"{URL}/upload", headers=headers, files={"file": (name, data)})


def test_upload_pdf_gives_the_cleaned_structure(client):
    headers = auth_header(client)
    chat = use_llm(FULL)

    r = upload(client, headers, "acme.pdf", make_pdf(JD_LINES))
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "created" and body["filename"] == "acme.pdf"
    jd = body["jd"]
    assert jd["title"] == "Senior Backend Engineer" and jd["industry"] == "FinTech" and jd["domain"] == "Backend"
    assert jd["work_mode"] == "hybrid" and jd["employment_type"] == "full_time"  # cleaned to allowed values
    assert jd["experience_years"] == {"min": 5.0, "max": None}
    assert jd["required_skills"] == ["Python", "FastAPI"]  # no repeat, no blank
    assert jd["salary"] is None and jd["education"] == [] and jd["location"] is None  # missing = optional

    sent = chat.calls[0][-1].content
    assert "Must have: Python, FastAPI" in sent  # the PDF's text reached the LLM


def test_same_jd_text_as_pdf_then_txt_is_parsed_once(client):
    headers = auth_header(client)
    chat = use_llm(FULL)
    first = upload(client, headers, "acme.pdf", make_pdf(JD_LINES)).json()
    again = upload(client, headers, "acme.txt", ("  " + "\n\n".join(JD_LINES) + "\n").encode()).json()

    assert again["status"] == "already_exists" and again["jd_id"] == first["jd_id"]
    assert again["jd"] == first["jd"]
    assert len(chat.calls) == 1  # whitespace differs, text is the same: no second LLM call


def test_requirements_only_jd_is_fine(client):
    use_llm({"required_skills": ["SQL"], "title": None, "responsibilities": []})
    body = upload(client, auth_header(client), "req.txt", b"Requirements: SQL").json()
    assert body["status"] == "created"
    assert body["jd"]["required_skills"] == ["SQL"] and body["jd"]["title"] is None


def test_not_a_jd_is_not_saved(client):
    headers = auth_header(client)
    use_llm({"title": None, "required_skills": [], "summary": "A recipe for cake."})
    r = upload(client, headers, "cake.txt", b"Mix flour and eggs.")
    assert r.status_code == 422 and "job description" in r.json()["detail"]
    assert client.get(f"{URL}/", headers=headers).json() == []


@pytest.mark.parametrize("name, data, message", [
    ("jd.docx", b"PK...", "use a .pdf or .txt file"),
    ("jd.pdf", b"hello", "not a PDF"),
    ("jd.txt", b"   \n  ", "no text found"),
    ("jd.txt", b"\xff\xfe\x00bad", "UTF-8"),
])
def test_bad_files(client, name, data, message):
    chat = use_llm(FULL)
    r = upload(client, auth_header(client), name, data)
    assert r.status_code == 422 and message in r.json()["detail"]
    assert chat.calls == []


def test_too_big(client, monkeypatch):
    monkeypatch.setattr("app.core.config.settings.jd_max_file_bytes", 10)
    use_llm(FULL)
    r = upload(client, auth_header(client), "jd.txt", b"x" * 50)
    assert r.status_code == 422 and "larger than" in r.json()["detail"]


def test_llm_errors_save_nothing(client):
    headers = auth_header(client)
    use_llm("Sure! The title is Backend Engineer")  # not JSON
    assert upload(client, headers, "jd.txt", JD_TEXT.encode()).status_code == 502
    use_llm(ConnectionError("deepinfra down"))
    assert upload(client, headers, "jd.txt", JD_TEXT.encode()).status_code == 503
    assert client.get(f"{URL}/", headers=headers).json() == []


def test_list_get_delete_only_own(client):
    alice, bob = auth_header(client, email="alice@x.com"), auth_header(client, email="bob@x.com")
    use_llm(FULL)
    [jd_id] = [upload(client, alice, "acme.txt", JD_TEXT.encode()).json()["jd_id"]]

    [item] = client.get(f"{URL}/", headers=alice).json()
    assert item["jd_id"] == jd_id and item["title"] == "Senior Backend Engineer" and item["company"] == "Acme"
    one = client.get(f"{URL}/{jd_id}", headers=alice).json()
    assert one["jd"]["required_skills"] == ["Python", "FastAPI"] and one["model_id"]

    assert client.get(f"{URL}/", headers=bob).json() == []
    assert client.get(f"{URL}/{jd_id}", headers=bob).status_code == 404
    assert client.delete(f"{URL}/{jd_id}", headers=bob).status_code == 404

    assert client.delete(f"{URL}/{jd_id}", headers=alice).json() == {"jd_id": jd_id, "status": "deleted"}
    assert client.get(f"{URL}/{jd_id}", headers=alice).status_code == 404
    assert client.get(f"{URL}/").status_code == 401


def test_cleaning_never_fails_on_bad_values():
    jd = JDStructure.from_raw({"title": {"x": 1}, "seniority": "rockstar", "experience_years": {"min": 8, "max": 3},
                               "salary": {"min": "25,00,000", "max": None, "currency": "INR", "period": "Annual"},
                               "required_skills": "Python", "education": None, "summary": "N/A"})
    assert jd.title is None and jd.seniority is None and jd.summary is None
    assert jd.experience_years.min == 3 and jd.experience_years.max == 8  # swapped
    assert jd.salary.min == 2_500_000 and jd.salary.period is None  # "Annual" isn't an allowed value
    assert jd.required_skills == ["Python"] and jd.education == []


def test_zero_placeholders_mean_not_stated():
    # seen for real: a JD without a salary came back with {"min": 0, "max": 0}
    jd = JDStructure.from_raw({"salary": {"min": 0, "max": 0}, "experience_years": {"min": 0, "max": 0}})
    assert jd.salary is None and jd.experience_years is None
    assert JDStructure.from_raw({"experience_years": {"min": 0, "max": 2}}).experience_years.max == 2  # 0-2 years ok


def test_text_hash_ignores_whitespace_only():
    assert text_hash("Python  developer\n\nRemote") == text_hash(" Python developer Remote ")
    assert text_hash("Python developer") != text_hash("python developer")  # case kept
