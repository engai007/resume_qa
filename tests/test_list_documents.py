from app.api.v1.endpoints.documents import get_parser_agent
from app.main import app
from app.services.document_service import file_hash
from tests.conftest import auth_header
from tests.resumer_parser.test_pipeline import RESUME_LINES, make_pdf
from tests.test_documents import fake_agent, upload

URL = "/api/v1/documents/"


def test_lists_only_my_resumes_newest_first(client):
    app.dependency_overrides[get_parser_agent] = fake_agent
    alice, bob = auth_header(client, email="alice@x.com"), auth_header(client, email="bob@x.com")
    one, two = make_pdf(RESUME_LINES), make_pdf(RESUME_LINES + ["Extra line"])
    [a1] = upload(client, alice, ("one.pdf", one)).json()
    [a2] = upload(client, alice, ("two.pdf", two)).json()
    upload(client, bob, ("bob.pdf", make_pdf(RESUME_LINES + ["Bob"])))

    docs = client.get(URL, headers=alice).json()
    assert [d["document_id"] for d in docs] == [a2["document_id"], a1["document_id"]]  # newest first, no bob
    first = docs[0]
    assert first["document_hash"] == file_hash(two)
    assert first["candidate_name"] == "Jane Doe"
    assert first["indexed"] is True and first["index_error"] is None
    assert first["uploaded_at"]

    assert len(client.get(URL, headers=bob).json()) == 1


def test_paging_and_deleted_resumes(client):
    app.dependency_overrides[get_parser_agent] = fake_agent
    headers = auth_header(client)
    upload(client, headers, ("one.pdf", make_pdf(RESUME_LINES)), ("two.pdf", make_pdf(RESUME_LINES + ["x"])))

    assert len(client.get(URL, headers=headers, params={"limit": 1}).json()) == 1
    assert len(client.get(URL, headers=headers, params={"limit": 1, "offset": 1}).json()) == 1
    assert client.get(URL, headers=headers, params={"limit": 500}).status_code == 422

    [newest, _] = client.get(URL, headers=headers).json()
    client.delete(f"{URL}{newest['document_id']}", headers=headers)
    assert [d["document_id"] for d in client.get(URL, headers=headers).json()] != [newest["document_id"]]
    assert len(client.get(URL, headers=headers).json()) == 1


def test_empty_list_and_login_needed(client):
    assert client.get(URL, headers=auth_header(client)).json() == []
    assert client.get(URL).status_code == 401
