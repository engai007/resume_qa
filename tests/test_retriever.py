from app.api.v1.endpoints.documents import get_parser_agent
from app.main import app
from app.services.document_service import file_hash
from tests.conftest import auth_header
from tests.resumer_parser.test_pipeline import RESUME_LINES, make_pdf
from tests.test_documents import fake_agent, upload

URL = "/api/v1/retriever/search"


def search(client, headers, **body):
    return client.post(URL, headers=headers, json={"query": "python", "top_k": 50, **body})


def test_search_only_returns_own_resumes(client):
    app.dependency_overrides[get_parser_agent] = fake_agent
    alice, bob = auth_header(client, email="alice@x.com"), auth_header(client, email="bob@x.com")
    [a] = upload(client, alice, ("a.pdf", make_pdf(RESUME_LINES))).json()
    [b] = upload(client, bob, ("b.pdf", make_pdf(RESUME_LINES + ["Extra line"]))).json()

    r = search(client, alice).json()
    assert r["query"] == "python"
    assert r["results"]
    assert {x["document_id"] for x in r["results"]} == {a["document_id"]}

    assert {x["document_id"] for x in search(client, bob).json()["results"]} == {b["document_id"]}


def test_search_filtered_by_document_hash(client):
    app.dependency_overrides[get_parser_agent] = fake_agent
    headers = auth_header(client)
    pdf1, pdf2 = make_pdf(RESUME_LINES), make_pdf(RESUME_LINES + ["Extra line"])
    upload(client, headers, ("one.pdf", pdf1), ("two.pdf", pdf2))

    assert len({x["document_hash"] for x in search(client, headers).json()["results"]}) == 2
    only = search(client, headers, document_hashes=[file_hash(pdf2)]).json()["results"]
    assert only and {x["document_hash"] for x in only} == {file_hash(pdf2)}


def test_search_respects_top_k_and_needs_login(client):
    app.dependency_overrides[get_parser_agent] = fake_agent
    headers = auth_header(client)
    upload(client, headers, ("one.pdf", make_pdf(RESUME_LINES)))

    assert len(search(client, headers, top_k=1).json()["results"]) == 1
    assert search(client, {}).status_code == 401
