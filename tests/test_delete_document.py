import asyncio

from qdrant_client import models

from app.api.v1.endpoints.documents import get_parser_agent
from app.main import app
from app.services import indexing_service
from tests.conftest import auth_header
from tests.resumer_parser.test_pipeline import RESUME_LINES, make_pdf
from tests.test_documents import fake_agent, upload


def count(qdrant, collection, document_id) -> int:
    flt = models.Filter(must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))])
    return asyncio.run(qdrant.count(collection, count_filter=flt)).count


def collections(qdrant) -> list[str]:
    return [c.name for c in asyncio.run(qdrant.get_collections()).collections]


def test_delete_removes_chunks_everywhere_and_allows_reupload(client, vector_db):
    app.dependency_overrides[get_parser_agent] = fake_agent
    headers = auth_header(client)
    pdf = make_pdf(RESUME_LINES)
    [doc] = upload(client, headers, ("jane.pdf", pdf)).json()
    doc_id = doc["document_id"]
    # pretend an old rollback collection still has this resume too
    asyncio.run(indexing_service.qdrant.upsert("resume_chunks_v3", points=[models.PointStruct(
        id=1, vector=[1.0] + [0.0] * 767, payload={"document_id": doc_id})]))
    assert count(vector_db, "resume_chunks", doc_id) > 0

    r = client.delete(f"/api/v1/documents/{doc_id}", headers=headers)
    assert r.status_code == 200 and r.json() == {"document_id": doc_id, "status": "deleted"}
    for name in collections(vector_db):
        assert count(vector_db, name, doc_id) == 0, name

    assert client.delete(f"/api/v1/documents/{doc_id}", headers=headers).status_code == 404  # already gone
    assert upload(client, headers, ("jane.pdf", pdf)).json()[0]["status"] == "created"  # not "already_exists"


def test_cannot_delete_someone_elses_resume(client, vector_db):
    app.dependency_overrides[get_parser_agent] = fake_agent
    alice, bob = auth_header(client, email="alice@x.com"), auth_header(client, email="bob@x.com")
    [doc] = upload(client, alice, ("a.pdf", make_pdf(RESUME_LINES))).json()

    assert client.delete(f"/api/v1/documents/{doc['document_id']}", headers=bob).status_code == 404
    assert client.delete(f"/api/v1/documents/{doc['document_id']}", headers={}).status_code == 401
    assert count(vector_db, "resume_chunks", doc["document_id"]) > 0  # untouched


def test_nothing_deleted_when_qdrant_is_down(client, vector_db, monkeypatch):
    app.dependency_overrides[get_parser_agent] = fake_agent
    headers = auth_header(client)
    pdf = make_pdf(RESUME_LINES)
    [doc] = upload(client, headers, ("jane.pdf", pdf)).json()

    async def qdrant_down(document_id):
        raise ConnectionError("Qdrant is down")

    monkeypatch.setattr("app.services.document_service.delete_document_vectors", qdrant_down)
    assert client.delete(f"/api/v1/documents/{doc['document_id']}", headers=headers).status_code == 503
    assert upload(client, headers, ("jane.pdf", pdf)).json()[0]["status"] == "already_exists"  # row still there
