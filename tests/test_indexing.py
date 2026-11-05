import asyncio

from qdrant_client import models

from app.api.v1.endpoints.documents import get_parser_agent
from app.core.config import settings
from app.main import app
from app.services import indexing_service
from app.services.indexing_service import point_id
from tests.conftest import auth_header
from tests.test_documents import fake_agent, upload
from tests.resumer_parser.test_pipeline import RESUME_LINES, make_pdf


def points(client, document_id):
    found, _ = asyncio.run(client.scroll(settings.qdrant_collection, with_payload=True, scroll_filter=models.Filter(
        must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))])))
    return found


def test_upload_indexes_chunks_with_metadata(client, vector_db):
    app.dependency_overrides[get_parser_agent] = fake_agent
    [r] = upload(client, auth_header(client, email="rec@x.com"), ("jane.pdf", make_pdf(RESUME_LINES))).json()
    assert r["status"] == "created" and r["indexed"] is True

    found = points(vector_db, r["document_id"])
    assert found
    payload = found[0].payload
    assert payload["uploaded_by"] == 1
    assert payload["user_email"] == "rec@x.com"
    assert payload["embedding_model"] == settings.embedding_model
    assert payload["index_version"] == indexing_service.INDEX_VERSION
    assert payload["text"]


def test_upload_still_saved_when_indexing_fails(client, vector_db, monkeypatch):
    async def qdrant_down(texts):
        raise ConnectionError("Qdrant is down")

    monkeypatch.setattr(indexing_service, "embed_texts", qdrant_down)
    app.dependency_overrides[get_parser_agent] = fake_agent
    [r] = upload(client, auth_header(client), ("jane.pdf", make_pdf(RESUME_LINES))).json()

    assert r["status"] == "created" and r["indexed"] is False
    assert points(vector_db, r["document_id"]) == []


def test_point_id_is_stable():
    assert point_id(4, 2) == point_id(4, 2)
    assert point_id(4, 2) != point_id(4, 3)
