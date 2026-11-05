from datetime import datetime, timezone

from app.models import Document, DocumentMetaData, User
from app.services.chunk_service import build_chunks


def test_build_chunks_adds_metadata():
    user = User(id=7, email="rec@x.com")
    uploaded = datetime(2026, 11, 5, tzinfo=timezone.utc)
    doc = Document(id=3, document_hash="abc123", generated_by=7, created_at=uploaded,
                   meta_data=DocumentMetaData(document_plain_data=[], parsed_doc_json={
                       "name": "Jane Doe", "skills": {"Languages": ["Python"]}}))

    chunks = build_chunks(doc, user)

    assert [c.text for c in chunks] == ["Profile\nName: Jane Doe", "Skills\nLanguages: Python"]
    assert chunks[1].metadata.model_dump() == {
        "document_id": 3, "document_hash": "abc123", "uploaded_by": 7,
        "user_email": "rec@x.com", "uploaded_at": uploaded, "candidate_name": "Jane Doe", "chunk_index": 1,
    }
