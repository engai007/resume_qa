from app.models import Document, User
from app.schemas.chunk import Chunk, ChunkMetadata
from app.services.resume_chunker import make_chunks


def build_metadata(doc: Document, user: User, chunk_index: int, section: str) -> ChunkMetadata:
    return ChunkMetadata(
        document_id=doc.id,
        document_hash=doc.document_hash,
        uploaded_by=user.id,
        user_email=user.email,
        uploaded_at=doc.created_at,
        candidate_name=doc.meta_data.parsed_doc_json.get("name"),
        chunk_index=chunk_index,
        section=section,
    )


def build_chunks(doc: Document, user: User) -> list[Chunk]:
    """Chunk a saved document's parsed resume and attach metadata to every chunk."""
    chunks = make_chunks(doc.meta_data.parsed_doc_json)
    return [Chunk(text=text, metadata=build_metadata(doc, user, i, section))
            for i, (section, text) in enumerate(chunks)]
