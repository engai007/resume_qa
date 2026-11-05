from app.models import Document, User
from app.schemas.chunk import Chunk, ChunkMetadata
from app.services.resume_chunker import make_chunks


def build_metadata(doc: Document, user: User, chunk_index: int) -> ChunkMetadata:
    return ChunkMetadata(
        document_id=doc.id,
        document_hash=doc.document_hash,
        uploaded_by=user.id,
        user_email=user.email,
        uploaded_at=doc.created_at,
        chunk_index=chunk_index,
    )


def build_chunks(doc: Document, user: User) -> list[Chunk]:
    """Chunk a saved document's parsed resume and attach metadata to every chunk."""
    texts = make_chunks(doc.meta_data.parsed_doc_json)
    return [Chunk(text=t, metadata=build_metadata(doc, user, i)) for i, t in enumerate(texts)]
