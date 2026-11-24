import hashlib

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.agents.resume_parser import ResumeParserAgent
from app.models import Document, DocumentMetaData, User
from app.services.access import accessible_documents
from app.services.indexing_service import INDEX_VERSION, delete_document_vectors, index_document
from app.services.resumer_parser import parse_resumes


def file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


async def list_documents(db: AsyncSession, user: User, limit: int = 50, offset: int = 0) -> list[dict]:
    """The resumes the user can read (recruiter: everyone's; candidate: own), newest first."""
    docs = await db.scalars(
        accessible_documents(user)
        .options(selectinload(Document.meta_data))
        .order_by(Document.created_at.desc(), Document.id.desc()).limit(limit).offset(offset))
    return [
        {
            "document_id": d.id,
            "document_hash": d.document_hash,
            "candidate_name": (d.meta_data.parsed_doc_json or {}).get("name") if d.meta_data else None,
            "uploaded_at": d.created_at,
            "indexed": d.indexed_version == INDEX_VERSION,
            "indexed_version": d.indexed_version,
            "index_error": d.index_error,
            "uploaded_by_me": d.generated_by == user.id,  # only these can be deleted
        }
        for d in docs
    ]


class VectorDBUnavailable(Exception):
    """Qdrant couldn't delete the chunks, so nothing was deleted."""


async def delete_document(db: AsyncSession, user: User, document_id: int) -> bool:
    """Delete one of the user's OWN resumes (recruiters too: reading all doesn't mean deleting all):
    chunks in Qdrant, then metadata + document in Postgres.
    False if the user has no such document. Qdrant goes first: if it's down this raises and nothing is
    deleted, so we never end up with a removed resume whose personal data is still searchable."""
    doc = await db.scalar(select(Document).where(Document.id == document_id, Document.generated_by == user.id)
                          .options(selectinload(Document.meta_data)))
    if not doc:
        return False
    try:
        await delete_document_vectors(doc.id)
    except Exception as e:
        raise VectorDBUnavailable(str(e)) from e
    if doc.meta_data:
        await db.delete(doc.meta_data)
    await db.delete(doc)
    await db.commit()
    return True


async def upload_documents(db: AsyncSession, user: User, files: list[tuple[str, bytes]],
                           agent: ResumeParserAgent) -> list[dict]:
    """Save new resumes for a user, then index them in Qdrant.
    Files the user already uploaded are not parsed again."""
    user_id = user.id
    hashes = [file_hash(content) for _, content in files]

    # documents this user already has
    rows = await db.scalars(select(Document).where(Document.generated_by == user_id,
                                                   Document.document_hash.in_(hashes)))
    existing = {doc.document_hash: doc for doc in rows}

    # new files to parse (a file sent twice in one upload is parsed once)
    to_parse = {}
    for (name, content), h in zip(files, hashes):
        if h not in existing and content.startswith(b"%PDF"):
            to_parse[h] = content
    parsed = await parse_resumes(to_parse, agent)

    created = {}
    for h, result in parsed.items():
        if isinstance(result, Exception):
            continue
        lines, resume = result
        doc = Document(document_hash=h, generated_by=user_id,
                       meta_data=DocumentMetaData(document_plain_data=lines, parsed_doc_json=resume))
        db.add(doc)
        created[h] = doc
    await db.commit()

    # a failed index does not fail the upload: the document is saved, indexed_version stays NULL
    for doc in created.values():
        await index_document(db, doc, user)

    results = []
    for (name, content), h in zip(files, hashes):
        if h in existing:
            doc = existing[h]
            results.append({"filename": name, "status": "already_exists", "document_id": doc.id,
                            "indexed": doc.indexed_version is not None})
        elif h in created:
            doc = created[h]
            results.append({"filename": name, "status": "created", "document_id": doc.id,
                            "indexed": doc.indexed_version is not None})
        elif not content.startswith(b"%PDF"):
            results.append({"filename": name, "status": "failed", "error": "not a PDF file"})
        else:
            results.append({"filename": name, "status": "failed", "error": str(parsed[h])})
    return results
