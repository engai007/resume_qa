import hashlib

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.resume_parser import ResumeParserAgent
from app.models import Document, DocumentMetaData, User
from app.services.indexing_service import index_document
from app.services.resumer_parser import parse_resumes


def file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


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
