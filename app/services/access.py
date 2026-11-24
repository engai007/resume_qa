"""Who may read which resumes. The ONLY place that decides it, for both stores:
- Qdrant: every read (search, section fetch, matching) starts from access_condition(user)
- Postgres: every resume list / match read starts from accessible_documents(user)
Request filters (document_hashes, sections) can only narrow it.

Rules:
- recruiter: reads every active resume, from every user (so a JD can be matched against all of them)
- candidate: reads only the resumes they uploaded
Reading only: deleting a resume stays owner-only (document_service.delete_document), for recruiters too.

Known gap: anyone can sign up as a recruiter today, so this is only as strict as signup. Next step: recruiter role
given by an admin, and later organisations (a recruiter reads their organisation's resumes; needs organization_id
in the chunk payload, i.e. a vector migration). Only this file would change.
"""
from qdrant_client import models
from sqlalchemy import Select, select

from app.models import Document, User


def is_recruiter(user: User) -> bool:
    return getattr(user.user_type, "name", None) == "recruiter"


def access_condition(user: User) -> models.FieldCondition | None:
    """Qdrant condition for the resumes the user may read; None = no limit (recruiter)."""
    if is_recruiter(user):
        return None
    return models.FieldCondition(key="uploaded_by", match=models.MatchValue(value=user.id))


def accessible_documents(user: User) -> Select:
    """Active resumes the user may read, as a query to add filters to."""
    query = select(Document).where(Document.is_active)
    return query if is_recruiter(user) else query.where(Document.generated_by == user.id)
