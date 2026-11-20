"""Who may read which resumes. The ONLY place that decides it: every Qdrant read (search, section fetch)
starts from access_condition(user), and request filters (document_hashes, sections) can only narrow it.

Today: a user reads only the resumes they uploaded.
Later (roles, see docs/planner_plan.md section 4): e.g. a recruiter reads every resume of their organisation.
Then only this function changes (+ organization_id in the chunk payload, which needs a vector migration).
"""
from qdrant_client import models

from app.models import User


def access_condition(user: User) -> models.FieldCondition:
    return models.FieldCondition(key="uploaded_by", match=models.MatchValue(value=user.id))
