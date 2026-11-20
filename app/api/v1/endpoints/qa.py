import logging

from fastapi import APIRouter, Depends, HTTPException

from app.agents.planner_agent import PlannerAgent
from app.agents.qa_agent import InvalidAnswer, QAAgent
from app.api.deps import get_current_user
from app.models import User
from app.schemas.qa import AskRequest, AskResponse
from app.services.qa_service import AnswerUnavailable, answer_question

logger = logging.getLogger(__name__)
router = APIRouter()


def get_qa_agent() -> QAAgent:
    return QAAgent()


def get_planner_agent() -> PlannerAgent:
    return PlannerAgent()


@router.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, user: User = Depends(get_current_user), agent: QAAgent = Depends(get_qa_agent),
              planner: PlannerAgent = Depends(get_planner_agent)):
    """Answer a question from your resumes. found=false when they don't answer it.
    Each fact in the answer has an [n] marker pointing at a source chunk.
    A small planner first decides how to fetch: search (top_k chunks) or every chunk of some sections
    ("list all projects"). `plan` in the reply shows what it chose; planner=false = plain search."""
    try:
        return await answer_question(user, body.question, agent, body.top_k, body.document_hashes,
                                     planner if body.planner else None)
    except InvalidAnswer as e:
        logger.warning("qa: invalid LLM reply: %s", e)
        raise HTTPException(502, "Answer service gave an invalid reply. Try again.")
    except AnswerUnavailable as e:
        logger.warning("qa: %s", e)
        raise HTTPException(503, "Answer service unavailable. Try again later.")
