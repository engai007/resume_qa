from fastapi import APIRouter

from app.agents.registry import AGENTS

router = APIRouter()


@router.get("/")
async def list_agents() -> list[dict]:
    return [{"name": a.name, "prompt": a.prompt, "model_id": a.model_id, "provider": a.provider}
            for a in AGENTS.values()]
