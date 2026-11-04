from fastapi import APIRouter, HTTPException

from app.agents.registry import AGENTS
from app.schemas.agent import AgentRequest, AgentResponse
from app.services.agent_service import run_agent

router = APIRouter()


@router.get("/")
async def list_agents() -> list[dict]:
    return [{"name": a.name, "prompt": a.prompt, "model_id": a.model_id, "provider": a.provider}
            for a in AGENTS.values()]


# @router.post("/{agent_name}/run", response_model=AgentResponse)
# async def run(agent_name: str, request: AgentRequest) -> AgentResponse:
#     if agent_name not in AGENTS:
#         raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
#     return await run_agent(agent_name, request)
