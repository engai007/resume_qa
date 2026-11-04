from fastapi import APIRouter, HTTPException

from app.agents.registry import agent_registry
from app.schemas.agent import AgentRequest, AgentResponse
from app.services.agent_service import run_agent

router = APIRouter()


@router.get("/")
async def list_agents() -> list[str]:
    return agent_registry.names()


@router.post("/{agent_name}/run", response_model=AgentResponse)
async def run(agent_name: str, request: AgentRequest) -> AgentResponse:
    if agent_name not in agent_registry.names():
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    return await run_agent(agent_name, request)
