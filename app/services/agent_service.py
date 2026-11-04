from app.agents.registry import AGENTS
from app.schemas.agent import AgentRequest, AgentResponse


async def run_agent(agent_name: str, request: AgentRequest) -> AgentResponse:
    agent = AGENTS[agent_name]()
    output = await agent.run(request.input, request.session_id)
    return AgentResponse(agent=agent_name, output=output, session_id=request.session_id)
