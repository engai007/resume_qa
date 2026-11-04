from app.agents.registry import agent_registry
from app.llm.client import LLMClient
from app.memory.store import memory_store
from app.schemas.agent import AgentRequest, AgentResponse
from app.tools.registry import tool_registry


async def run_agent(agent_name: str, request: AgentRequest) -> AgentResponse:
    agent_cls = agent_registry.get(agent_name)
    agent = agent_cls(llm=LLMClient(), memory=memory_store, tools=tool_registry.all())
    output = await agent.run(request.input, request.session_id)
    return AgentResponse(agent=agent_name, output=output, session_id=request.session_id)
