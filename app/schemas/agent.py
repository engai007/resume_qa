from pydantic import BaseModel


class AgentRequest(BaseModel):
    input: str
    session_id: str | None = None


class AgentResponse(BaseModel):
    agent: str
    output: str
    session_id: str | None = None
