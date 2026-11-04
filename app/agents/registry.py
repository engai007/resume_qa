from app.agents.base import BaseAgent
from app.agents.qa_agent import QAAgent


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, type[BaseAgent]] = {}

    def register(self, agent_cls: type[BaseAgent]) -> None:
        self._agents[agent_cls.name] = agent_cls

    def get(self, name: str) -> type[BaseAgent]:
        return self._agents[name]

    def names(self) -> list[str]:
        return list(self._agents)


agent_registry = AgentRegistry()
agent_registry.register(QAAgent)
