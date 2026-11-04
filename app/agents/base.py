from abc import ABC, abstractmethod

from app.llm.client import LLMClient
from app.memory.store import MemoryStore
from app.tools.base import BaseTool


class BaseAgent(ABC):
    name: str = "base"
    system_prompt: str = "You are a helpful assistant."

    def __init__(self, llm: LLMClient, memory: MemoryStore, tools: list[BaseTool] | None = None):
        self.llm = llm
        self.memory = memory
        self.tools = tools or []

    @abstractmethod
    async def run(self, user_input: str, session_id: str | None = None) -> str:
        ...
