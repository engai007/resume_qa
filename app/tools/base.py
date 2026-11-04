from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    name: str
    description: str
    input_schema: dict[str, Any] = {"type": "object", "properties": {}}

    @abstractmethod
    async def run(self, **kwargs: Any) -> str:
        ...
