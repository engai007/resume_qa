from app.core.config import settings


class LLMClient:
    """Thin wrapper around your LLM provider. Swap the implementation of `chat`."""

    def __init__(self, model: str = settings.llm_model, api_key: str = settings.llm_api_key):
        self.model = model
        self.api_key = api_key

    async def chat(self, system: str, messages: list[dict]) -> str:
        # TODO: call the real provider SDK here.
        return f"[stub:{self.model}] {messages[-1]['content']}"
