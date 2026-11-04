from app.agents.base import BaseAgent
from app.memory.store import memory_store


class QAAgent(BaseAgent):
    name = "qa"
    prompt = "qa_agent/system"

    async def run(self, user_input: str, session_id: str | None = None) -> str:
        history = memory_store.get(session_id) if session_id else []
        reply, _ = await self.call_llm([*history, {"role": "user", "content": user_input}])
        if session_id:
            memory_store.append(session_id, {"role": "user", "content": user_input})
            memory_store.append(session_id, {"role": "assistant", "content": reply})
        return reply
