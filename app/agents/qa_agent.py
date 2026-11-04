from app.agents.base import BaseAgent


class QAAgent(BaseAgent):
    name = "qa"
    system_prompt = "You answer questions accurately and concisely."

    async def run(self, user_input: str, session_id: str | None = None) -> str:
        history = self.memory.get(session_id) if session_id else []
        messages = [*history, {"role": "user", "content": user_input}]
        reply = await self.llm.chat(system=self.system_prompt, messages=messages)
        if session_id:
            self.memory.append(session_id, {"role": "user", "content": user_input})
            self.memory.append(session_id, {"role": "assistant", "content": reply})
        return reply
