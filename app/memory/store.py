from collections import defaultdict


class MemoryStore:
    """In-memory conversation history. Replace with Redis/DB for persistence."""

    def __init__(self) -> None:
        self._data: dict[str, list[dict]] = defaultdict(list)

    def get(self, session_id: str) -> list[dict]:
        return list(self._data[session_id])

    def append(self, session_id: str, message: dict) -> None:
        self._data[session_id].append(message)


memory_store = MemoryStore()
