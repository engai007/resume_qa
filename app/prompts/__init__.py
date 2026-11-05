from functools import cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


@cache
def load_prompt(name: str) -> str:
    """load_prompt("resume_parser/system") reads app/prompts/resume_parser/system.md"""
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8").strip()
