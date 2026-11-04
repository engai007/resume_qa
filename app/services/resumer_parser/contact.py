"""Deterministic contact extraction (no LLM)."""
import re

from app.services.resumer_parser.patterns import EMAIL, GITHUB, LINKEDIN, PHONE


def find_phone(text: str) -> str | None:
    for m in PHONE.finditer(text):
        if len(re.sub(r"\D", "", m.group(0))) >= 10:  # skip date ranges like "2020 - 2022"
            return m.group(0).strip()
    return None


def extract_contact(texts: list[str]) -> dict:
    full = "\n".join(texts)
    first = lambda p: (m.group(0) if (m := p.search(full)) else None)
    return {
        "email": first(EMAIL),
        "phone": find_phone(full),
        "linkedin": first(LINKEDIN),
        "github": first(GITHUB),
    }
