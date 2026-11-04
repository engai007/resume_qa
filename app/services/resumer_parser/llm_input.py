"""Step 2: compress lines into the numbered input the LLM sees."""
import re

from app.services.resumer_parser.patterns import BULLET_RE, DATE_RANGE, EMAIL, GITHUB, LINKEDIN


def compact_numbered(texts: list[str], max_words: int = 8) -> str:
    """Numbered lines for the LLM. Keeps the bullet symbol (it shows nesting),
    truncates long description lines, keeps date lines whole."""
    out = []
    for i, t in enumerate(texts):
        m = BULLET_RE.match(t)
        prefix = f"{m.group(1)[0]} " if m else ""
        t = t[m.end():] if m else t
        for p in (EMAIL, LINKEDIN, GITHUB):
            t = p.sub("", t)
        t = re.sub(r"https?://\S+", "", t).strip()
        if not t:
            continue
        words = t.split()
        if len(words) > max_words and not DATE_RANGE.search(t):
            t = " ".join(words[:max_words]) + "…"
        out.append(f"{i}|{prefix}{t}")
    return "\n".join(out)
