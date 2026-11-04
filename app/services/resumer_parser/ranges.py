"""Step 4: deterministic clean-up of the LLM's line ranges."""
import logging

from app.services.resumer_parser.patterns import BULLET_RE, DATE_RANGE, SMALL_WORDS

logger = logging.getLogger(__name__)


def is_section_heading(raw: str) -> bool:
    """Short, un-bulleted, Title Case line, e.g. 'Publications', 'Honors and Awards'."""
    t = raw.strip().rstrip(":")
    words = t.split()
    return (0 < len(words) <= 4 and not BULLET_RE.match(raw.strip())
            and all(w.lower() in SMALL_WORDS or not w[0].isalpha() or w[0].isupper() for w in words))


def is_header_line(texts: list[str], i: int) -> bool:
    raw = texts[i].strip()
    t = BULLET_RE.sub("", raw)
    return bool(DATE_RANGE.search(t)) or t.endswith(":") or is_section_heading(raw)


def collect_ranges(parsed: dict) -> list[tuple[str, dict, str]]:
    """(label, owner dict, key) for every [start, end] range in the LLM output."""
    refs = []
    for e in parsed.get("experience") or []:
        refs.append((e.get("company"), e, "lines"))
        for p in e.get("projects") or []:
            refs.append((f"{e.get('company')} / {p.get('name')}", p, "lines"))
    for p in parsed.get("projects") or []:
        refs.append((p.get("name"), p, "lines"))
    for s in parsed.get("other_sections") or []:
        refs.append((s.get("heading"), s, "lines"))
    for k in ("summary", "skills", "certifications", "publications", "awards"):
        refs.append((k, parsed, k))
    return [r for r in refs if isinstance(r[1].get(r[2]), list) and len(r[1][r[2]]) == 2]


def fix_ranges(parsed: dict, texts: list[str]) -> dict:
    """Trim header lines off ranges and stop them overlapping. Mutates and returns `parsed`."""
    last = len(texts) - 1
    refs = sorted(collect_ranges(parsed), key=lambda r: int(r[1][r[2]][0]))
    for idx, (label, obj, key) in enumerate(refs):
        a, b = int(obj[key][0]), int(obj[key][1])
        orig = (a, b)
        a, b = max(0, a), min(b, last)  # guard against hallucinated line numbers

        # 1. drop leading header lines (the item's own title)
        while a < b and is_header_line(texts, a):
            a += 1
        # 2. never run past a company/date line
        for i in range(a + 1, b + 1):
            if DATE_RANGE.search(texts[i]):
                b = i - 1
                break
        # 3. never overlap the next range
        if idx + 1 < len(refs):
            nxt_obj, nxt_key = refs[idx + 1][1], refs[idx + 1][2]
            b = min(b, int(nxt_obj[nxt_key][0]) - 1)
        # 4. drop trailing header lines (they belong to the next item)
        while b > a and is_header_line(texts, b):
            b -= 1

        obj[key] = [a, b]
        if (a, b) != orig:
            logger.debug("fixed %s: %s -> %s", label, list(orig), [a, b])
    return parsed
