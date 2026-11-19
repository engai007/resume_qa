"""Step 5: turn line pointers into the final resume JSON."""
import re

from app.services.resumer_parser.patterns import BULLETS


def lines_in(rng: list | None, texts: list[str]) -> list[str]:
    """Turn an [start, end] range from the LLM into the actual cleaned lines."""
    if not rng or len(rng) != 2:
        return []
    a, b = int(rng[0]), int(rng[1])
    return [t for i in range(a, b + 1) if 0 <= i < len(texts)
            if (t := texts[i].lstrip(BULLETS).strip())]


SKILL_SEPARATORS = re.compile(r"[,|•·;]")  # "Python, SQL" / "Python | SQL" / "Python • SQL" ...


def split_skills(text: str) -> list[str]:
    return [s.strip() for s in SKILL_SEPARATORS.split(text) if s.strip()]


def parse_skills(skill_lines: list[str]) -> dict[str, list[str]]:
    """'Languages: Python, SQL' -> {'Languages': ['Python', 'SQL']}; lines without a category -> 'Other'."""
    skills: dict[str, list[str]] = {}
    for l in skill_lines:
        if ":" in l:
            cat, items = l.split(":", 1)
            skills[cat.strip()] = split_skills(items.strip(": "))
        else:
            skills.setdefault("Other", []).extend(split_skills(l))
    return skills


def with_details(items: list[dict] | None, texts: list[str]) -> list[dict]:
    out = []
    for it in items or []:
        d = {k: v for k, v in it.items() if k not in ("lines", "projects")}
        d["details"] = lines_in(it.get("lines"), texts)
        if "projects" in it:                       # nested company projects
            d["projects"] = with_details(it.get("projects"), texts)
        out.append(d)
    return out


def build_resume(parsed: dict, contact: dict, texts: list[str]) -> dict:
    return {
        "name": parsed.get("name"),
        "title": parsed.get("title"),
        "location": parsed.get("location"),
        **contact,
        "summary": " ".join(lines_in(parsed.get("summary"), texts)) or None,
        "skills": parse_skills(lines_in(parsed.get("skills"), texts)),
        "experience": with_details(parsed.get("experience"), texts),
        "education": parsed.get("education") or [],
        "projects": with_details(parsed.get("projects"), texts),
        "certifications": lines_in(parsed.get("certifications"), texts),
        "publications": lines_in(parsed.get("publications"), texts),
        "awards": lines_in(parsed.get("awards"), texts),
        "other_sections": with_details(parsed.get("other_sections"), texts),
    }


PERSONAL_WORDS = ("personal project", "side project", "academic project",
                  "college project", "hobby project", "self project")


def says_personal_under_header(company: str, texts: list[str]) -> bool:
    """The PDF line with the company name, or one of the 2 lines below it, says "personal project".
    Needed because the LLM sometimes leaves that line out of the entry's details:
        36 '• AI Job Search Agent: Feb 2026 - Aug 2026'
        37 '- Personal project'          <- not in any LLM range
    """
    name = company.lower()
    for i, line in enumerate(texts):
        if name in line.lower():
            window = " ".join(texts[i:i + 3]).lower()
            if any(w in window for w in PERSONAL_WORDS):
                return True
    return False


def is_personal(e: dict, texts: list[str] | None = None) -> bool:
    """No company, or the entry (or the PDF lines right under its header) says it's a personal/side project."""
    if not e.get("company"):
        return True
    text = " ".join([e.get("company") or "", e.get("role") or ""] + (e.get("details") or [])).lower()
    if any(w in text for w in PERSONAL_WORDS):
        return True
    return bool(texts) and says_personal_under_header(e["company"], texts)


def move_personal_projects(r: dict, texts: list[str] | None = None) -> dict:
    """Move personal/side projects the LLM put under experience into projects.
    `texts` = the PDF lines, to also catch a "Personal project" line the LLM didn't include."""
    keep = []
    for e in r.get("experience") or []:
        if not is_personal(e, texts):
            keep.append(e)
            continue
        # the entry becomes one project; its sub-projects are folded into its details
        details = [d for d in e.get("details") or [] if d.strip().lower() not in PERSONAL_WORDS]
        for p in e.get("projects") or []:
            details.append(p.get("name"))
            details.extend(p.get("details") or [])
        r.setdefault("projects", []).append({
            "name": e.get("company"),
            "start": e.get("start"),
            "end": e.get("end"),
            "details": details,
        })
    r["experience"] = keep
    return r
