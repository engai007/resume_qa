"""Step 5: turn line pointers into the final resume JSON."""
from app.services.resumer_parser.patterns import BULLETS


def lines_in(rng: list | None, texts: list[str]) -> list[str]:
    """Turn an [start, end] range from the LLM into the actual cleaned lines."""
    if not rng or len(rng) != 2:
        return []
    a, b = int(rng[0]), int(rng[1])
    return [t for i in range(a, b + 1) if 0 <= i < len(texts)
            if (t := texts[i].lstrip(BULLETS).strip())]


def parse_skills(skill_lines: list[str]) -> dict[str, list[str]]:
    """'Languages: Python, SQL' -> {'Languages': ['Python', 'SQL']}"""
    skills: dict[str, list[str]] = {}
    for l in skill_lines:
        if ":" in l:
            cat, items = l.split(":", 1)
            skills[cat.strip()] = [s.strip() for s in items.strip(": ").split(",") if s.strip()]
        else:
            skills.setdefault("Other", []).extend(s.strip() for s in l.split(",") if s.strip())
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


def is_personal(e: dict) -> bool:
    """No company, or the entry says it's a personal/side project."""
    if not e.get("company"):
        return True
    text = " ".join([e.get("company") or "", e.get("role") or ""] + (e.get("details") or [])).lower()
    return any(w in text for w in PERSONAL_WORDS)


def move_personal_projects(r: dict) -> dict:
    """Move personal/side projects the LLM put under experience into projects."""
    keep = []
    for e in r.get("experience") or []:
        if not is_personal(e):
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
