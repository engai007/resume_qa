SKILLS_PER_CHUNK = 8  # a long skill list is split, so each chunk's vector stays about a few skills


def lines_to_text(lines):
    return "\n".join(f"- {l}" for l in lines or [])


def in_groups(items: list, size: int) -> list[list]:
    """[1..10], 8 -> [[1..8], [9, 10]]"""
    return [items[i:i + size] for i in range(0, len(items), size)]


def make_chunks(r: dict) -> list[str]:
    """Parsed resume dict -> list of text chunks. One topic per chunk, so its vector stays focused:
    profile, each skill category (in groups), each company project, each education, ..."""
    chunks = []

    # profile
    fields = [
        ("Name", r.get("name")),
        ("Title", r.get("title")),
        ("Location", r.get("location")),
        ("Email", r.get("email")),
        ("Phone", r.get("phone")),
        ("LinkedIn", r.get("linkedin")),
        ("GitHub", r.get("github")),
    ]
    profile = "Profile\n" + "\n".join(f"{k}: {v}" for k, v in fields if v)
    if r.get("summary"):
        profile += f"\nSummary: {r['summary']}"
    chunks.append(profile)

    # skills: one chunk per category; long lists (e.g. uncategorized "Other") split into groups
    for cat, items in (r.get("skills") or {}).items():
        label = "Skills" if cat == "Other" else f"Skills - {cat}"
        for group in in_groups(items, SKILLS_PER_CHUNK):
            chunks.append(f"{label}: {', '.join(group)}")

    # experience: one chunk per project, each repeating the company header
    for e in r.get("experience") or []:
        header = f"Experience: {e.get('company')}\nStart: {e.get('start')}\nEnd: {e.get('end')}"
        if e.get("role"):
            header += f"\nRole: {e['role']}"
        projects = e.get("projects") or []
        if e.get("details") or not projects:  # bullets not under a project, or a company without projects
            chunks.append(header + (f"\n{lines_to_text(e['details'])}" if e.get("details") else ""))
        for p in projects:
            chunks.append(f"{header}\nProject: {p.get('name')}\n{lines_to_text(p.get('details'))}")

    # education
    for ed in r.get("education") or []:
        chunks.append(f"Education: {ed.get('degree')} {ed.get('field') or ''}, "
                      f"{ed.get('institution')}\nStart: {ed.get('start')}\nEnd: {ed.get('end')}")

    # personal projects
    for p in r.get("projects") or []:
        chunks.append(f"Project: {p.get('name')}\nStart: {p.get('start')}\nEnd: {p.get('end')}\n"
                      f"{lines_to_text(p.get('details'))}")

    # other sections
    for key in ("certifications", "publications", "awards"):
        if r.get(key):
            chunks.append(f"{key.title()}\n{lines_to_text(r[key])}")
    for s in r.get("other_sections") or []:
        chunks.append(f"{s.get('heading')}\n{lines_to_text(s.get('details'))}")

    return chunks
