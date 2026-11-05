def lines_to_text(lines):
    return "\n".join(f"- {l}" for l in lines or [])


def make_chunks(r: dict) -> list[str]:
    """Parsed resume dict -> list of text chunks (one per section / company / project)."""
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

    # skills
    if r.get("skills"):
        chunks.append("Skills\n" + "\n".join(f"{cat}: {', '.join(items)}"
                                             for cat, items in r["skills"].items()))

    # experience: ONE chunk per company, with all its projects inside
    for e in r.get("experience") or []:
        text = f"Experience: {e.get('company')}\nStart: {e.get('start')}\nEnd: {e.get('end')}"
        if e.get("role"):
            text += f"\nRole: {e['role']}"
        if e.get("details"):
            text += "\n" + lines_to_text(e["details"])
        for p in e.get("projects") or []:
            text += f"\n\nProject: {p.get('name')}\n{lines_to_text(p.get('details'))}"
        chunks.append(text)

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
