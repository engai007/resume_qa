import re
from datetime import date

SKILLS_PER_CHUNK = 8  # a long skill list is split, so each chunk's vector stays about a few skills

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
PRESENT_WORDS = ("present", "current", "now", "till date", "to date", "ongoing")


def parse_month(text, is_end: bool, today: date) -> int | None:
    """A resume date -> month number (year * 12 + month - 1), so two dates can be subtracted.
    "Mar 2020" / "March 2020" / "03/2020" / "2020-03" / "2020" / "present". None if it can't be read.
    Year only: start = January, end = December."""
    t = (text or "").strip().lower()
    if not t:
        return None
    if any(w in t for w in PRESENT_WORDS):
        return today.year * 12 + today.month - 1
    if m := re.search(r"([a-z]{3,})\.?\s*,?\s*(\d{4})", t):  # Mar 2020, March 2020, Sept. 2020
        if m.group(1)[:3] in MONTHS:
            return int(m.group(2)) * 12 + MONTHS.index(m.group(1)[:3])
    if m := re.fullmatch(r"(\d{1,2})[/-](\d{4})", t):  # 03/2020
        return int(m.group(2)) * 12 + int(m.group(1)) - 1 if 1 <= int(m.group(1)) <= 12 else None
    if m := re.fullmatch(r"(\d{4})[/-](\d{1,2})", t):  # 2020-03
        return int(m.group(1)) * 12 + int(m.group(2)) - 1 if 1 <= int(m.group(2)) <= 12 else None
    if m := re.fullmatch(r"(\d{4})", t):  # 2020
        return int(m.group(1)) * 12 + (11 if is_end else 0)
    return None


def job_span(e: dict, today: date) -> tuple[int, int] | None:
    """(start, end) month numbers of a job, or None if its dates can't be read."""
    start, end = parse_month(e.get("start"), False, today), parse_month(e.get("end"), True, today)
    if start is None or end is None or end < start:
        return None
    return start, end


def total_months(spans: list[tuple[int, int]]) -> int:
    """Months worked; overlapping jobs count once, gaps don't count.
    [(Mar 2020, Dec 2022), (Dec 2022, Jun 2023)] -> 33 + 6 = 39"""
    total, cur_start, cur_end = 0, None, None
    for start, end in sorted(spans):
        if cur_end is not None and start <= cur_end:  # overlaps the previous one: stretch it
            cur_end = max(cur_end, end)
            continue
        if cur_end is not None:
            total += cur_end - cur_start
        cur_start, cur_end = start, end
    if cur_end is not None:
        total += cur_end - cur_start
    return total


def as_years(months: int) -> str:
    """80 -> "6 years 8 months", 12 -> "1 year", 9 -> "9 months" """
    y, m = divmod(months, 12)
    parts = [f"{y} year{'s' if y > 1 else ''}"] if y else []
    if m or not y:
        parts.append(f"{m} month{'s' if m != 1 else ''}")
    return " ".join(parts)


def month_name(n: int) -> str:
    return f"{MONTHS[n % 12].title()} {n // 12}"


def experience_summary(r: dict, today: date) -> str | None:
    """One chunk with total work experience, calculated here so the LLM never adds up dates.
    Only company jobs (r["experience"]); personal projects (r["projects"]) are not counted."""
    jobs = r.get("experience") or []
    if not jobs:
        return None
    # Only the total, no per-company lines: those made this chunk match every "work at <company>" question.
    # Each company's dates are already in its own Experience chunk.
    spans = [s for s in (job_span(e, today) for e in jobs) if s]
    unclear = len(jobs) - len(spans)
    if not spans:
        return "Total experience: unknown (job dates unclear)"
    companies = f"{len(jobs)} compan{'y' if len(jobs) == 1 else 'ies'}"
    note = f"; {unclear} with unclear dates not counted" if unclear else ""
    # no "(company jobs only, personal projects not counted)" note: the words "company" and "project"
    # made this chunk rank above real company / project chunks
    return (f"Total experience: {as_years(total_months(spans))}\n"
            f"Since: {month_name(min(s for s, _ in spans))}, at {companies}{note}")


def lines_to_text(lines):
    return "\n".join(f"- {l}" for l in lines or [])


def in_groups(items: list, size: int) -> list[list]:
    """[1..10], 8 -> [[1..8], [9, 10]]"""
    return [items[i:i + size] for i in range(0, len(items), size)]


# `section` of each chunk, stored in Qdrant so search can filter on it (e.g. every project chunk of a resume)
SECTIONS = ("profile", "skills", "experience", "project", "education", "personal_project",
            "certifications", "publications", "awards", "other", "experience_summary")


def make_chunks(r: dict, today: date | None = None) -> list[tuple[str, str]]:
    """Parsed resume dict -> list of (section, text) chunks. One topic per chunk, so its vector stays focused:
    profile, each skill category (in groups), each company project, each education, ...,
    and last an experience summary with the total. `today` is what "present" means (default: today).
    section "experience" = company bullets outside any project, "project" = one project at a company."""
    today = today or date.today()
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
    chunks.append(("profile", profile))

    # skills: one chunk per category; long lists (e.g. uncategorized "Other") split into groups
    for cat, items in (r.get("skills") or {}).items():
        label = "Skills" if cat == "Other" else f"Skills - {cat}"
        for group in in_groups(items, SKILLS_PER_CHUNK):
            chunks.append(("skills", f"{label}: {', '.join(group)}"))

    # experience: one chunk per project, each repeating the company header
    for e in r.get("experience") or []:
        header = f"Experience: {e.get('company')}\nStart: {e.get('start')}\nEnd: {e.get('end')}"
        if e.get("role"):
            header += f"\nRole: {e['role']}"
        projects = e.get("projects") or []
        if e.get("details") or not projects:  # bullets not under a project, or a company without projects
            chunks.append(("experience", header + (f"\n{lines_to_text(e['details'])}" if e.get("details") else "")))
        for p in projects:
            chunks.append(("project", f"{header}\nProject: {p.get('name')}\n{lines_to_text(p.get('details'))}"))

    # education
    for ed in r.get("education") or []:
        chunks.append(("education", f"Education: {ed.get('degree')} {ed.get('field') or ''}, "
                                    f"{ed.get('institution')}\nStart: {ed.get('start')}\nEnd: {ed.get('end')}"))

    # personal projects: labelled, so neither search nor the LLM takes them for a job
    for p in r.get("projects") or []:
        chunks.append(("personal_project", f"Personal project: {p.get('name')}\nStart: {p.get('start')}\n"
                                           f"End: {p.get('end')}\n{lines_to_text(p.get('details'))}"))

    # other sections
    for key in ("certifications", "publications", "awards"):
        if r.get(key):
            chunks.append((key, f"{key.title()}\n{lines_to_text(r[key])}"))
    for s in r.get("other_sections") or []:
        chunks.append(("other", f"{s.get('heading')}\n{lines_to_text(s.get('details'))}"))

    # last, so the chunk numbers above don't move
    if summary := experience_summary(r, today):
        chunks.append(("experience_summary", summary))
    return chunks
