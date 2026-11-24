"""Job description structure. Every field is optional: JDs vary a lot (some list only requirements).
See docs/jd_parser_plan.md."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel

WorkMode = Literal["onsite", "remote", "hybrid"]
EmploymentType = Literal["full_time", "part_time", "contract", "internship"]
Seniority = Literal["intern", "junior", "mid", "senior", "lead", "principal"]
SalaryPeriod = Literal["year", "month", "hour"]

MAX_TEXT = 300  # one short field (title, a skill, ...)
MAX_SUMMARY = 1000
MAX_ITEMS = 50  # per list


class YearsRange(BaseModel):
    min: float | None = None  # "5+ years" -> min 5, max None
    max: float | None = None


class Salary(BaseModel):
    min: float | None = None
    max: float | None = None
    currency: str | None = None  # "INR", "USD", ...
    period: SalaryPeriod | None = None


class JDStructure(BaseModel):
    title: str | None = None
    company: str | None = None
    industry: str | None = None  # the company's sector: FinTech, Healthcare, ...
    domain: str | None = None  # the role's area: Backend, Data Engineering, ...
    location: str | None = None
    work_mode: WorkMode | None = None
    employment_type: EmploymentType | None = None
    seniority: Seniority | None = None
    experience_years: YearsRange | None = None
    education: list[str] = []
    required_skills: list[str] = []
    preferred_skills: list[str] = []
    responsibilities: list[str] = []
    summary: str | None = None
    salary: Salary | None = None

    def is_empty(self) -> bool:
        """Nothing a JD would have: the file probably isn't a job description."""
        return not (self.title or self.required_skills or self.preferred_skills or self.responsibilities
                    or self.experience_years or self.education)

    @classmethod
    def from_raw(cls, raw: dict) -> "JDStructure":
        """Clean the LLM's JSON. Never fails on a bad value: it becomes null / is dropped.
        Only whitelisted enum values get through; skills are trimmed and de-duplicated."""
        years = clean_range(raw.get("experience_years"))
        salary = raw.get("salary") if isinstance(raw.get("salary"), dict) else {}
        pay = clean_range(salary)
        salary_out = None
        if pay:
            salary_out = Salary(min=pay.min, max=pay.max, currency=clean_text(salary.get("currency"), 10),
                                period=pick(salary.get("period"), SalaryPeriod))
        return cls(
            title=clean_text(raw.get("title")),
            company=clean_text(raw.get("company")),
            industry=clean_text(raw.get("industry")),
            domain=clean_text(raw.get("domain")),
            location=clean_text(raw.get("location")),
            work_mode=pick(raw.get("work_mode"), WorkMode),
            employment_type=pick(raw.get("employment_type"), EmploymentType),
            seniority=pick(raw.get("seniority"), Seniority),
            experience_years=years,
            education=clean_list(raw.get("education")),
            required_skills=clean_list(raw.get("required_skills")),
            preferred_skills=clean_list(raw.get("preferred_skills")),
            responsibilities=clean_list(raw.get("responsibilities")),
            summary=clean_text(raw.get("summary"), MAX_SUMMARY),
            salary=salary_out,
        )


def clean_text(value, limit: int = MAX_TEXT) -> str | None:
    if value is None or isinstance(value, (dict, list)):
        return None
    text = " ".join(str(value).split())[:limit]
    return text if text and text.lower() not in ("null", "none", "n/a", "not mentioned", "not specified") else None


def clean_list(value) -> list[str]:
    """str or list -> trimmed, no empty items, no repeats (case-insensitive), at most MAX_ITEMS."""
    items = [value] if isinstance(value, str) else value if isinstance(value, list) else []
    out, seen = [], set()
    for item in items:
        text = clean_text(item)
        if text and text.lower() not in seen:
            seen.add(text.lower())
            out.append(text)
    return out[:MAX_ITEMS]


def pick(value, allowed):
    """"Full-time" -> "full_time" if allowed, else None."""
    v = clean_text(value)
    v = v.lower().replace("-", "_").replace(" ", "_") if v else None
    return v if v in allowed.__args__ else None


def to_number(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        n = float(value)
    else:
        try:
            n = float(str(value).replace(",", "").replace("+", "").strip())
        except (TypeError, ValueError):
            return None
    return n if n >= 0 else None


def clean_range(value) -> YearsRange | None:
    """{"min": "5+", "max": 8} -> YearsRange(5, 8); min > max -> swapped; nothing -> None."""
    if not isinstance(value, dict):
        return None
    lo, hi = to_number(value.get("min")), to_number(value.get("max"))
    if lo is None and hi is None or not lo and not hi:  # nothing, or only zeros (a copied placeholder)
        return None
    if lo is not None and hi is not None and lo > hi:
        lo, hi = hi, lo
    return YearsRange(min=lo, max=hi)


class UploadJDResult(BaseModel):
    jd_id: int
    status: Literal["created", "already_exists"]
    filename: str
    jd: JDStructure


class JDListItem(BaseModel):
    jd_id: int
    filename: str
    title: str | None
    company: str | None
    created_at: datetime


class JDOut(BaseModel):
    jd_id: int
    filename: str
    model_id: str
    created_at: datetime
    jd: JDStructure


class DeleteJDResult(BaseModel):
    jd_id: int
    status: Literal["deleted"]
