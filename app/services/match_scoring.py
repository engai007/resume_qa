"""Score one resume against one JD, in code (no LLM). See docs/jd_matching_plan.md section 3.

score = 0.50 required + 0.15 preferred + 0.15 experience + 0.20 semantic
A part the JD doesn't give (no preferred skills, no years) is left out and the other weights are rescaled,
so a JD that lists only requirements isn't penalised.
"""
from datetime import date

from app.services.resume_chunker import job_span, make_chunks, total_months
from app.services.skill_match import match_skills

WEIGHTS = {"required": 0.50, "preferred": 0.15, "experience": 0.15, "semantic": 0.20}  # first guess, tune with eval
# Semantic part = how close the candidate's actual WORK is to the JD's. Skills lists are left out: skill matching
# already scores them (65%), and a listed "LangChain" isn't LLM work. Measured on real resumes (2026-11-24):
# a total mismatch (ICU nurse JD) still has cosine ~0.40-0.43, good fits 0.58-0.76, so 0.40-0.80 is stretched to 0-1.
SEMANTIC_SECTIONS = ["project", "experience", "personal_project"]
SEMANTIC_FLOOR, SEMANTIC_CEILING = 0.40, 0.80
MAX_FOCUS_CHARS = 1500  # ~400 tokens: bge-base reads at most 512


def jd_focus_text(jd: dict) -> str:
    """What the resume's work is compared with: title, summary, responsibilities, required skills.
    Not the whole JD: company blurb and benefits would only add noise, and bge reads 512 tokens at most."""
    parts = [jd.get("title"), jd.get("summary"), "; ".join(jd.get("responsibilities") or []),
             ", ".join(jd.get("required_skills") or [])]
    return ". ".join(p.strip().rstrip(".") for p in parts if p and p.strip())[:MAX_FOCUS_CHARS]


def semantic_part(similarity: float | None, chunk_index: int | None = None) -> dict | None:
    """Raw cosine of the best work chunk -> 0-1 score. None (left out) when the resume has no work chunks."""
    if similarity is None:
        return None
    score = (similarity - SEMANTIC_FLOOR) / (SEMANTIC_CEILING - SEMANTIC_FLOOR)
    return {"score": round(min(1.0, max(0.0, score)), 4), "similarity": round(similarity, 4),
            "best_chunk_index": chunk_index}


def candidate_months(resume: dict, today: date | None = None) -> int | None:
    """Total work months, same rules as the experience summary chunk (overlaps once, gaps not counted).
    None if no job has readable dates."""
    today = today or date.today()
    spans = [s for s in (job_span(e, today) for e in resume.get("experience") or []) if s]
    return total_months(spans) if spans else None


def experience_part(resume: dict, years: dict | None, today: date | None = None) -> dict | None:
    """Candidate years vs the JD's minimum: >= min -> 1.0, else candidate / min. Over the max isn't penalised.
    None (left out) when the JD gives no minimum. Unknown candidate years -> 0 (can't show it)."""
    required_min = (years or {}).get("min")
    if not required_min:
        return None
    months = candidate_months(resume, today)
    cand_years = round(months / 12, 1) if months is not None else None
    score = 0.0 if cand_years is None else min(1.0, cand_years / required_min)
    return {"score": round(score, 4), "candidate_years": cand_years, "required_min": required_min,
            "required_max": (years or {}).get("max")}


def combine(parts: dict[str, float | None]) -> float:
    """Weighted mean of the parts that exist. {"required": 1.0, "preferred": None, ...} -> weights rescaled."""
    used = {k: v for k, v in parts.items() if v is not None}
    if not used:
        return 0.0
    total = sum(WEIGHTS[k] for k in used)
    return round(sum(WEIGHTS[k] * v for k, v in used.items()) / total, 4)


NO_WORK = {"score": 0.0, "similarity": None, "best_chunk_index": None}  # indexed, but no work chunk at all


def score_resume(resume: dict, jd: dict, similarity: float | None = None, best_chunk_index: int | None = None,
                 today: date | None = None, semantic: bool = False) -> dict:
    """resume = parsed_doc_json, jd = JDStructure dict, similarity = cosine of the resume's best work chunk
    (best_chunks_per_resume). semantic=False: the part is left out (unit use). semantic=True and no similarity:
    the resume has no work chunks, so the part is 0, not left out (else skills alone would score it).
    -> {"score", "breakdown": {"required", "preferred", "experience", "semantic"}}"""
    chunks = make_chunks(resume, today)
    breakdown = {
        "required": match_skills(jd.get("required_skills") or [], chunks),
        "preferred": match_skills(jd.get("preferred_skills") or [], chunks),
        "experience": experience_part(resume, jd.get("experience_years"), today),
        "semantic": semantic_part(similarity, best_chunk_index) if similarity is not None
                    else dict(NO_WORK) if semantic else None,
    }
    score = combine({k: (v or {}).get("score") for k, v in breakdown.items()})
    return {"score": score, "breakdown": breakdown}
