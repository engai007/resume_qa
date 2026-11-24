from typing import Literal

from pydantic import BaseModel, Field


class MatchRequest(BaseModel):
    document_hashes: list[str] = []  # empty = every resume you can access
    limit: int = Field(default=50, ge=1, le=200)  # results returned (all in-scope resumes are scored)
    explain: bool = True  # LLM verdict for the top_n; false = ranking only, no LLM call
    top_n: int = Field(default=5, ge=0, le=10)  # how many top candidates get a verdict


class SkillEvidence(BaseModel):
    skill: str  # as written in the JD
    found_as: str  # the name that matched, e.g. "Postgres" for "PostgreSQL"
    section: str
    chunk_index: int


class SkillPart(BaseModel):
    score: float  # share of the JD's skills found
    matched: list[SkillEvidence]
    missing: list[str]


class ExperiencePart(BaseModel):
    score: float
    candidate_years: float | None
    required_min: float
    required_max: float | None


class SemanticPart(BaseModel):
    score: float  # best work chunk's cosine, 0.40-0.80 stretched to 0-1
    similarity: float | None  # raw cosine; None = no work chunk
    best_chunk_index: int | None


class Breakdown(BaseModel):
    required: SkillPart | None  # None = the JD has none, part left out of the score
    preferred: SkillPart | None
    experience: ExperiencePart | None
    semantic: SemanticPart | None


class VerdictSource(BaseModel):
    n: int  # the [n] in the strengths
    chunk_index: int
    section: str


class Verdict(BaseModel):
    fit: Literal["strong", "possible", "weak"] | None
    strengths: list[str]  # each cites a chunk [n]
    gaps: list[str]
    summary: str | None
    sources: list[VerdictSource]


class MatchResult(BaseModel):
    rank: int
    document_id: int
    document_hash: str
    candidate_name: str | None
    score: float  # 0.50 required + 0.15 preferred + 0.15 experience + 0.20 semantic (missing parts rescaled)
    breakdown: Breakdown
    verdict: Verdict | None = None  # only for the top_n with explain=true; null if that LLM call failed


class MatchResponse(BaseModel):
    jd_id: int
    jd_title: str | None
    resumes_compared: int
    skipped_not_indexed: int  # resumes not (yet) in the vector DB: index them to include them
    results: list[MatchResult]
