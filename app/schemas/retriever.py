from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.services.resume_chunker import SECTIONS


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    document_hashes: list[str] = []  # empty = search all of your resumes
    top_k: int = Field(default=5, ge=1, le=50)
    mode: Literal["hybrid", "dense", "sparse"] = "hybrid"  # hybrid = meaning + keywords, merged with RRF
    sections: list[str] = []  # empty = all sections; e.g. ["project", "personal_project"]

    @field_validator("sections")
    @classmethod
    def known_sections(cls, v: list[str]) -> list[str]:
        unknown = [s for s in v if s not in SECTIONS]
        if unknown:
            raise ValueError(f"unknown sections {unknown}, use: {', '.join(SECTIONS)}")
        return v


class SearchResult(BaseModel):
    score: float | None  # None = fetched by section, not ranked. Higher = better; its scale depends on mode (hybrid = RRF score, not similarity)
    text: str
    document_id: int
    document_hash: str
    candidate_name: str | None
    chunk_index: int
    section: str | None  # profile, skills, project, ... (None before INDEX_VERSION 5)


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
