from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    document_hashes: list[str] = []  # empty = search all of your resumes
    top_k: int = Field(default=5, ge=1, le=50)


class SearchResult(BaseModel):
    score: float  # higher = closer match
    text: str
    document_id: int
    document_hash: str
    candidate_name: str | None
    chunk_index: int


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
