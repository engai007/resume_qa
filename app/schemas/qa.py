from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    document_hashes: list[str] = []  # empty = all of your resumes
    top_k: int = Field(default=5, ge=1, le=20)  # chunks given to the LLM


class Source(BaseModel):
    n: int  # the [n] marker in the answer
    document_id: int
    document_hash: str
    candidate_name: str | None
    chunk_index: int
    text: str


class AskResponse(BaseModel):
    question: str
    found: bool  # false = the resumes don't answer the question
    answer: str  # with [n] markers pointing at sources
    sources: list[Source]
