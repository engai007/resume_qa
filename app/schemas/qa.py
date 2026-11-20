from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    document_hashes: list[str] = []  # empty = all of your resumes
    top_k: int = Field(default=5, ge=1, le=20)  # chunks given to the LLM (search route)
    planner: bool = True  # false = plain search, the behaviour before the planner (for comparing)


class Source(BaseModel):
    n: int  # the [n] marker in the answer
    document_id: int
    document_hash: str
    candidate_name: str | None
    chunk_index: int
    section: str | None  # profile, skills, project, ...
    text: str


class PlanOut(BaseModel):
    route: str  # "search" or "section"
    sections: list[str]
    fallback: bool  # true = the planner failed, plain search was used


class AskResponse(BaseModel):
    question: str
    found: bool  # false = the resumes don't answer the question
    answer: str  # with [n] markers pointing at sources
    sources: list[Source]
    plan: PlanOut | None  # how the chunks were fetched; null when planner=false
    truncated: bool  # true = more chunks matched than the LLM gets: pick fewer resumes (document_hashes)
