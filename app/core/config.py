from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]  # project root


class Settings(BaseSettings):
    app_name: str = "AI Agent System"
    debug: bool = False

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/resumer_qa"

    # vector database
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""  # empty for local Qdrant without auth
    qdrant_collection: str = "resume_chunks"  # alias; real collections are resume_chunks_v1, _v2...

    # embeddings via HF Inference (changing the model needs a vector migration, see vector_migrations/)
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_provider: str = "hf-inference"
    embedding_size: int = 768
    embedding_timeout: int = 60

    # Auth (JWT)
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 60

    # Hugging Face
    hf_token: str = ""

    # Default LLM for agents that don't override it (Hugging Face inference provider)
    default_model_id: str = "Qwen/Qwen2.5-72B-Instruct"
    default_provider: str = "deepinfra"
    # same model on another provider, used on retry 1 ("" = none). None for Qwen 72B: since 2026-11-25 only
    # deepinfra serves it through the HF router (featherless-ai, together, novita, ... answer "not supported")
    default_fallback_provider: str = ""
    default_temperature: float = 0.01
    default_max_tokens: int = 1024
    default_timeout: int = 60          # seconds per try; QA answers take 2-8 s

    # Every LLM call (agent, model, tokens, seconds) is appended here
    llm_log_file: str = "logs/llm_calls.jsonl"

    # Query planner: small fast model, reads only the question (see docs/planner_plan.md)
    planner_model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    planner_provider: str = "groq"             # novita dropped Llama 8B on 2026-11-25; groq ~0.4 s
    planner_fallback_provider: str = "deepinfra"
    planner_max_tokens: int = 150      # the plan is ~30 tokens
    planner_timeout: int = 20          # seconds per try; normal ~1 s

    # Job description parser (see docs/jd_parser_plan.md)
    jd_parser_model_id: str = "Qwen/Qwen2.5-72B-Instruct"
    jd_parser_provider: str = "deepinfra"
    jd_parser_fallback_provider: str = ""
    jd_parser_max_tokens: int = 1024   # the structure is ~300-600 tokens
    jd_parser_timeout: int = 60        # seconds per try
    jd_max_file_bytes: int = 2_000_000

    # RAGAS judge, evals only (see docs/ragas_eval_plan.md). Another model family than the answer model (Qwen):
    # a model grading its own answers is too lenient
    judge_model_id: str = "openai/gpt-oss-120b"
    judge_provider: str = "cerebras"           # ~1.2 s per check; groq ~2 s
    judge_fallback_provider: str = "groq"      # used with: ragas_eval.py --provider groq
    judge_max_tokens: int = 4096       # gpt-oss reasons first; a small limit gives an empty reply
    judge_timeout: int = 120

    # Resume parser
    parser_model_id: str = "Qwen/Qwen2.5-72B-Instruct"
    parser_provider: str = "deepinfra"
    parser_fallback_provider: str = ""
    parser_max_new_tokens: int = 800      # only structure + line numbers, not full text
    parser_temperature: float = 0.01      # near-deterministic (HF rejects exactly 0)
    parser_timeout: int = 90              # seconds per try; parses take 14-29 s, a hung call is retried
    parser_gap_ratio: float = 0.15        # char gap (x font size) treated as a space
    parser_max_words: int = 8             # truncate long description lines sent to the LLM
    parser_concurrency: int = 4           # resumes parsed in parallel
    parser_input_dir: str = "resumes"     # script default: PDFs to parse
    parser_output_dir: str = "outputs"

    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")


settings = Settings()
