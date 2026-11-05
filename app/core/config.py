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
    default_temperature: float = 0.01
    default_max_tokens: int = 1024
    default_timeout: int = 120

    # Every LLM call (agent, model, tokens, seconds) is appended here
    llm_log_file: str = "logs/llm_calls.jsonl"

    # Resume parser
    parser_model_id: str = "Qwen/Qwen2.5-72B-Instruct"
    parser_provider: str = "deepinfra"
    parser_max_new_tokens: int = 800      # only structure + line numbers, not full text
    parser_temperature: float = 0.01      # near-deterministic (HF rejects exactly 0)
    parser_timeout: int = 600
    parser_gap_ratio: float = 0.15        # char gap (x font size) treated as a space
    parser_max_words: int = 8             # truncate long description lines sent to the LLM
    parser_concurrency: int = 4           # resumes parsed in parallel
    parser_input_dir: str = "resumes"     # script default: PDFs to parse
    parser_output_dir: str = "outputs"

    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")


settings = Settings()
