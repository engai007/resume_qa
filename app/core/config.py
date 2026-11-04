from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]  # project root


class Settings(BaseSettings):
    app_name: str = "AI Agent System"
    debug: bool = False

    # Hugging Face
    hf_token: str = ""

    # Default LLM for agents that don't override it (Hugging Face inference provider)
    default_model_id: str = "Qwen/Qwen2.5-72B-Instruct"
    default_provider: str = "deepinfra"
    default_temperature: float = 0.01
    default_max_tokens: int = 1024
    default_timeout: int = 120

    # Every LLM call (agent, model, prompt version, tokens, seconds, errors) is appended here
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
