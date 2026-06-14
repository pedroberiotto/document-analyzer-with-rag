import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    llm_provider: str = "openai"
    embedding_provider: str = "openai"

    openai_api_key: str | None = Field(default=None)
    llm_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"

    ollama_model: str = "llama3.2:3b"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"

    llm_temperature: float = 0.0
    llm_seed: int = 42

    llm_timeout: float = 60.0
    llm_max_retries: int = 3

    chunk_size: int = 1000
    chunk_overlap: int = 200
    extract_tables: bool = True
    reindex_if_exists: bool = False

    default_strategy: str = "dense"
    default_reranker: str = "none"
    retrieval_k: int = 8
    rrf_k: int = 60
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    agentic_confidence_threshold: float = 0.6
    agentic_max_iterations: int = 3

    log_level: str = "INFO"
    log_format: str = "console"

    data_dir: Path = PROJECT_ROOT / "data"
    pricing_file: Path = PROJECT_ROOT / "config" / "pricing.json"

    @property
    def active_llm_model(self) -> str:
        return {
            "openai": self.llm_model,
            "ollama": self.ollama_model,
        }.get(self.llm_provider, self.llm_model)

    @property
    def active_embedding_model(self) -> str:
        return {
            "openai": self.embedding_model,
            "ollama": self.ollama_embedding_model,
        }.get(self.embedding_provider, self.embedding_model)

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def telemetry_db(self) -> Path:
        return self.data_dir / "telemetry.db"

    def ensure_dirs(self) -> None:
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

    def export_openai_key(self) -> None:
        if self.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = self.openai_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    settings.export_openai_key()
    return settings
