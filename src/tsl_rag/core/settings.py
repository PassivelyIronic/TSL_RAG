from functools import lru_cache
from typing import Literal

from pydantic import PostgresDsn, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # App
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Provider switch — "ollama" (local, free) lub "openai" (cloud)
    llm_provider: Literal["ollama", "openai"] = "ollama"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "mistral:7b-instruct-q4_K_M"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_embed_dimensions: int = 768

    # OpenAI (opcjonalne)
    openai_api_key: SecretStr | None = None
    openai_chat_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-large"
    openai_embedding_dimensions: int = 3072

    # Gemini (opcjonalne, pod ewaluację)
    gemini_api_key: SecretStr | None = None

    @property
    def embedding_dimensions(self) -> int:
        if self.llm_provider == "openai":
            return self.openai_embedding_dimensions
        return self.ollama_embed_dimensions

    @property
    def active_llm_model(self) -> str:
        if self.llm_provider == "openai":
            return self.openai_chat_model
        return self.ollama_llm_model

    # LLM params
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024

    postgres_dsn: PostgresDsn

    # Retrieval
    retrieval_top_k: int = 20
    retrieval_rerank_top_n: int = 7
    bm25_weight: float = 0.5
    dense_weight: float = 0.5

    # Reranker
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Generation
    max_context_tokens: int = 4096
    max_answer_tokens: int = 1024

    # Ingestion
    chunk_size: int = 400
    chunk_overlap: int = 50
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"

    @model_validator(mode="after")
    def validate_weights(self) -> "Settings":
        total = round(self.bm25_weight + self.dense_weight, 6)
        if total != 1.0:
            raise ValueError(f"bm25_weight + dense_weight musi = 1.0, got {total}")
        return self

    @model_validator(mode="after")
    def validate_openai_key(self) -> "Settings":
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY wymagany gdy LLM_PROVIDER=openai")
        return self

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
