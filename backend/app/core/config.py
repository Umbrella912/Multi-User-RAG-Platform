"""Application configuration loaded from environment variables.

All settings are typed and read from the environment (and an optional ``.env``
file). No real secrets are hardcoded; non-secret, local-dev defaults are provided
and every secret **must** be overridden via the environment in real deployments.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings sourced from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    PROJECT_NAME: str = "Multi-User RAG Platform"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api"
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: list[str] = ["http://localhost:8501"]

    # --- Database ---
    DATABASE_URL: str = "postgresql+asyncpg://rag_user:rag_password@localhost:5432/rag_platform"
    DB_ECHO: bool = False

    # --- Auth / JWT ---
    JWT_SECRET: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # --- LLM provider (switch via LLM_PROVIDER; never commit real keys) ---
    LLM_PROVIDER: Literal["groq", "openai", "gemini", "ollama"] = "groq"
    GROQ_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 1024

    # --- LLM retry / timeout (exponential backoff for timeouts & rate limits) ---
    LLM_TIMEOUT_SECONDS: float = 60.0
    LLM_MAX_RETRIES: int = 3
    LLM_RETRY_BASE_DELAY: float = 0.5
    LLM_RETRY_MAX_DELAY: float = 8.0

    # --- Embeddings ---
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

    # --- Retrieval / RAG ---
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 160
    TOP_K_VECTOR: int = 20
    TOP_K_BM25: int = 20
    RRF_K: int = 60
    RERANK_ENABLED: bool = True
    RERANK_TOP_N: int = 6
    SIMILARITY_THRESHOLD: float = 0.35
    CONTEXT_CHUNKS: int = 6
    HISTORY_TURNS: int = 6

    # --- Upload limits ---
    MAX_UPLOAD_SIZE_MB: int = 25
    MAX_KB_SIZE_MB: int = 200
    ALLOWED_FILE_TYPES: list[str] = ["pdf", "docx", "txt", "md"]

    # --- Storage ---
    STORAGE_BACKEND: Literal["local", "s3"] = "local"
    STORAGE_LOCAL_PATH: str = "./data/uploads"
    # S3 (or S3-compatible) object storage for uploaded files.
    S3_BUCKET: str = ""
    S3_REGION: str = "us-east-1"
    S3_ENDPOINT_URL: str = ""  # leave empty for AWS; set for MinIO/R2/etc.
    S3_ACCESS_KEY: str = ""
    S3_SECRET_KEY: str = ""
    S3_PREFIX: str = "uploads"  # key prefix inside the bucket

    # --- Chroma vector DB ---
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    CHROMA_PERSIST_DIR: str = "./data/chroma"

    # --- Rate limiting (per user) ---
    RATE_LIMIT_CHAT_PER_MIN: int = 20
    RATE_LIMIT_UPLOAD_PER_MIN: int = 10

    @field_validator("CORS_ORIGINS", "ALLOWED_FILE_TYPES", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Parse comma-separated strings from the environment into lists."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @property
    def database_url_sync(self) -> str:
        """Synchronous DB URL (psycopg) for tooling that cannot use asyncpg."""
        return self.DATABASE_URL.replace("+asyncpg", "+psycopg")


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    return Settings()


# Module-level singleton used throughout the application.
settings: Settings = get_settings()
