from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path
import os
import sys


class Settings(BaseSettings):
    # Paths
    DATA_DIR: Path = Field(default=Path(os.getenv("DATA_DIR", "./data")))
    MAX_UPLOAD_MB: int = 25

    # OpenAI
    OPENAI_API_KEY: str = Field(default="")
    CHAT_MODEL: str = Field(default="gpt-4o-mini")
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-small")

    # Qdrant
    QDRANT_URL: str = Field(default="http://localhost:6333")

    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379/0")

    # MinIO
    MINIO_ENDPOINT: str = Field(default="http://localhost:9000")
    MINIO_ACCESS_KEY: str = Field(default="minioadmin")
    MINIO_SECRET_KEY: str = Field(default="minioadmin")

    # OpenTelemetry (Jaeger)
    OTEL_EXPORTER_OTLP_ENDPOINT: str = Field(default="")

    # Python Debug
    PYTHONASYNCIODEBUG: str = Field(default="0")

    # Retrieval Settings
    TOPK_VEC: int = Field(default=20)
    TOPK_BM25: int = Field(default=50)
    FUSION_TOPK: int = Field(default=6)

    # Reranking Settings
    RERANK_METHOD: str = Field(default="mmr")
    RERANK_K: int = Field(default=6)
    RERANK_LAMBDA: float = Field(default=0.7)

    # Use modern Pydantic V2 ConfigDict instead of class Config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  # Allow case-insensitive env var names
        extra="ignore",  # Ignore extra fields in .env that aren't defined
    )


settings = Settings()

# Only create directory if not running tests
if "pytest" not in sys.modules:
    try:
        settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        # In Docker, directory is created by Dockerfile
        pass
