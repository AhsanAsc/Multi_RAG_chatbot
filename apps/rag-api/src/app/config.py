from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    DATA_DIR: Path = Field(default=Path("/app/data"))
    MAX_UPLOAD_MB: int = 25

    class config:
        env_file = ".env"


settings = Settings()
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
