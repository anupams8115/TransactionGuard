"""Application settings loaded from environment variables (and an optional .env file).

Why pydantic-settings?  Every setting is typed and validated once at start-up, so a typo
in an environment variable fails loudly instead of producing a confusing runtime error.
All variables use the ``TG_`` prefix, e.g. ``TG_DATABASE_URL``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TG_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    log_json: bool = False  # human-readable logs locally, JSON lines in containers

    # --- persistence ---
    # SQLite keeps the default setup runnable without Docker; Docker Compose uses PostgreSQL.
    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'transactionguard.db').as_posix()}"

    # --- async jobs ---
    redis_url: str = "redis://localhost:6379/0"
    # When true, Celery runs tasks inline in the API process (no Redis/worker needed).
    celery_eager: bool = True
    celery_task_time_limit_s: int = 600
    celery_max_retries: int = 3

    # --- auth ---
    jwt_secret: SecretStr = Field(default=SecretStr("change-me-in-dotenv"))
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 60
    cookie_secure: bool = False  # set true behind HTTPS

    # --- ML artifacts & tracking ---
    data_dir: Path = PROJECT_ROOT / "data"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    reports_dir: Path = PROJECT_ROOT / "reports"
    mlflow_tracking_uri: str = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    mlflow_experiment: str = "transactionguard"
    random_seed: int = 42

    # --- API behaviour ---
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:3000"]
    max_batch_rows: int = 50_000
    max_upload_mb: int = 20
    # Explanations for single predictions are computed inline when fast enough;
    # batch explanations are always generated asynchronously.
    explain_inline: bool = True

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        # Allow TG_CORS_ORIGINS=http://a,http://b as well as a JSON list.
        if isinstance(v, str) and not v.strip().startswith("["):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @property
    def bundles_dir(self) -> Path:
        return self.artifacts_dir / "bundles"

    def assert_safe_for_production(self) -> None:
        """Refuse to start a production deployment with the placeholder secret."""
        if self.environment == "production" and self.jwt_secret.get_secret_value() in {
            "change-me-in-dotenv",
            "",
        }:
            raise RuntimeError("TG_JWT_SECRET must be set to a strong secret in production")


@lru_cache
def get_settings() -> Settings:
    return Settings()
