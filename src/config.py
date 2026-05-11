"""Settings via pydantic-settings — reads from .env."""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    # Mistral
    mistral_api_key: str
    mistral_embed_model: str = "mistral-embed"
    mistral_gen_model_large: str = "mistral-large-latest"
    mistral_gen_model_small: str = "mistral-small-latest"

    # Postgres
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "familyhq_rag"
    postgres_user: str = "rag"
    postgres_password: str = "change_me_locally"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Eval
    eval_set_path: Path = Field(default=Path("eval_results/eval_set.jsonl"))
    eval_runs_dir: Path = Field(default=Path("eval_results/runs"))

    # Source data paths (gitignored — local only)
    trendradar_db_dir: Path = Field(default=Path("~/TrendRadar/output/news").expanduser())
    familyhq_products_dir: Path = Field(default=Path("~/Documents/FamilyHQ/PDFs_Live").expanduser())
    familyhq_product_copy: Path = Field(default=Path("~/Documents/FamilyHQ/Scripts/upload_3_new.py").expanduser())

    @field_validator(
        "trendradar_db_dir",
        "familyhq_products_dir",
        "familyhq_product_copy",
        "eval_set_path",
        "eval_runs_dir",
        mode="after",
    )
    @classmethod
    def expand_user_in_paths(cls, v: Path) -> Path:
        """Expand ~ in path values regardless of source (default vs .env)."""
        return v.expanduser()

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
