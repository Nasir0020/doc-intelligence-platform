"""
config.py
=========
Centralized application configuration.

Why this file exists:
    Every piece of environment-specific configuration (database URLs, API
    keys, feature flags) should live in exactly ONE place, read from
    environment variables. This file is that single source of truth.

Why not just use os.environ.get(...) scattered across the codebase?
    - No validation: a typo'd env var name fails silently (returns None)
      instead of raising an error at startup.
    - No type safety: os.environ always returns strings, so "PORT" would
      be the string "8000", not the integer 8000, unless you remember to
      cast it everywhere you use it.
    - No single reference point: if you need to know every configurable
      value in the system, you'd have to grep the entire codebase.

Which other files use this:
    Nearly every module (ingestion, retrieval, db) imports `settings` from
    this file to get things like database URLs, model names, and API keys.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Defines every configurable value the application needs, with types
    and (where sensible) defaults.

    Inheriting from `BaseSettings` (instead of plain Pydantic `BaseModel`)
    is what gives this class its special power: on instantiation, it
    automatically reads matching values from environment variables (and,
    via `model_config` below, from a `.env` file) and validates them
    against the type hints declared here.
    """

    # --- Application metadata ---
    APP_NAME: str = "Document Intelligence Platform"
    APP_ENV: str = "development"  # "development" | "staging" | "production"
    DEBUG: bool = True

    # --- API server ---
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # --- Database (Postgres) — populated when we build Module 4/8 ---
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "docintel"
    POSTGRES_USER: str = "docintel_user"
    POSTGRES_PASSWORD: str = "changeme"

    # --- Vector DB (Weaviate) — populated when we build Module 4 ---
    WEAVIATE_URL: str = "http://localhost:8080"

    # --- LLM provider — populated when we build Module 7 ---
    ANTHROPIC_API_KEY: str = ""

    @property
    def postgres_url(self) -> str:
        """
        Builds the full SQLAlchemy-style connection string from the
        individual pieces above.

        Why a @property instead of just storing the full URL directly?
        Because storing the pieces separately lets each piece be
        independently overridden by environment variables (useful in
        Docker Compose, where the hostname changes from "localhost" to
        the service name "postgres"), while still giving every other
        part of the app one convenient, ready-to-use connection string.
        """
        return (
            f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # `model_config` tells pydantic-settings HOW to find these values:
    # 1. First check actual environment variables.
    # 2. Fall back to reading a `.env` file at the project root if present.
    # 3. Env var names are matched case-insensitively to the fields above.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",  # ignore unrelated env vars instead of erroring
    )


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached singleton instance of Settings.

    Why @lru_cache (a decorator that memoizes function results):
        Without it, every call to get_settings() would re-read environment
        variables and re-construct a new Settings object. That's wasteful,
        and more importantly, it means different parts of the app could
        theoretically see different config if env vars changed mid-run.
        @lru_cache ensures Settings is built exactly once and the same
        instance is reused everywhere — this is the standard FastAPI
        pattern for configuration (and is also how FastAPI's own
        dependency-injection docs recommend handling settings).

    Why a function instead of a plain module-level `settings = Settings()`:
        Using a function makes this easy to override in tests — a test can
        call `app.dependency_overrides[get_settings] = lambda: test_settings`
        to inject different config without touching real environment
        variables. We'll use this in Module 12 (testing).
    """
    return Settings()


settings = get_settings()
