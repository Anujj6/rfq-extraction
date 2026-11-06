"""Application configuration loaded from environment variables (and .env for local dev).

All required secrets use ``SecretStr`` so they are never revealed in logs,
``repr()``, or validation error messages.  The app will raise a clear
``ValueError`` at startup if any required variable is missing or empty.
"""

from __future__ import annotations

from typing import Optional

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


_REQUIRED_SECRET_FIELDS = ("GEMINI_API_KEY", "SERVICE_API_KEY")
_REQUIRED_PLAIN_FIELDS = ("GEMINI_MODEL",)


class Settings(BaseSettings):
    """Runtime configuration for the RFQ extraction microservice."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Extra env vars are silently ignored — forward-compatible.
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Required secrets (never printed, never logged)
    # ------------------------------------------------------------------
    GEMINI_API_KEY: SecretStr
    SERVICE_API_KEY: SecretStr

    # ------------------------------------------------------------------
    # Required plain strings
    # ------------------------------------------------------------------
    GEMINI_MODEL: str

    # ------------------------------------------------------------------
    # Optional with defaults
    # ------------------------------------------------------------------
    MAX_FILE_MB: int = 15
    STORAGE_PROVIDER: str = "local"
    STORAGE_LOCAL_DIR: str = "./samples"
    ENABLE_DOCS: bool = False
    MEDIA_RESOLUTION: str = "default"

    # ------------------------------------------------------------------
    # S3 / Object storage configuration
    # ------------------------------------------------------------------
    STORAGE_BUCKET: str = ""
    S3_ENDPOINT_URL: str = ""
    S3_REGION: str = ""
    S3_ACCESS_KEY_ID: str = ""
    S3_SECRET_ACCESS_KEY: Optional[SecretStr] = None

    # ------------------------------------------------------------------
    # Validators: reject empty strings for required fields
    # ------------------------------------------------------------------

    @field_validator("GEMINI_MODEL", mode="before")
    @classmethod
    def _gemini_model_non_empty(cls, v: str) -> str:
        if not v or not str(v).strip():
            raise ValueError(
                "GEMINI_MODEL must be set to a non-empty value. "
                "Check your environment or .env file."
            )
        return v

    @field_validator("MEDIA_RESOLUTION", mode="before")
    @classmethod
    def _media_resolution_valid(cls, v: str | None) -> str:
        if v is None:
            return "default"
        val = str(v).strip().lower()
        allowed = {"default", "low", "medium", "high"}
        if val not in allowed:
            raise ValueError(
                f"MEDIA_RESOLUTION must be one of {sorted(allowed)}, got {v!r}. "
                "Check your environment or .env file."
            )
        return val

    @field_validator("STORAGE_PROVIDER", mode="before")
    @classmethod
    def _storage_provider_valid(cls, v: str | None) -> str:
        if v is None:
            return "local"
        val = str(v).strip().lower()
        if val not in {"local", "s3"}:
            raise ValueError(f"STORAGE_PROVIDER must be 'local' or 's3', got {v!r}")
        return val

    @model_validator(mode="after")
    def _secrets_non_empty(self) -> "Settings":
        """Ensure SecretStr fields actually contain a non-empty secret."""
        for field_name in _REQUIRED_SECRET_FIELDS:
            secret: SecretStr = getattr(self, field_name)
            if not secret.get_secret_value().strip():
                raise ValueError(
                    f"{field_name} must be set to a non-empty value. "
                    "Check your environment or .env file."
                )

        if self.STORAGE_PROVIDER == "s3":
            missing_s3: list[str] = []
            if not self.STORAGE_BUCKET or not self.STORAGE_BUCKET.strip():
                missing_s3.append("STORAGE_BUCKET")
            if not self.S3_ACCESS_KEY_ID or not self.S3_ACCESS_KEY_ID.strip():
                missing_s3.append("S3_ACCESS_KEY_ID")
            if not self.S3_SECRET_ACCESS_KEY or not self.S3_SECRET_ACCESS_KEY.get_secret_value().strip():
                missing_s3.append("S3_SECRET_ACCESS_KEY")
            if missing_s3:
                raise ValueError(
                    f"STORAGE_PROVIDER='s3' requires the following variables to be set: {', '.join(missing_s3)}. "
                    "Check your environment or .env file."
                )

        return self


def get_settings() -> Settings:
    """Return a cached ``Settings`` instance, raising on first bad config."""
    return _settings


# Instantiate once at import time so startup errors surface immediately.
_settings = Settings()
