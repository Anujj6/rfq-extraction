"""Gemini API client: wraps google-genai for structured RFQ extraction calls.

SDK version pinned: google-genai==2.32.0

Official docs consulted:
  - https://ai.google.dev/gemini-api/docs/gemini-3  (Gemini 3 guide: thinking_level, temperature 1.0 default)
  - https://ai.google.dev/gemini-api/docs/thinking  (ThinkingConfig, thinking_level, dynamic thinking)
  - https://ai.google.dev/gemini-api/docs/media-resolution  (Media resolution: global / per part, token counts)
  - https://ai.google.dev/gemini-api/docs/models/gemini  (Official models list confirming gemini-3.8-flash)

Configuration notes:
  - Thinking level:
      Gemini 3 models use dynamic thinking. We use types.ThinkingConfig with
      thinking_level=types.ThinkingLevel.LOW ("low"). Legacy thinking_budget is
      NOT used (docs state mixing thinking_level and thinking_budget causes a 400).
      When thinking_level is omitted, Gemini 3 models default to "high".
  - Temperature:
      Omitted (left unset / None). Docs strongly recommend the default value of
      1.0 for all Gemini 3 models and advise against setting lower values.
  - Media resolution:
      Configured via MEDIA_RESOLUTION env var. The default is "default", which
      leaves media_resolution unset (None) to match AI Studio's "Default" setting.
      "low", "medium", and "high" map to the respective types.MediaResolution enums.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

from app.config import get_settings
from app.prompts import PROMPT_VERSION, SYSTEM_INSTRUCTION
from app.schemas import load_rfq_schema

# ---------------------------------------------------------------------------
# Logging — never log file contents, model output, or API key
# ---------------------------------------------------------------------------

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REQUEST_TIMEOUT_MS = 120_000  # 120 s in milliseconds (HttpOptions.timeout unit)
_MAX_ATTEMPTS = 3
_BACKOFF_BASE = 2.0  # seconds


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class ExtractionError(Exception):
    """Raised when the Gemini extraction pipeline fails.

    The message is safe to surface to callers — it never contains raw SDK
    error text, model output, or request data.
    """

    def __init__(
        self,
        message: str,
        *,
        cause_class: str | None = None,
        http_code: int | None = None,
        status: str | None = None,
    ) -> None:
        super().__init__(message)
        self.cause_class = cause_class
        self.http_code = http_code
        self.status = status


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_media_resolution(resolution_setting: str) -> types.MediaResolution | None:
    """Map the MEDIA_RESOLUTION setting to an SDK MediaResolution enum or None."""
    mapping = {
        "low": types.MediaResolution.MEDIA_RESOLUTION_LOW,
        "medium": types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
        "high": types.MediaResolution.MEDIA_RESOLUTION_HIGH,
        "default": None,
    }
    return mapping.get(resolution_setting.strip().lower(), None)


def _build_client() -> genai.Client:
    """Construct a genai.Client using the configured API key and timeout."""
    settings = get_settings()
    return genai.Client(
        api_key=settings.GEMINI_API_KEY.get_secret_value(),
        http_options=types.HttpOptions(timeout=_REQUEST_TIMEOUT_MS),
    )


def _build_parts(file_parts: list[tuple[bytes, str]]) -> list[types.Part]:
    """Convert (bytes, mime_type) tuples into SDK Part objects.

    text/plain  → types.Part.from_text (the SDK treats inline text more
                  efficiently and avoids base64 overhead for plain text).
    everything else → types.Part.from_bytes with the supplied MIME type.
    """
    parts: list[types.Part] = []
    for data, mime_type in file_parts:
        if mime_type == "text/plain":
            parts.append(types.Part.from_text(text=data.decode("utf-8", errors="replace")))
        else:
            parts.append(types.Part.from_bytes(data=data, mime_type=mime_type))
    return parts


def _is_retryable(exc: Exception) -> bool:
    """Return True only for 429 (rate-limit) and 5xx (server) errors."""
    if isinstance(exc, ServerError):
        return True
    if isinstance(exc, ClientError) and exc.code == 429:
        return True
    return False


def _safe_usage_log(usage: Any) -> dict:
    """Extract only token-count fields from usage_metadata for logging."""
    if usage is None:
        return {}
    return {
        "prompt_tokens": getattr(usage, "prompt_token_count", None),
        "candidates_tokens": getattr(usage, "candidates_token_count", None),
        "thoughts_tokens": getattr(usage, "thoughts_token_count", None),
        "total_tokens": getattr(usage, "total_token_count", None),
    }


def _extract_error_details(exc: Exception) -> tuple[str, int | None, str | None]:
    """Extract (class_name, http_code, error_status) safely without error messages.

    Guarantees:
    - Never includes exc message text, prompt, or file data.
    - http_code is an integer if present, else None.
    - error_status is a valid Google status identifier (e.g. RESOURCE_EXHAUSTED), else None.
    """
    exc_class = type(exc).__name__

    raw_code = getattr(exc, "code", getattr(exc, "status_code", None))
    http_code: int | None = None
    if isinstance(raw_code, int):
        http_code = raw_code
    elif raw_code is not None:
        try:
            http_code = int(raw_code)
        except (ValueError, TypeError):
            http_code = None

    error_status: str | None = None
    raw_status = getattr(exc, "status", None)
    if isinstance(raw_status, str):
        cleaned = raw_status.strip()
        if cleaned and len(cleaned) <= 64 and re.match(r"^[A-Z0-9_]+$", cleaned):
            error_status = cleaned
    elif isinstance(getattr(exc, "details", None), dict):
        details = exc.details
        status_val = details.get("status") or details.get("error", {}).get("status")
        if isinstance(status_val, str):
            cleaned = status_val.strip()
            if cleaned and len(cleaned) <= 64 and re.match(r"^[A-Z0-9_]+$", cleaned):
                error_status = cleaned

    return exc_class, http_code, error_status


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_rfq(
    file_parts: list[tuple[bytes, str]],
    request_id: str,
) -> dict:
    """Call the Gemini API to extract structured RFQ data from one or more files.

    Parameters
    ----------
    file_parts:
        A list of (file_bytes, mime_type) tuples.  Multiple files in one
        request are supported.  ``text/plain`` is sent as a text Part; PDFs
        and images are sent as inline binary Parts.
    request_id:
        Opaque identifier logged with every request for traceability.

    Returns
    -------
    dict
        The parsed JSON object returned by Gemini, validated to match the
        structure in docs/rfq_schema.json.  This is the raw dict — callers
        should pass it to ``RfqExtracted.model_validate`` for full Pydantic
        validation.

    Raises
    ------
    ExtractionError
        On any failure after all retry attempts.  The message is safe to
        surface externally.
    """
    settings = get_settings()
    model_name: str = settings.GEMINI_MODEL

    schema = load_rfq_schema()
    parts = _build_parts(file_parts)
    media_res = _resolve_media_resolution(settings.MEDIA_RESOLUTION)

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        # Structured output: application/json + our schema
        response_mime_type="application/json",
        response_schema=schema,
        # Thinking: Low level using thinking_level (no thinking_budget)
        thinking_config=types.ThinkingConfig(
            thinking_level=types.ThinkingLevel.LOW,
        ),
        # Media resolution: None when "default"; otherwise MediaResolution enum
        media_resolution=media_res,
        # Temperature: omitted (left as None, allowing model default 1.0)
        # Grounding + code execution: off (no tools supplied)
        tools=None,
    )

    client = _build_client()
    last_exc: Exception | None = None

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        t0 = time.monotonic()
        try:
            _log.info(
                "Gemini request start",
                extra={
                    "request_id": request_id,
                    "model": model_name,
                    "prompt_version": PROMPT_VERSION,
                    "attempt": attempt,
                    "num_parts": len(file_parts),
                },
            )

            response = client.models.generate_content(
                model=model_name,
                contents=[types.Content(role="user", parts=parts)],
                config=config,
            )

            latency_ms = round((time.monotonic() - t0) * 1000)
            _log.info(
                "Gemini request complete",
                extra={
                    "request_id": request_id,
                    "model": model_name,
                    "latency_ms": latency_ms,
                    "usage": _safe_usage_log(response.usage_metadata),
                },
            )

            # response.text is the raw JSON string from the model.
            # We never log it — it could contain confidential document data.
            raw_text: str | None = response.text
            if not raw_text:
                raise ExtractionError(
                    "Gemini returned an empty response. "
                    "The model may have blocked the content or timed out."
                )

            # Parse JSON — never pass to eval/exec/shell/URL fetch.
            try:
                return json.loads(raw_text)
            except json.JSONDecodeError:
                raise ExtractionError(
                    "Gemini response was not valid JSON. "
                    "Check that the model and schema are compatible."
                )

        except ExtractionError:
            raise  # already safe — re-raise immediately, no retry

        except Exception as exc:  # noqa: BLE001
            latency_ms = round((time.monotonic() - t0) * 1000)
            last_exc = exc
            exc_class, http_code, error_status = _extract_error_details(exc)

            attempt_parts = [f"attempt {attempt}", exc_class]
            if http_code is not None:
                attempt_parts.append(f"status_code={http_code}")
            if error_status:
                attempt_parts.append(f"status={error_status}")
            attempt_str = ", ".join(attempt_parts)

            if not _is_retryable(exc):
                _log.error(
                    "Gemini request failed (non-retryable): %s",
                    attempt_str,
                    extra={
                        "request_id": request_id,
                        "attempt": attempt,
                        "latency_ms": latency_ms,
                        "error_type": exc_class,
                        "http_code": http_code,
                        "error_status": error_status,
                    },
                )
                raise ExtractionError(
                    f"Extraction failed with a non-retryable error "
                    f"({exc_class}). See server logs for details.",
                    cause_class=exc_class,
                    http_code=http_code,
                    status=error_status,
                ) from exc

            backoff = _BACKOFF_BASE ** (attempt - 1)
            _log.warning(
                "Gemini request failed (retryable), backing off: %s",
                attempt_str,
                extra={
                    "request_id": request_id,
                    "attempt": attempt,
                    "max_attempts": _MAX_ATTEMPTS,
                    "latency_ms": latency_ms,
                    "error_type": exc_class,
                    "http_code": http_code,
                    "error_status": error_status,
                    "backoff_s": backoff,
                },
            )
            if attempt < _MAX_ATTEMPTS:
                time.sleep(backoff)

    cause_class, http_code, error_status = (
        _extract_error_details(last_exc) if last_exc else (None, None, None)
    )
    raise ExtractionError(
        f"Extraction failed after {_MAX_ATTEMPTS} attempts. "
        "See server logs for details.",
        cause_class=cause_class,
        http_code=http_code,
        status=error_status,
    ) from last_exc
