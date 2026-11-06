"""Tests for:
  1. app/services/gemini.py  — Gemini client behaviour (mocked; no real API calls)
  2. app/prompts.py          — SYSTEM_INSTRUCTION must match docs/system_instructions.md
  3. app/schemas.py          — Pydantic model field names and required lists must
                               match docs/rfq_schema.json
"""

from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from google.genai import types
from google.genai.errors import ClientError, ServerError

from app.services.gemini import (
    ExtractionError,
    _build_parts,
    _is_retryable,
    extract_rfq,
)
from app.schemas import (
    RfqExtracted,
    Client,
    RfqDetails,
    Delivery,
    LineItem,
    CommercialTerms,
    Attachment,
    SourceRef,
    load_rfq_schema,
)
from app.prompts import SYSTEM_INSTRUCTION, PROMPT_VERSION

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DOCS = Path(__file__).parent.parent.parent / "docs"
_SYSTEM_INSTRUCTIONS_PATH = _DOCS / "system_instructions.md"
_SCHEMA_PATH = _DOCS / "rfq_schema.json"
_EXPECTED_DIR = Path(__file__).parent.parent.parent / "samples" / "expected"

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _minimal_valid_payload() -> dict:
    """Return the first available expected sample as a dict."""
    samples = sorted(_EXPECTED_DIR.glob("*.json"))
    assert samples, "No expected sample files found"
    return json.loads(samples[0].read_text(encoding="utf-8"))


def _mock_response(text: str) -> MagicMock:
    """Return a mock that looks like a genai GenerateContentResponse."""
    resp = MagicMock()
    resp.text = text
    resp.usage_metadata = MagicMock(
        prompt_token_count=100,
        candidates_token_count=200,
        thoughts_token_count=50,
        total_token_count=350,
    )
    return resp


# ---------------------------------------------------------------------------
# 1. PROMPTS — system instruction must match docs file
# ---------------------------------------------------------------------------

def test_prompt_matches_docs_file() -> None:
    """SYSTEM_INSTRUCTION constant must be byte-identical to system_instructions.md."""
    on_disk = _SYSTEM_INSTRUCTIONS_PATH.read_text(encoding="utf-8")
    assert SYSTEM_INSTRUCTION == on_disk, (
        "app/prompts.py SYSTEM_INSTRUCTION has drifted from "
        "docs/system_instructions.md.  Update the docs file or the constant."
    )


def test_prompt_version_is_string() -> None:
    assert isinstance(PROMPT_VERSION, str) and PROMPT_VERSION, (
        "PROMPT_VERSION must be a non-empty string"
    )


# ---------------------------------------------------------------------------
# 2. SCHEMA CONTRACT — Pydantic models must mirror rfq_schema.json
# ---------------------------------------------------------------------------

def _collect_schema_fields(schema: dict, path: str = "") -> dict[str, set]:
    """Recursively collect {path: set_of_required_field_names} from a JSON schema."""
    result: dict[str, set] = {}
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    if required:
        result[path or "root"] = required
    for field_name, field_schema in props.items():
        child_path = f"{path}.{field_name}" if path else field_name
        if field_schema.get("type") == "object":
            result.update(_collect_schema_fields(field_schema, child_path))
        elif field_schema.get("type") == "array":
            items = field_schema.get("items", {})
            if items.get("type") == "object":
                result.update(_collect_schema_fields(items, f"{child_path}[]"))
    return result


def _collect_model_fields(model_cls, path: str = "") -> dict[str, set]:
    """Recursively collect required field names from Pydantic models."""
    import typing
    from pydantic_core import PydanticUndefined
    result: dict[str, set] = {}
    required: set[str] = set()

    from pydantic import BaseModel

    for fname, finfo in model_cls.model_fields.items():
        # A field is "required" in Pydantic v2 when it has no default and no
        # default_factory.
        is_required = (
            finfo.default is PydanticUndefined
            and finfo.default_factory is None
        )
        if is_required:
            required.add(fname)

        # Recurse into sub-models
        annotation = finfo.annotation
        # Unwrap Optional[X] → X
        origin = getattr(annotation, "__origin__", None)
        args = getattr(annotation, "__args__", ())
        if origin is typing.Union and type(None) in args:
            inner = [a for a in args if a is not type(None)]
            if inner:
                annotation = inner[0]
            origin = getattr(annotation, "__origin__", None)
            args = getattr(annotation, "__args__", ())

        child_path = f"{path}.{fname}" if path else fname
        try:
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                result.update(_collect_model_fields(annotation, child_path))
            elif origin is list and args:
                item_type = args[0]
                if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                    result.update(_collect_model_fields(item_type, f"{child_path}[]"))
        except TypeError:
            pass

    if required:
        result[path or "root"] = required
    return result


def test_pydantic_required_fields_match_schema() -> None:
    """Every required field in rfq_schema.json must be required in the Pydantic model
    and vice-versa, at every level of nesting.
    """
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    schema_required = _collect_schema_fields(schema)
    model_required = _collect_model_fields(RfqExtracted)

    mismatches: list[str] = []

    all_paths = set(schema_required) | set(model_required)
    for p in sorted(all_paths):
        s_req = schema_required.get(p, set())
        m_req = model_required.get(p, set())
        only_in_schema = s_req - m_req
        only_in_model  = m_req - s_req
        if only_in_schema:
            mismatches.append(
                f"  [{p}] required in schema but NOT in Pydantic: {sorted(only_in_schema)}"
            )
        if only_in_model:
            mismatches.append(
                f"  [{p}] required in Pydantic but NOT in schema: {sorted(only_in_model)}"
            )

    assert not mismatches, (
        "Pydantic models and rfq_schema.json have diverged:\n" + "\n".join(mismatches)
    )


def test_pydantic_property_names_match_schema() -> None:
    """Every property name in rfq_schema.json must exist as a field in the Pydantic
    model (and vice-versa) at the top level and in nested objects / array items.
    """
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))

    def _schema_props(s: dict) -> set[str]:
        return set(s.get("properties", {}).keys())

    def _model_props(model_cls) -> set[str]:
        return set(model_cls.model_fields.keys())

    checks = [
        ("root", schema, RfqExtracted),
        ("client", schema["properties"]["client"], Client),
        ("rfq_details", schema["properties"]["rfq_details"], RfqDetails),
        ("delivery", schema["properties"]["delivery"], Delivery),
        ("line_items[]", schema["properties"]["line_items"]["items"], LineItem),
        ("commercial_terms", schema["properties"]["commercial_terms"], CommercialTerms),
        ("attachments[]", schema["properties"]["attachments"]["items"], Attachment),
        ("line_items[].source_ref", schema["properties"]["line_items"]["items"]["properties"]["source_ref"], SourceRef),
    ]

    mismatches: list[str] = []
    for label, sub_schema, model_cls in checks:
        sp = _schema_props(sub_schema)
        mp = _model_props(model_cls)
        only_schema = sp - mp
        only_model  = mp - sp
        if only_schema:
            mismatches.append(f"  [{label}] in schema but not in model: {sorted(only_schema)}")
        if only_model:
            mismatches.append(f"  [{label}] in model but not in schema: {sorted(only_model)}")

    assert not mismatches, (
        "Field name mismatch between rfq_schema.json and Pydantic models:\n"
        + "\n".join(mismatches)
    )


# ---------------------------------------------------------------------------
# 3. _build_parts helper
# ---------------------------------------------------------------------------

def test_build_parts_text_plain_becomes_text_part() -> None:
    parts = _build_parts([(b"hello world", "text/plain")])
    assert len(parts) == 1
    assert parts[0].text == "hello world"
    assert parts[0].inline_data is None


def test_build_parts_pdf_becomes_inline_blob() -> None:
    fake_pdf = b"%PDF-1.4 ..."
    parts = _build_parts([(fake_pdf, "application/pdf")])
    assert len(parts) == 1
    assert parts[0].inline_data is not None
    assert parts[0].inline_data.mime_type == "application/pdf"
    assert parts[0].inline_data.data == fake_pdf


def test_build_parts_image_becomes_inline_blob() -> None:
    fake_img = b"\xff\xd8\xff"  # JPEG magic bytes
    parts = _build_parts([(fake_img, "image/jpeg")])
    assert len(parts) == 1
    assert parts[0].inline_data.mime_type == "image/jpeg"


def test_build_parts_multiple_files() -> None:
    file_parts = [
        (b"text content", "text/plain"),
        (b"%PDF-...", "application/pdf"),
        (b"\x89PNG", "image/png"),
    ]
    parts = _build_parts(file_parts)
    assert len(parts) == 3
    assert parts[0].text == "text content"
    assert parts[1].inline_data.mime_type == "application/pdf"
    assert parts[2].inline_data.mime_type == "image/png"


# ---------------------------------------------------------------------------
# 4. _is_retryable
# ---------------------------------------------------------------------------

def test_server_error_is_retryable() -> None:
    exc = ServerError(500, {"message": "internal"})
    assert _is_retryable(exc) is True


def test_rate_limit_is_retryable() -> None:
    exc = ClientError(429, {"message": "rate limit"})
    assert _is_retryable(exc) is True


def test_bad_request_not_retryable() -> None:
    exc = ClientError(400, {"message": "bad request"})
    assert _is_retryable(exc) is False


def test_auth_error_not_retryable() -> None:
    exc = ClientError(401, {"message": "unauthorized"})
    assert _is_retryable(exc) is False


def test_not_found_not_retryable() -> None:
    exc = ClientError(404, {"message": "not found"})
    assert _is_retryable(exc) is False


def test_generic_exception_not_retryable() -> None:
    assert _is_retryable(ValueError("oops")) is False


# ---------------------------------------------------------------------------
# 5. extract_rfq — mocked Gemini client
# ---------------------------------------------------------------------------

_MOCK_TARGET = "app.services.gemini._build_client"


def _valid_json_response() -> str:
    return json.dumps(_minimal_valid_payload())


@patch(_MOCK_TARGET)
def test_extract_rfq_success(mock_build_client) -> None:
    """Happy path: Gemini returns valid JSON → extract_rfq returns a dict."""
    mock_client = MagicMock()
    mock_build_client.return_value = mock_client
    mock_client.models.generate_content.return_value = _mock_response(
        _valid_json_response()
    )

    result = extract_rfq([(b"some text", "text/plain")], request_id="req-001")

    assert isinstance(result, dict)
    assert "line_items" in result
    mock_client.models.generate_content.assert_called_once()


@patch(_MOCK_TARGET)
def test_extract_rfq_result_validates_against_pydantic(mock_build_client) -> None:
    """The dict returned by extract_rfq must be valid against RfqExtracted."""
    mock_client = MagicMock()
    mock_build_client.return_value = mock_client
    mock_client.models.generate_content.return_value = _mock_response(
        _valid_json_response()
    )

    result = extract_rfq([(b"some text", "text/plain")], request_id="req-002")
    # Should not raise
    RfqExtracted.model_validate(result)


@patch(_MOCK_TARGET)
def test_extract_rfq_empty_response_raises(mock_build_client) -> None:
    """An empty/None response text must raise ExtractionError."""
    mock_client = MagicMock()
    mock_build_client.return_value = mock_client
    mock_client.models.generate_content.return_value = _mock_response("")

    with pytest.raises(ExtractionError):
        extract_rfq([(b"text", "text/plain")], request_id="req-003")


@patch(_MOCK_TARGET)
def test_extract_rfq_invalid_json_raises(mock_build_client) -> None:
    """Non-JSON model output must raise ExtractionError."""
    mock_client = MagicMock()
    mock_build_client.return_value = mock_client
    mock_client.models.generate_content.return_value = _mock_response(
        "this is not json {{"
    )

    with pytest.raises(ExtractionError):
        extract_rfq([(b"text", "text/plain")], request_id="req-004")


@patch(_MOCK_TARGET)
def test_extract_rfq_non_retryable_4xx_raises_immediately(mock_build_client) -> None:
    """A 400 error must raise ExtractionError without retrying."""
    mock_client = MagicMock()
    mock_build_client.return_value = mock_client
    mock_client.models.generate_content.side_effect = ClientError(
        400, {"message": "invalid argument"}
    )

    with pytest.raises(ExtractionError):
        extract_rfq([(b"text", "text/plain")], request_id="req-005")

    # Must have been called exactly once — no retry
    assert mock_client.models.generate_content.call_count == 1


@patch("time.sleep", return_value=None)  # don't actually sleep in tests
@patch(_MOCK_TARGET)
def test_extract_rfq_retries_on_server_error(mock_build_client, mock_sleep) -> None:
    """A 500 error should be retried up to MAX_ATTEMPTS times."""
    mock_client = MagicMock()
    mock_build_client.return_value = mock_client
    mock_client.models.generate_content.side_effect = ServerError(
        500, {"message": "internal server error"}
    )

    with pytest.raises(ExtractionError):
        extract_rfq([(b"text", "text/plain")], request_id="req-006")

    # Should have tried 3 times total
    assert mock_client.models.generate_content.call_count == 3


@patch("time.sleep", return_value=None)
@patch(_MOCK_TARGET)
def test_extract_rfq_retries_on_rate_limit(mock_build_client, mock_sleep) -> None:
    """A 429 error should be retried."""
    mock_client = MagicMock()
    mock_build_client.return_value = mock_client
    # Fail twice with 429, then succeed
    mock_client.models.generate_content.side_effect = [
        ClientError(429, {"message": "quota exceeded"}),
        ClientError(429, {"message": "quota exceeded"}),
        _mock_response(_valid_json_response()),
    ]

    result = extract_rfq([(b"text", "text/plain")], request_id="req-007")

    assert isinstance(result, dict)
    assert mock_client.models.generate_content.call_count == 3


@patch("time.sleep", return_value=None)
@patch(_MOCK_TARGET)
def test_extract_rfq_exponential_backoff(mock_build_client, mock_sleep) -> None:
    """sleep() should be called with exponentially increasing durations."""
    mock_client = MagicMock()
    mock_build_client.return_value = mock_client
    mock_client.models.generate_content.side_effect = ServerError(
        500, {"message": "server error"}
    )

    with pytest.raises(ExtractionError):
        extract_rfq([(b"text", "text/plain")], request_id="req-008")

    sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
    # Should have slept twice (between attempts 1→2 and 2→3)
    assert len(sleep_calls) == 2
    # Backoff should be strictly increasing (2^0=1 and 2^1=2)
    assert sleep_calls[0] < sleep_calls[1]


@patch(_MOCK_TARGET)
def test_extract_rfq_exact_config_values(mock_build_client) -> None:
    """Mocked test asserting the exact config values sent:
    - thinking_level: types.ThinkingLevel.LOW (and thinking_budget is None)
    - temperature: None (unset, allowing model default 1.0)
    - media_resolution: None (when default MEDIA_RESOLUTION="default")
    - response_schema: load_rfq_schema()
    - response_mime_type: "application/json"
    - system_instruction: SYSTEM_INSTRUCTION
    """
    mock_client = MagicMock()
    mock_build_client.return_value = mock_client
    mock_client.models.generate_content.return_value = _mock_response(
        _valid_json_response()
    )

    extract_rfq([(b"text", "text/plain")], request_id="req-exact-cfg")

    _, kwargs = mock_client.models.generate_content.call_args
    cfg = kwargs.get("config")
    assert cfg is not None, "config was not passed to generate_content"

    # 1. Thinking level must be LOW, and thinking_budget must NOT be set
    assert cfg.thinking_config is not None
    assert cfg.thinking_config.thinking_level == types.ThinkingLevel.LOW
    assert cfg.thinking_config.thinking_budget is None

    # 2. Temperature must not be set (None -> model default 1.0)
    assert cfg.temperature is None

    # 3. Media resolution: default is None (unset, matching AI Studio)
    assert cfg.media_resolution is None

    # 4. Schema: exact JSON schema from docs/rfq_schema.json
    expected_schema = load_rfq_schema()
    assert cfg.response_schema == expected_schema

    # 5. MIME type: application/json
    assert cfg.response_mime_type == "application/json"

    # 6. System instruction: verbatim from app/prompts.py
    assert cfg.system_instruction == SYSTEM_INSTRUCTION


@pytest.mark.parametrize(
    "env_val,expected_enum",
    [
        ("default", None),
        ("low", types.MediaResolution.MEDIA_RESOLUTION_LOW),
        ("medium", types.MediaResolution.MEDIA_RESOLUTION_MEDIUM),
        ("high", types.MediaResolution.MEDIA_RESOLUTION_HIGH),
    ],
)
@patch(_MOCK_TARGET)
def test_extract_rfq_media_resolution_setting(mock_build_client, env_val, expected_enum, monkeypatch) -> None:
    """Assert that MEDIA_RESOLUTION setting maps correctly to config.media_resolution."""
    from app.config import get_settings
    monkeypatch.setattr(get_settings(), "MEDIA_RESOLUTION", env_val)

    mock_client = MagicMock()
    mock_build_client.return_value = mock_client
    mock_client.models.generate_content.return_value = _mock_response(
        _valid_json_response()
    )

    extract_rfq([(b"text", "text/plain")], request_id="req-media-res")

    _, kwargs = mock_client.models.generate_content.call_args
    cfg = kwargs.get("config")
    assert cfg.media_resolution == expected_enum


def test_config_media_resolution_validation() -> None:
    """Assert that Settings validates MEDIA_RESOLUTION allowed values."""
    from app.config import Settings
    for val in ["default", "low", "medium", "high", "LOW", "MEDIUM", "HIGH", "DEFAULT"]:
        s = Settings(
            GEMINI_API_KEY="test",
            SERVICE_API_KEY="test",
            GEMINI_MODEL="gemini-3.8-flash",
            MEDIA_RESOLUTION=val,
        )
        assert s.MEDIA_RESOLUTION == val.lower()

    with pytest.raises(ValueError, match="MEDIA_RESOLUTION must be one of"):
        Settings(
            GEMINI_API_KEY="test",
            SERVICE_API_KEY="test",
            GEMINI_MODEL="gemini-3.8-flash",
            MEDIA_RESOLUTION="ultra_high",
        )


@patch(_MOCK_TARGET)
def test_extract_rfq_multiple_files(mock_build_client) -> None:
    """Multiple file parts in one call must all appear in the request contents."""
    mock_client = MagicMock()
    mock_build_client.return_value = mock_client
    mock_client.models.generate_content.return_value = _mock_response(
        _valid_json_response()
    )

    file_parts = [
        (b"email text", "text/plain"),
        (b"%PDF...", "application/pdf"),
        (b"\x89PNG...", "image/png"),
    ]
    extract_rfq(file_parts, request_id="req-012")

    _, kwargs = mock_client.models.generate_content.call_args
    contents = kwargs.get("contents")
    assert contents is not None
    # The single Content object should have 3 parts
    assert len(contents) == 1
    assert len(contents[0].parts) == 3


@patch(_MOCK_TARGET)
def test_extract_rfq_error_message_does_not_contain_api_key(mock_build_client) -> None:
    """ExtractionError messages must never echo the API key."""
    mock_client = MagicMock()
    mock_build_client.return_value = mock_client
    mock_client.models.generate_content.side_effect = ClientError(
        400, {"message": "bad request with key=SUPERSECRET"}
    )

    with pytest.raises(ExtractionError) as exc_info:
        extract_rfq([(b"text", "text/plain")], request_id="req-013")

    assert "SUPERSECRET" not in str(exc_info.value)
    assert "SUPERSECRET" not in exc_info.value.args[0]


@patch("time.sleep", return_value=None)
@patch(_MOCK_TARGET)
def test_extract_rfq_retryable_failure_logging_does_not_leak_message(
    mock_build_client, mock_sleep, caplog
) -> None:
    """Each failed retryable attempt must log class name, code, status string, and attempt number,

    and must NOT log the exception message text.
    """
    mock_client = MagicMock()
    mock_build_client.return_value = mock_client

    secret_prompt = "CONFIDENTIAL_BID_DATA_DO_NOT_LOG_98765"
    fake_error = ClientError(
        429,
        {
            "error": {
                "code": 429,
                "message": f"Resource has been exhausted: {secret_prompt}",
                "status": "RESOURCE_EXHAUSTED",
            }
        },
    )
    mock_client.models.generate_content.side_effect = fake_error

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ExtractionError):
            extract_rfq([(b"sample text", "text/plain")], request_id="req-test-retry-log")

    # Class name and status code must be present in the log
    assert "ClientError" in caplog.text
    assert "429" in caplog.text

    # Google error status string must be present in the log
    assert "RESOURCE_EXHAUSTED" in caplog.text

    # Attempt numbers must be logged
    assert "attempt 1" in caplog.text
    assert "attempt 2" in caplog.text
    assert "attempt 3" in caplog.text

    # Exception message text must NEVER be logged
    assert secret_prompt not in caplog.text


@patch(_MOCK_TARGET)
def test_extract_rfq_non_retryable_failure_logging_does_not_leak_message(
    mock_build_client, caplog
) -> None:
    """A failed non-retryable attempt must log class name, code, status string, and attempt number,

    and must NOT log the exception message text.
    """
    mock_client = MagicMock()
    mock_build_client.return_value = mock_client

    secret_leak = "LEAKY_INTERNAL_PROMPT_DETAILS_12345"
    fake_error = ClientError(
        400,
        {
            "error": {
                "code": 400,
                "message": f"Invalid field in request: {secret_leak}",
                "status": "INVALID_ARGUMENT",
            }
        },
    )
    mock_client.models.generate_content.side_effect = fake_error

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ExtractionError):
            extract_rfq([(b"sample text", "text/plain")], request_id="req-test-non-retry-log")

    # Class name, status code, and error status must be present
    assert "ClientError" in caplog.text
    assert "400" in caplog.text
    assert "INVALID_ARGUMENT" in caplog.text
    assert "attempt 1" in caplog.text

    # Exception message text must NEVER be logged
    assert secret_leak not in caplog.text
