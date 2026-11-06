"""Unit tests for POST /extract and storage layer.

Tests cover:
- Authentication: missing key and wrong key return 401
- LocalStorage security: path traversal key, absolute key, URL-scheme key, and symlink escape are rejected
- Media type detection: unsupported file type returns 415, and file named .pdf with non-PDF bytes is rejected
- Size limits: oversize file returns 413 without downloading it
- File count bounds: > 5 files is rejected with 422
- Success path: valid files return HTTP 200 and ExtractionResult
- Generic 500 error: unexpected exceptions return generic error shape without internal details
- Docs disabled: /docs returns 404 when ENABLE_DOCS=false
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.schemas import ExtractionResult, RfqExtracted
from app.services.storage import (
    InvalidStorageKeyError,
    LocalStorage,
    ObjectNotFoundError,
)

_CLIENT = TestClient(app, raise_server_exceptions=False)
_VALID_API_KEY = get_settings().SERVICE_API_KEY.get_secret_value()
_SAMPLES_EXPECTED = Path(__file__).parent.parent.parent / "samples" / "expected" / "rfq_email.json"


# ---------------------------------------------------------------------------
# 1. Authentication
# ---------------------------------------------------------------------------

def test_extract_missing_api_key_returns_401() -> None:
    """Missing X-API-Key header must return 401."""
    resp = _CLIENT.post("/extract", json={"files": [{"storage_key": "k", "original_filename": "f"}]})
    assert resp.status_code == 401
    assert resp.json() == {
        "error": {"code": "unauthorized", "message": "Invalid or missing API key."}
    }


def test_extract_wrong_api_key_returns_401() -> None:
    """Incorrect X-API-Key header must return 401."""
    resp = _CLIENT.post(
        "/extract",
        headers={"X-API-Key": "completely-wrong-key"},
        json={"files": [{"storage_key": "k", "original_filename": "f"}]},
    )
    assert resp.status_code == 401
    assert resp.json() == {
        "error": {"code": "unauthorized", "message": "Invalid or missing API key."}
    }


# ---------------------------------------------------------------------------
# 2. LocalStorage Key Security & Traversal
# ---------------------------------------------------------------------------

def test_local_storage_key_traversal_rejected() -> None:
    """Path traversal with '..' must be rejected."""
    storage = LocalStorage("./samples")
    with pytest.raises(InvalidStorageKeyError):
        storage.get_size("../outside.txt")
    with pytest.raises(InvalidStorageKeyError):
        storage.download("../outside.txt")


def test_local_storage_absolute_key_rejected() -> None:
    """Absolute paths starting with '/' must be rejected."""
    storage = LocalStorage("./samples")
    with pytest.raises(InvalidStorageKeyError):
        storage.get_size("/etc/passwd")


def test_local_storage_url_scheme_key_rejected() -> None:
    """URL schemes like 'http:' or 's3:' must be rejected."""
    storage = LocalStorage("./samples")
    for bad_key in ["http://attacker.com/file", "https://s3.amazonaws.com/b", "s3://my-bucket/key", "file:///etc/hosts"]:
        with pytest.raises(InvalidStorageKeyError):
            storage.get_size(bad_key)


def test_local_storage_symlink_escape_rejected() -> None:
    """A symlink or resolved path escaping the base directory must be rejected."""
    storage = LocalStorage(Path(__file__).parent)
    outside_path = Path("C:/some_other_folder/outside.txt")

    with patch.object(Path, "resolve", return_value=outside_path):
        with pytest.raises(InvalidStorageKeyError):
            storage.get_size("symlink_to_outside")
        with pytest.raises(InvalidStorageKeyError):
            storage.download("symlink_to_outside")


def test_s3_storage_startup_missing_vars_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """When STORAGE_PROVIDER='s3', missing variables must raise ValueError naming them."""
    from app.config import Settings
    with pytest.raises(ValueError, match="STORAGE_BUCKET, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY"):
        Settings(
            GEMINI_API_KEY="dummy",
            SERVICE_API_KEY="dummy",
            STORAGE_PROVIDER="s3",
            STORAGE_BUCKET="",
            S3_ACCESS_KEY_ID="",
            S3_SECRET_ACCESS_KEY=None,
        )



# ---------------------------------------------------------------------------
# 3. File Type Detection (415)
# ---------------------------------------------------------------------------

@patch("app.main.get_storage")
def test_extract_unsupported_file_type_returns_415(mock_get_storage) -> None:
    """Files with arbitrary binary bytes not matching any supported magic bytes return 415."""
    mock_storage = MagicMock()
    mock_storage.get_size.return_value = 100
    # Arbitrary binary data containing NUL byte (not PDF, not image, not UTF-8 plain text)
    mock_storage.download.return_value = b"\x00\x01\x02\x03random_binary"
    mock_get_storage.return_value = mock_storage

    resp = _CLIENT.post(
        "/extract",
        headers={"X-API-Key": _VALID_API_KEY},
        json={"files": [{"storage_key": "unsupported.bin", "original_filename": "unsupported.bin"}]},
    )
    assert resp.status_code == 415
    assert resp.json()["error"]["code"] == "unsupported_media_type"


@patch("app.main.get_storage")
def test_extract_fake_pdf_name_with_non_pdf_bytes_rejected(mock_get_storage) -> None:
    """A file named something.pdf whose bytes do NOT start with %PDF must be rejected (415)."""
    mock_storage = MagicMock()
    mock_storage.get_size.return_value = 50
    # Binary bytes not starting with %PDF
    mock_storage.download.return_value = b"\x00\x01not a pdf document at all"
    mock_get_storage.return_value = mock_storage

    resp = _CLIENT.post(
        "/extract",
        headers={"X-API-Key": _VALID_API_KEY},
        json={"files": [{"storage_key": "fake.pdf", "original_filename": "fake.pdf"}]},
    )
    assert resp.status_code == 415
    assert resp.json()["error"]["code"] == "unsupported_media_type"


# ---------------------------------------------------------------------------
# 4. Oversize File Check (413 without downloading)
# ---------------------------------------------------------------------------

@patch("app.main.get_storage")
def test_extract_oversize_file_returns_413_without_downloading(mock_get_storage) -> None:
    """If file size exceeds MAX_FILE_MB, return 413 immediately without calling download()."""
    mock_storage = MagicMock()
    # 16 MB when limit is 15 MB
    oversize_bytes = 16 * 1024 * 1024
    mock_storage.get_size.return_value = oversize_bytes
    mock_get_storage.return_value = mock_storage

    resp = _CLIENT.post(
        "/extract",
        headers={"X-API-Key": _VALID_API_KEY},
        json={"files": [{"storage_key": "huge.pdf", "original_filename": "huge.pdf"}]},
    )

    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "payload_too_large"
    # download() must NEVER be called for an oversize file
    mock_storage.download.assert_not_called()


# ---------------------------------------------------------------------------
# 5. File Count Bounds (> 5 files rejected with 422)
# ---------------------------------------------------------------------------

def test_extract_more_than_five_files_rejected() -> None:
    """Requesting more than 5 files must return 422."""
    six_files = [{"storage_key": f"key_{i}.pdf", "original_filename": f"file_{i}.pdf"} for i in range(6)]
    resp = _CLIENT.post(
        "/extract",
        headers={"X-API-Key": _VALID_API_KEY},
        json={"files": six_files},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_request"


# ---------------------------------------------------------------------------
# 6. Success Path
# ---------------------------------------------------------------------------

@patch("app.main.extract_rfq_from_files")
@patch("app.main.get_storage")
def test_extract_success_path_returns_extraction_result(mock_get_storage, mock_extract) -> None:
    """Valid request returns HTTP 200 with ExtractionResult."""
    mock_storage = MagicMock()
    mock_storage.get_size.return_value = 1024
    # Valid PDF magic bytes
    mock_storage.download.return_value = b"%PDF-1.4 mock pdf content"
    mock_get_storage.return_value = mock_storage

    expected_rfq = RfqExtracted.model_validate(json.loads(_SAMPLES_EXPECTED.read_text(encoding="utf-8")))
    mock_result = ExtractionResult(
        schema_version="1.0",
        request_id="req-12345",
        status="ok",
        warnings=[],
        model_name="gemini-3.8-flash",
        prompt_version="1.0",
        rfq=expected_rfq,
    )
    mock_extract.return_value = mock_result

    resp = _CLIENT.post(
        "/extract",
        headers={"X-API-Key": _VALID_API_KEY},
        json={"files": [{"storage_key": "rfq.pdf", "original_filename": "client_rfq.pdf"}]},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["schema_version"] == "1.0"
    assert data["request_id"] == "req-12345"
    assert data["rfq"] is not None
    assert len(data["rfq"]["line_items"]) == len(expected_rfq.line_items)


# ---------------------------------------------------------------------------
# 7. Generic 500 Error Handler (No Leakage)
# ---------------------------------------------------------------------------

@patch("app.main.get_storage")
def test_extract_unexpected_exception_returns_generic_500(mock_get_storage) -> None:
    """Unexpected exceptions must return generic 500 shape without leaking internal details."""
    mock_storage = MagicMock()
    mock_storage.get_size.side_effect = RuntimeError("AWS_SECRET_LEAK_database_url_at_/var/run/secret")
    mock_get_storage.return_value = mock_storage

    resp = _CLIENT.post(
        "/extract",
        headers={"X-API-Key": _VALID_API_KEY},
        json={"files": [{"storage_key": "some.pdf", "original_filename": "some.pdf"}]},
    )

    assert resp.status_code == 500
    body = resp.json()
    assert body == {
        "error": {
            "code": "internal_error",
            "message": "An internal error occurred. Please try again later.",
        }
    }
    # Assert sensitive leaked content is NOT in response text
    assert "AWS_SECRET_LEAK" not in resp.text
    assert "/var/run" not in resp.text


# ---------------------------------------------------------------------------
# 8. Docs Disabled When ENABLE_DOCS=false
# ---------------------------------------------------------------------------

def test_docs_disabled_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ENABLE_DOCS=False, /docs, /redoc, and /openapi.json must return 404."""
    import importlib
    monkeypatch.setenv("ENABLE_DOCS", "false")
    import app.config
    import app.main

    importlib.reload(app.config)
    reloaded_main = importlib.reload(app.main)
    try:
        client = TestClient(reloaded_main.app, raise_server_exceptions=False)
        assert client.get("/docs").status_code == 404
        assert client.get("/redoc").status_code == 404
        assert client.get("/openapi.json").status_code == 404
    finally:
        monkeypatch.undo()
        importlib.reload(app.config)
        importlib.reload(app.main)


# ---------------------------------------------------------------------------
# 9. Direct Multipart Upload (POST /extract-upload)
# ---------------------------------------------------------------------------

def test_extract_upload_missing_api_key_returns_401() -> None:
    """POST /extract-upload without X-API-Key must return 401 before reading body."""
    resp = _CLIENT.post(
        "/extract-upload",
        files=[("files", ("test.pdf", b"%PDF-1.4 mock", "application/pdf"))],
    )
    assert resp.status_code == 401
    assert resp.json() == {
        "error": {"code": "unauthorized", "message": "Invalid or missing API key."}
    }


def test_extract_upload_wrong_api_key_returns_401() -> None:
    """POST /extract-upload with invalid X-API-Key must return 401 before reading body."""
    resp = _CLIENT.post(
        "/extract-upload",
        headers={"X-API-Key": "completely-wrong-key"},
        files=[("files", ("test.pdf", b"%PDF-1.4 mock", "application/pdf"))],
    )
    assert resp.status_code == 401
    assert resp.json() == {
        "error": {"code": "unauthorized", "message": "Invalid or missing API key."}
    }


def test_extract_upload_unsupported_media_type_returns_415() -> None:
    """Arbitrary binary data not matching any supported magic bytes must return 415."""
    resp = _CLIENT.post(
        "/extract-upload",
        headers={"X-API-Key": _VALID_API_KEY},
        files=[("files", ("unsupported.bin", b"\x00\x01\x02\x03random_binary", "application/octet-stream"))],
    )
    assert resp.status_code == 415
    assert resp.json()["error"]["code"] == "unsupported_media_type"


def test_extract_upload_fake_pdf_returns_415() -> None:
    """File named .pdf whose content is plain text (not PDF magic bytes) must return 415."""
    resp = _CLIENT.post(
        "/extract-upload",
        headers={"X-API-Key": _VALID_API_KEY},
        files=[("files", ("fake.pdf", b"This is plain text and not a real pdf", "application/pdf"))],
    )
    assert resp.status_code == 415
    assert resp.json()["error"]["code"] == "unsupported_media_type"


def test_extract_upload_oversize_content_returns_413() -> None:
    """File exceeding 4 MB limit must return 413 payload_too_large."""
    oversize_data = b"%PDF" + b"0" * (4 * 1024 * 1024 + 10)
    resp = _CLIENT.post(
        "/extract-upload",
        headers={"X-API-Key": _VALID_API_KEY},
        files=[("files", ("oversize.pdf", oversize_data, "application/pdf"))],
    )
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "payload_too_large"


def test_extract_upload_oversize_content_length_header_returns_413() -> None:
    """Request with Content-Length header exceeding 4 MB must return 413 before reading body."""
    resp = _CLIENT.post(
        "/extract-upload",
        headers={
            "X-API-Key": _VALID_API_KEY,
            "Content-Length": str(5 * 1024 * 1024),
            "Content-Type": "multipart/form-data; boundary=boundary",
        },
        content=b"",
    )
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "payload_too_large"


def test_extract_upload_file_count_bounds_rejected() -> None:
    """Uploading more than 5 files must return 422 invalid_request."""
    six_files = [("files", (f"file_{i}.pdf", b"%PDF-1.4", "application/pdf")) for i in range(6)]
    resp = _CLIENT.post(
        "/extract-upload",
        headers={"X-API-Key": _VALID_API_KEY},
        files=six_files,
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_request"


@patch("app.main.extract_rfq_from_files")
def test_extract_upload_success_path(mock_extract) -> None:
    """Valid multipart upload returns HTTP 200 with ExtractionResult."""
    expected_rfq = RfqExtracted.model_validate(json.loads(_SAMPLES_EXPECTED.read_text(encoding="utf-8")))
    mock_result = ExtractionResult(
        schema_version="1.0",
        request_id="upload-req-999",
        status="ok",
        warnings=[],
        model_name="gemini-3.8-flash",
        prompt_version="1.0",
        rfq=expected_rfq,
    )
    mock_extract.return_value = mock_result

    resp = _CLIENT.post(
        "/extract-upload",
        headers={"X-API-Key": _VALID_API_KEY},
        files=[("files", ("invoice.pdf", b"%PDF-1.4 valid content", "application/pdf"))],
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["request_id"] == "upload-req-999"
    assert data["rfq"] is not None
    assert len(data["rfq"]["line_items"]) == len(expected_rfq.line_items)


@patch("app.main.extract_rfq_from_files")
def test_extract_upload_unexpected_exception_returns_generic_500(mock_extract) -> None:
    """Unexpected exception in extraction pipeline returns generic 500 without leaking secrets."""
    mock_extract.side_effect = RuntimeError("AWS_SECRET_LEAK_database_url_at_/var/run/secret")
    resp = _CLIENT.post(
        "/extract-upload",
        headers={"X-API-Key": _VALID_API_KEY},
        files=[("files", ("invoice.pdf", b"%PDF-1.4 valid content", "application/pdf"))],
    )
    assert resp.status_code == 500
    assert resp.json() == {
        "error": {
            "code": "internal_error",
            "message": "An internal error occurred. Please try again later.",
        }
    }
    assert "AWS_SECRET_LEAK" not in resp.text
    assert "/var/run" not in resp.text


# ---------------------------------------------------------------------------
# 10. Uniform 404 and 405 Error Shape
# ---------------------------------------------------------------------------

def test_unmatched_route_returns_uniform_404() -> None:
    """Unmatched route must return the uniform error shape {"error": {"code": "not_found", "message": "Not Found"}}."""
    resp = _CLIENT.get("/nonexistent-endpoint-404")
    assert resp.status_code == 404
    assert resp.json() == {
        "error": {"code": "not_found", "message": "Not Found"}
    }


def test_wrong_method_returns_uniform_405() -> None:
    """Wrong HTTP method must return the uniform error shape {"error": {"code": "method_not_allowed", "message": "Method Not Allowed"}}."""
    resp = _CLIENT.get("/extract-upload")
    assert resp.status_code == 405
    assert resp.json() == {
        "error": {"code": "method_not_allowed", "message": "Method Not Allowed"}
    }


