"""Unit tests for app/services/validator.py and app/services/extractor.py.

All Gemini API calls are mocked — no real network requests are made.
Tests verify:
- ok status with samples/expected/*.json
- needs_review statuses: empty line_items, missing quantity_raw, duplicate line_no
- failed status on schema mismatch
- safe handling on ExtractionError
- privacy assertion: validation failures never leak extracted text into warnings
- source_ref.file is overwritten with original filename and source_ref.page is null for text/plain
- multiple file source_ref handling (PDF keeps page, image clears page, unmatched cleared with warning)
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from app.schemas import ExtractionResult, RfqExtracted
from app.services.extractor import extract_rfq_from_files
from app.services.gemini import ExtractionError
from app.services.validator import validate_rfq_dict

_EXPECTED_DIR = Path(__file__).parent.parent.parent / "samples" / "expected"
_MOCK_GEMINI_CALL = "app.services.extractor.extract_rfq"


def _load_expected_sample(filename: str = "rfq_email.json") -> dict:
    """Load an expected JSON sample as a raw dict."""
    path = _EXPECTED_DIR / filename
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. OK case
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sample_name", ["rfq_email.json", "rfq_sample.pdf.json", "rfq_sample.jpeg.json"])
@patch(_MOCK_GEMINI_CALL)
def test_extractor_ok_samples(mock_extract_rfq, sample_name: str) -> None:
    """ExtractionResult should be status='ok' for valid samples with valid line items."""
    sample_dict = _load_expected_sample(sample_name)
    mock_extract_rfq.return_value = sample_dict

    files = [(b"file content", "text/plain", "rfq_input.txt")]
    result = extract_rfq_from_files(files, request_id="test-req-ok")

    assert isinstance(result, ExtractionResult)
    assert result.status == "ok"
    assert result.warnings == []
    assert result.rfq is not None
    assert result.request_id == "test-req-ok"
    assert result.schema_version == "1.0"
    assert result.prompt_version == "1.0"
    assert len(result.rfq.line_items) == len(sample_dict["line_items"])


# ---------------------------------------------------------------------------
# 2. needs_review cases
# ---------------------------------------------------------------------------

@patch(_MOCK_GEMINI_CALL)
def test_extractor_needs_review_empty_line_items(mock_extract_rfq) -> None:
    """Empty line_items list must flag a warning and set status='needs_review'."""
    sample = copy.deepcopy(_load_expected_sample("rfq_email.json"))
    sample["line_items"] = []
    mock_extract_rfq.return_value = sample

    result = extract_rfq_from_files([(b"content", "text/plain", "rfq.txt")])

    assert result.status == "needs_review"
    assert result.rfq is not None
    assert any("line_items is empty" in w for w in result.warnings)


@patch(_MOCK_GEMINI_CALL)
def test_extractor_needs_review_missing_quantity_raw(mock_extract_rfq) -> None:
    """If quantity is non-null but quantity_raw is null, flag a warning."""
    sample = copy.deepcopy(_load_expected_sample("rfq_email.json"))
    assert len(sample["line_items"]) > 0
    sample["line_items"][0]["quantity"] = 5.0
    sample["line_items"][0]["quantity_raw"] = None
    mock_extract_rfq.return_value = sample

    result = extract_rfq_from_files([(b"content", "text/plain", "rfq.txt")])

    assert result.status == "needs_review"
    assert result.rfq is not None
    assert any("quantity is present but quantity_raw is missing" in w for w in result.warnings)


@patch(_MOCK_GEMINI_CALL)
def test_extractor_needs_review_duplicate_line_no(mock_extract_rfq) -> None:
    """Duplicate line_no values must be flagged with a warning."""
    sample = copy.deepcopy(_load_expected_sample("rfq_email.json"))
    assert len(sample["line_items"]) >= 2
    # Set second line item to duplicate the first line_no
    sample["line_items"][1]["line_no"] = sample["line_items"][0]["line_no"]
    mock_extract_rfq.return_value = sample

    result = extract_rfq_from_files([(b"content", "text/plain", "rfq.txt")])

    assert result.status == "needs_review"
    assert result.rfq is not None
    assert any("duplicate line_no" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# 3. failed cases (invalid schema & ExtractionError)
# ---------------------------------------------------------------------------

@patch(_MOCK_GEMINI_CALL)
def test_extractor_failed_invalid_json_structure(mock_extract_rfq) -> None:
    """If model output violates the Pydantic schema, status must be 'failed' and rfq null."""
    # Corrupt structure: missing required top-level field 'client' and extra forbidden field
    bad_output = {
        "document_language": "en",
        "unknown_field": "invalid",
    }
    mock_extract_rfq.return_value = bad_output

    result = extract_rfq_from_files([(b"content", "text/plain", "rfq.txt")])

    assert result.status == "failed"
    assert result.rfq is None
    assert len(result.warnings) > 0
    assert any("schema validation error" in w for w in result.warnings)


@patch(_MOCK_GEMINI_CALL)
def test_extractor_failed_extraction_error(mock_extract_rfq) -> None:
    """When Gemini client raises ExtractionError, result is status='failed' with safe message."""
    mock_extract_rfq.side_effect = ExtractionError("Service unavailable after retries")

    result = extract_rfq_from_files([(b"content", "text/plain", "rfq.txt")])

    assert result.status == "failed"
    assert result.rfq is None
    assert result.warnings == ["Service unavailable after retries"]


# ---------------------------------------------------------------------------
# 4. Privacy: failed validation never puts extracted text into warnings
# ---------------------------------------------------------------------------

@patch(_MOCK_GEMINI_CALL)
def test_failed_validation_does_not_leak_extracted_value_in_warnings(mock_extract_rfq) -> None:
    """Assert that a schema failure NEVER exposes sensitive extracted text in warnings."""
    sample = copy.deepcopy(_load_expected_sample("rfq_email.json"))
    # Recognizeable sensitive company name
    secret_name = "Ganesh Material Handling Industries Pvt. Ltd."
    assert sample["client"]["company_name"] == secret_name

    # Corrupt another field to trigger a ValidationError (e.g. client.email is not an allowed type or extra field)
    sample["client"]["extra_secret_key"] = "forbidden_value"
    mock_extract_rfq.return_value = sample

    result = extract_rfq_from_files([(b"content", "text/plain", "rfq.txt")])

    assert result.status == "failed"
    assert result.rfq is None
    # Verify that secret_name or forbidden_value is NEVER present in any warning
    all_warnings_text = " ".join(result.warnings)
    assert secret_name not in all_warnings_text
    assert "forbidden_value" not in all_warnings_text
    assert "Ganesh" not in all_warnings_text


# ---------------------------------------------------------------------------
# 5. source_ref.file overwrite and source_ref.page null for text/plain
# ---------------------------------------------------------------------------

@patch(_MOCK_GEMINI_CALL)
def test_source_ref_overwritten_and_page_null_for_text_plain(mock_extract_rfq) -> None:
    """Single text/plain file must overwrite source_ref.file with actual name and clear page."""
    sample = copy.deepcopy(_load_expected_sample("rfq_email.json"))
    # Simulate model hallucinating file name and page
    for item in sample["line_items"]:
        item["source_ref"]["file"] = "hallucinated_model_file.pdf"
        item["source_ref"]["page"] = 42

    mock_extract_rfq.return_value = sample

    actual_filename = "verified_original_rfq.txt"
    result = extract_rfq_from_files([(b"email content", "text/plain", actual_filename)])

    assert result.status == "ok"
    assert result.rfq is not None
    for item in result.rfq.line_items:
        # source_ref.file must be overwritten with original filename
        assert item.source_ref.file == actual_filename
        # source_ref.page must be None for text/plain
        assert item.source_ref.page is None


# ---------------------------------------------------------------------------
# 6. Multiple files source_ref handling
# ---------------------------------------------------------------------------

@patch(_MOCK_GEMINI_CALL)
def test_multiple_files_source_ref_handling(mock_extract_rfq) -> None:
    """Multiple files: match keeps file, PDF preserves page, image clears page, unmatched cleared with warning."""
    sample = copy.deepcopy(_load_expected_sample("rfq_email.json"))
    assert len(sample["line_items"]) >= 3

    # Item 0 matches a PDF -> keep file, keep page
    sample["line_items"][0]["source_ref"]["file"] = "drawing.pdf"
    sample["line_items"][0]["source_ref"]["page"] = 3

    # Item 1 matches an image -> keep file, clear page to None
    sample["line_items"][1]["source_ref"]["file"] = "photo.jpeg"
    sample["line_items"][1]["source_ref"]["page"] = 1

    # Item 2 has a file that does not match any supplied filename -> clear file & page, add warning
    sample["line_items"][2]["source_ref"]["file"] = "unknown_document.docx"
    sample["line_items"][2]["source_ref"]["page"] = 2

    mock_extract_rfq.return_value = sample

    files = [
        (b"%PDF...", "application/pdf", "drawing.pdf"),
        (b"\xFF\xD8...", "image/jpeg", "photo.jpeg"),
    ]
    result = extract_rfq_from_files(files)

    assert result.status == "needs_review"
    assert result.rfq is not None

    items = result.rfq.line_items
    assert items[0].source_ref.file == "drawing.pdf"
    assert items[0].source_ref.page == 3

    assert items[1].source_ref.file == "photo.jpeg"
    assert items[1].source_ref.page is None

    assert items[2].source_ref.file is None
    assert items[2].source_ref.page is None
    assert any("does not match any supplied filename" in w for w in result.warnings)


@patch(_MOCK_GEMINI_CALL)
def test_extractor_failed_extraction_logging_includes_class_and_code_without_message(
    mock_extract_rfq, caplog
) -> None:
    """Assert final 'Extraction failed' line includes last attempt's class and status code,

    without leaking the exception message.
    """
    secret_message = "SECRET_DOCUMENT_SNIPPET_DO_NOT_LOG"
    extraction_err = ExtractionError(
        "Extraction failed after 3 attempts. See server logs for details.",
        cause_class="ClientError",
        http_code=429,
        status="RESOURCE_EXHAUSTED",
    )
    mock_extract_rfq.side_effect = extraction_err

    with caplog.at_level(logging.DEBUG):
        result = extract_rfq_from_files([(b"content", "text/plain", "rfq.txt")])

    assert result.status == "failed"
    assert result.rfq is None

    # Assert final 'Extraction failed' line contains class name and status code
    assert "Extraction failed with ExtractionError" in caplog.text
    assert "ClientError" in caplog.text
    assert "429" in caplog.text
    assert "RESOURCE_EXHAUSTED" in caplog.text

    # Assert secret message text is NOT logged
    assert secret_message not in caplog.text
