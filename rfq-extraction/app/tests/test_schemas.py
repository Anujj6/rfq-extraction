"""Unit tests for app/schemas.py.

Test strategy
=============
1. validate_samples   – Validate every file in samples/expected/*.json against
                        RfqExtracted.  If any file fails, report the exact
                        mismatch and mark the test as failed (do NOT modify the
                        sample or loosen the model).
2. missing_required   – Remove a required key from a valid payload; expect a
                        ValidationError.
3. extra_field        – Add an unknown key to a valid payload; expect a
                        ValidationError (extra="forbid").
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

# Make sure the project root is importable regardless of how pytest is invoked.
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.schemas import RfqExtracted, load_rfq_schema

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EXPECTED_DIR = Path(__file__).parent.parent.parent / "samples" / "expected"
SCHEMA_PATH  = Path(__file__).parent.parent.parent / "docs" / "rfq_schema.json"

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _all_expected_files() -> list[Path]:
    return sorted(EXPECTED_DIR.glob("*.json"))


# ---------------------------------------------------------------------------
# 1. Validate every expected sample
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sample_path", _all_expected_files(), ids=lambda p: p.name)
def test_validate_sample(sample_path: Path) -> None:
    """Each sample must parse without a ValidationError."""
    data = _load(sample_path)
    try:
        RfqExtracted.model_validate(data)
    except ValidationError as exc:
        # Surface a clear, structured failure message so the developer can
        # decide whether the schema or the sample is wrong.
        errors = exc.errors(include_url=False)
        details = "\n".join(
            f"  • loc={e['loc']}  type={e['type']}  msg={e['msg']}"
            for e in errors
        )
        pytest.fail(
            f"Validation failed for {sample_path.name}:\n{details}",
            pytrace=False,
        )


# ---------------------------------------------------------------------------
# 2. Missing required key must fail
# ---------------------------------------------------------------------------

def _first_valid_sample() -> dict:
    """Return the first available expected sample as a Python dict."""
    files = _all_expected_files()
    assert files, "No expected sample files found in samples/expected/"
    return _load(files[0])


def test_missing_required_key_fails() -> None:
    """Removing a required top-level key must raise ValidationError."""
    data = _first_valid_sample()
    # 'document_language' is required at the top level
    del data["document_language"]
    with pytest.raises(ValidationError) as exc_info:
        RfqExtracted.model_validate(data)
    errors = exc_info.value.errors(include_url=False)
    locs = [e["loc"] for e in errors]
    assert any("document_language" in loc for loc in locs), (
        f"Expected 'document_language' in error locs, got: {locs}"
    )


def test_missing_nested_required_key_fails() -> None:
    """Removing a required key inside a nested object must raise ValidationError."""
    data = _first_valid_sample()
    # 'email' is required inside client (even though it is Optional/nullable)
    del data["client"]["email"]
    with pytest.raises(ValidationError) as exc_info:
        RfqExtracted.model_validate(data)
    errors = exc_info.value.errors(include_url=False)
    locs = [e["loc"] for e in errors]
    assert any("email" in loc for loc in locs), (
        f"Expected 'email' in error locs, got: {locs}"
    )


def test_missing_line_item_required_key_fails() -> None:
    """Removing a required key from a line_item must raise ValidationError."""
    data = copy.deepcopy(_first_valid_sample())
    if not data.get("line_items"):
        pytest.skip("Sample has no line items")
    del data["line_items"][0]["source_text"]
    with pytest.raises(ValidationError) as exc_info:
        RfqExtracted.model_validate(data)
    errors = exc_info.value.errors(include_url=False)
    locs = [e["loc"] for e in errors]
    assert any("source_text" in loc for loc in locs), (
        f"Expected 'source_text' in error locs, got: {locs}"
    )


# ---------------------------------------------------------------------------
# 3. Extra / unknown key must fail (extra="forbid")
# ---------------------------------------------------------------------------

def test_extra_top_level_key_fails() -> None:
    """An unknown top-level key must raise ValidationError."""
    data = _first_valid_sample()
    data["_unknown_field"] = "oops"
    with pytest.raises(ValidationError) as exc_info:
        RfqExtracted.model_validate(data)
    errors = exc_info.value.errors(include_url=False)
    types = [e["type"] for e in errors]
    assert any("extra_forbidden" in t for t in types), (
        f"Expected extra_forbidden error, got types: {types}"
    )


def test_extra_nested_key_fails() -> None:
    """An unknown key inside a nested object must raise ValidationError."""
    data = _first_valid_sample()
    data["client"]["_sneaky"] = "injected"
    with pytest.raises(ValidationError) as exc_info:
        RfqExtracted.model_validate(data)
    errors = exc_info.value.errors(include_url=False)
    types = [e["type"] for e in errors]
    assert any("extra_forbidden" in t for t in types), (
        f"Expected extra_forbidden error in nested model, got types: {types}"
    )


# ---------------------------------------------------------------------------
# 4. load_rfq_schema smoke-test
# ---------------------------------------------------------------------------

def test_load_rfq_schema_returns_dict() -> None:
    schema = load_rfq_schema()
    assert isinstance(schema, dict), "load_rfq_schema() must return a dict"
    assert schema.get("type") == "object", (
        "Root schema type should be 'object', got: " + repr(schema.get("type"))
    )
    assert "properties" in schema, "Schema must have a 'properties' key"


def test_load_rfq_schema_matches_file() -> None:
    """Loaded schema must equal the raw JSON file contents."""
    schema = load_rfq_schema()
    raw = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema == raw, "load_rfq_schema() must return identical content to the file"
