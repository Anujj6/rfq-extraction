"""Validator service for Gemini RFQ extraction output.

Performs schema validation and business integrity checks on raw model output.
Never includes input values, raw ValidationError text, or extracted content
in warnings, errors, or logs.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import ValidationError

from app.schemas import RfqExtracted


def validate_rfq_dict(
    raw_dict: Any,
) -> tuple[RfqExtracted | None, Literal["ok", "needs_review", "failed"], list[str]]:
    """Validate a raw dictionary returned by the Gemini extraction pipeline.

    Checks:
    1. Parses into Pydantic RfqExtracted model. If not, this is a hard failure
       (status="failed", rfq=None). Warning text is built ONLY from the error
       location (field path) and error type. Input values, raw ValidationError
       messages, and extracted text are NEVER included.
    2. Every line item has a non-empty source_text.
    3. If quantity is non-null, quantity_raw must be non-null.
    4. Duplicate line_no values are flagged.
    5. If line_items is empty, add a warning.
    6. If any line item has a null quantity and missing_or_unclear is empty,
       add a warning.

    Returns
    -------
    tuple[Optional[RfqExtracted], Literal["ok", "needs_review", "failed"], list[str]]
        - (rfq, status, warnings)
        - On check 1 failure: (None, "failed", [warnings...])
        - On checks 2-6 warnings: (rfq, "needs_review", [warnings...])
        - When all pass: (rfq, "ok", [])
    """
    if not isinstance(raw_dict, dict):
        return None, "failed", ["schema validation error at root: not_a_dictionary"]

    # -----------------------------------------------------------------------
    # Check 1: Pydantic parsing (hard failure if invalid)
    # -----------------------------------------------------------------------
    try:
        rfq = RfqExtracted.model_validate(raw_dict)
    except ValidationError as exc:
        warnings: list[str] = []
        for err in exc.errors():
            loc_parts = err.get("loc", ())
            loc_str = " -> ".join(str(p) for p in loc_parts) if loc_parts else "root"
            err_type = str(err.get("type", "validation_error"))
            # STRICT PRIVACY: NEVER include err['input'], err['msg'], or raw text.
            warnings.append(f"schema validation error at {loc_str}: {err_type}")
        return None, "failed", warnings

    warnings = []

    # -----------------------------------------------------------------------
    # Check 2: Every line item has a non-empty source_text
    # -----------------------------------------------------------------------
    for idx, item in enumerate(rfq.line_items):
        item_id = item.line_no if item.line_no is not None else idx + 1
        if not item.source_text or not item.source_text.strip():
            warnings.append(f"line item {item_id}: source_text is empty")

    # -----------------------------------------------------------------------
    # Check 3: If quantity is non-null, quantity_raw must be non-null
    # -----------------------------------------------------------------------
    for idx, item in enumerate(rfq.line_items):
        item_id = item.line_no if item.line_no is not None else idx + 1
        if item.quantity is not None:
            if item.quantity_raw is None or not str(item.quantity_raw).strip():
                warnings.append(f"line item {item_id}: quantity is present but quantity_raw is missing")

    # -----------------------------------------------------------------------
    # Check 4: Duplicate line_no values are flagged
    # -----------------------------------------------------------------------
    seen_line_nos: set[int] = set()
    duplicates: set[int] = set()
    for item in rfq.line_items:
        if item.line_no is not None:
            if item.line_no in seen_line_nos:
                duplicates.add(item.line_no)
            else:
                seen_line_nos.add(item.line_no)

    for dup in sorted(duplicates):
        warnings.append(f"duplicate line_no: {dup}")

    # -----------------------------------------------------------------------
    # Check 5: If line_items is empty, add a warning
    # -----------------------------------------------------------------------
    if len(rfq.line_items) == 0:
        warnings.append("line_items is empty")

    # -----------------------------------------------------------------------
    # Check 6: If any line item has a null quantity and missing_or_unclear is empty,
    # add a warning
    # -----------------------------------------------------------------------
    has_null_quantity = any(item.quantity is None for item in rfq.line_items)
    if has_null_quantity and len(rfq.missing_or_unclear) == 0:
        warnings.append("line item has null quantity but missing_or_unclear is empty")

    # -----------------------------------------------------------------------
    # Final status determination
    # -----------------------------------------------------------------------
    status: Literal["ok", "needs_review", "failed"] = "needs_review" if warnings else "ok"
    return rfq, status, warnings
