"""Pydantic request/response schemas for the RFQ extraction API.

All models are built to mirror docs/rfq_schema.json exactly:
- Every field that is nullable in the JSON schema is typed Optional[...] but is
  still *required* (present in the schema's "required" list).  Pydantic treats a
  field as required when it has no default; Optional only widens the type to also
  accept None.
- extra="forbid" is set on every model so an unknown key causes a ValidationError
  rather than being silently ignored.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# Shared base that rejects extra fields everywhere
# ---------------------------------------------------------------------------

class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Leaf / nested models
# ---------------------------------------------------------------------------

class SourceRef(_Base):
    file: Optional[str]
    page: Optional[int]


class Client(_Base):
    company_name: Optional[str]
    contact_person: Optional[str]
    email: Optional[str]
    phone: Optional[str]
    website: Optional[str]
    address: Optional[str]
    gst_number: Optional[str]
    client_reference_number: Optional[str]


class RfqDetails(_Base):
    rfq_number: Optional[str]
    rfq_date: Optional[str]
    response_due_date: Optional[str]
    project_name: Optional[str]
    subject_or_summary: Optional[str]


class Delivery(_Base):
    location: Optional[str]
    required_date: Optional[str]
    instructions: Optional[str]


class LineItem(_Base):
    line_no: Optional[int]
    item_name: Optional[str]
    description: Optional[str]
    part_number: Optional[str]
    drawing_number: Optional[str]
    brand_or_make: Optional[str]
    material: Optional[str]
    grade: Optional[str]
    standard: Optional[str]
    size_raw: Optional[str]
    other_specs: Optional[str]
    certifications_raw: Optional[str]
    quantity: Optional[float]
    quantity_raw: Optional[str]
    unit_raw: Optional[str]
    target_price_raw: Optional[str]
    item_delivery_date: Optional[str]
    remarks: Optional[str]
    # source_text is NOT nullable in the schema (no "nullable": true)
    source_text: str
    source_ref: SourceRef


class CommercialTerms(_Base):
    payment_terms: Optional[str]
    freight_terms: Optional[str]
    warranty: Optional[str]
    taxes: Optional[str]
    other_terms: Optional[str]


class Attachment(_Base):
    file_name: Optional[str]
    # "type" is an enum and NOT nullable in the schema
    type: Literal["drawing", "spec_sheet", "image", "other"]
    description: Optional[str]


# ---------------------------------------------------------------------------
# Top-level extracted RFQ
# ---------------------------------------------------------------------------

class RfqExtracted(_Base):
    """Direct representation of docs/rfq_schema.json."""

    document_language: Optional[str]
    client: Client
    rfq_details: RfqDetails
    delivery: Delivery
    line_items: list[LineItem]
    commercial_terms: CommercialTerms
    attachments: list[Attachment]
    # missing_or_unclear items are plain strings (not nullable individually)
    missing_or_unclear: list[str]
    unstructured_notes: Optional[str]


# ---------------------------------------------------------------------------
# Wrapper returned by the extraction pipeline
# ---------------------------------------------------------------------------

class ExtractionResult(_Base):
    """Envelope returned by the extraction service."""

    schema_version: Literal["1.0"] = "1.0"
    request_id: str
    status: Literal["ok", "needs_review", "failed"]
    warnings: list[str]
    model_name: str
    prompt_version: str
    # rfq is None when status is "failed"
    rfq: Optional[RfqExtracted]


# ---------------------------------------------------------------------------
# Schema loader
# ---------------------------------------------------------------------------

_SCHEMA_PATH = Path(__file__).parent.parent / "docs" / "rfq_schema.json"


def load_rfq_schema() -> dict:
    """Load and return the raw JSON schema from docs/rfq_schema.json.

    The returned dict can be passed directly to Gemini as the
    ``response_schema`` so the model output is constrained to the same
    structure validated by the Pydantic models above.
    """
    with _SCHEMA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)
