"""Extraction service: coordinates Gemini RFQ extraction and validation.

Enforces source_ref metadata safety, wraps results in ExtractionResult,
and prevents leakage of sensitive documents, extracted text, or API keys
in logs and warning messages.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from app.config import get_settings
from app.prompts import PROMPT_VERSION
from app.schemas import ExtractionResult
from app.services.gemini import ExtractionError, extract_rfq
from app.services.validator import validate_rfq_dict

# ---------------------------------------------------------------------------
# Logging — never log file contents, model output, or API keys
# ---------------------------------------------------------------------------

_log = logging.getLogger(__name__)


def _is_image_mime(mime: str) -> bool:
    """Return True if MIME type represents an image."""
    return mime.startswith("image/") or mime.lower() in {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
        "image/tiff",
        "image/bmp",
    }


def extract_rfq_from_files(
    files: list[tuple[bytes, str, str]],
    request_id: Optional[str] = None,
) -> ExtractionResult:
    """Extract structured RFQ data from one or more uploaded files and validate.

    Parameters
    ----------
    files:
        List of (file_bytes, mime_type, original_filename) tuples.
    request_id:
        Optional request identifier. If not provided, a new UUID4 string is generated.

    Returns
    -------
    ExtractionResult
        Envelope containing schema_version, request_id, status ("ok", "needs_review", "failed"),
        warnings, model_name, prompt_version, and validated rfq object (or None if failed).
    """
    settings = get_settings()
    model_name = settings.GEMINI_MODEL
    req_id = request_id or str(uuid.uuid4())

    if not files:
        _log.warning(
            "Extraction aborted: no files supplied",
            extra={
                "request_id": req_id,
                "status": "failed",
                "line_items_count": 0,
                "warnings_count": 1,
            },
        )
        return ExtractionResult(
            schema_version="1.0",
            request_id=req_id,
            status="failed",
            warnings=["No files supplied for extraction"],
            model_name=model_name,
            prompt_version=PROMPT_VERSION,
            rfq=None,
        )

    # 1. Prepare parts for Gemini API (bytes, mime_type)
    file_parts = [(data, mime) for data, mime, _ in files]

    # 2. Call Gemini extraction pipeline
    try:
        raw_dict = extract_rfq(file_parts, request_id=req_id)
    except ExtractionError as exc:
        cause_class = getattr(exc, "cause_class", None)
        http_code = getattr(exc, "http_code", None)
        error_status = getattr(exc, "status", None)

        if cause_class is None and exc.__cause__:
            cause_class = type(exc.__cause__).__name__
        if http_code is None and exc.__cause__:
            raw_c = getattr(exc.__cause__, "code", getattr(exc.__cause__, "status_code", None))
            if isinstance(raw_c, int):
                http_code = raw_c
            elif raw_c is not None:
                try:
                    http_code = int(raw_c)
                except (ValueError, TypeError):
                    http_code = None
        if error_status is None and exc.__cause__:
            raw_s = getattr(exc.__cause__, "status", None)
            if isinstance(raw_s, str) and raw_s.strip():
                error_status = raw_s.strip()

        cause_parts = []
        if cause_class:
            cause_parts.append(cause_class)
        if http_code is not None:
            cause_parts.append(f"status_code={http_code}")
        if error_status:
            cause_parts.append(f"status={error_status}")
        cause_str = f": {', '.join(cause_parts)}" if cause_parts else ""

        _log.error(
            "Extraction failed with ExtractionError%s",
            cause_str,
            extra={
                "request_id": req_id,
                "status": "failed",
                "line_items_count": 0,
                "warnings_count": 1,
                "last_error_type": cause_class,
                "last_http_code": http_code,
                "last_error_status": error_status,
            },
        )
        return ExtractionResult(
            schema_version="1.0",
            request_id=req_id,
            status="failed",
            warnings=[str(exc)],  # Safe message prepared by ExtractionError
            model_name=model_name,
            prompt_version=PROMPT_VERSION,
            rfq=None,
        )
    except Exception as exc:  # noqa: BLE001
        cause_class = type(exc).__name__
        raw_c = getattr(exc, "code", getattr(exc, "status_code", None))
        http_code = raw_c if isinstance(raw_c, int) else None
        cause_parts = [cause_class]
        if http_code is not None:
            cause_parts.append(f"status_code={http_code}")
        cause_str = f": {', '.join(cause_parts)}"

        _log.error(
            "Extraction failed with unexpected internal error%s",
            cause_str,
            extra={
                "request_id": req_id,
                "status": "failed",
                "line_items_count": 0,
                "warnings_count": 1,
                "last_error_type": cause_class,
                "last_http_code": http_code,
            },
        )
        return ExtractionResult(
            schema_version="1.0",
            request_id=req_id,
            status="failed",
            warnings=["Extraction failed due to an unexpected error. See server logs for details."],
            model_name=model_name,
            prompt_version=PROMPT_VERSION,
            rfq=None,
        )

    # 3. Validate model output
    rfq, status, warnings = validate_rfq_dict(raw_dict)

    # 4. If schema validation failed, return failed result immediately
    if status == "failed" or rfq is None:
        _log.warning(
            "Extraction validation failed",
            extra={
                "request_id": req_id,
                "status": "failed",
                "line_items_count": 0,
                "warnings_count": len(warnings),
            },
        )
        return ExtractionResult(
            schema_version="1.0",
            request_id=req_id,
            status="failed",
            warnings=warnings,
            model_name=model_name,
            prompt_version=PROMPT_VERSION,
            rfq=None,
        )

    # 5. Post-process source_ref.file and source_ref.page
    filename_to_mime = {filename: mime for _, mime, filename in files}

    if len(files) == 1:
        single_filename = files[0][2]
        single_mime = files[0][1]
        for item in rfq.line_items:
            # Overwrite source_ref.file with original filename (never trust model)
            item.source_ref.file = single_filename
            # Adjust source_ref.page based on media type
            if single_mime == "text/plain" or _is_image_mime(single_mime):
                item.source_ref.page = None
            # For application/pdf, keep the model's value
    else:
        for idx, item in enumerate(rfq.line_items):
            model_file = item.source_ref.file
            item_id = item.line_no if item.line_no is not None else idx + 1

            if model_file and model_file in filename_to_mime:
                matched_mime = filename_to_mime[model_file]
                if matched_mime == "text/plain" or _is_image_mime(matched_mime):
                    item.source_ref.page = None
                # For application/pdf, keep the model's value
            else:
                item.source_ref.file = None
                item.source_ref.page = None
                warnings.append(
                    f"line item {item_id}: source_ref.file does not match any supplied filename"
                )
                if status == "ok":
                    status = "needs_review"

    # 6. Log non-sensitive summary
    _log.info(
        "RFQ extraction completed",
        extra={
            "request_id": req_id,
            "status": status,
            "line_items_count": len(rfq.line_items),
            "warnings_count": len(warnings),
        },
    )

    return ExtractionResult(
        schema_version="1.0",
        request_id=req_id,
        status=status,
        warnings=warnings,
        model_name=model_name,
        prompt_version=PROMPT_VERSION,
        rfq=rfq,
    )
