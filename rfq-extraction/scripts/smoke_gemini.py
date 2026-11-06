#!/usr/bin/env python
"""Manual smoke test for the Gemini extraction pipeline.

Usage
-----
Run from the project root with the venv activated:

    python scripts/smoke_gemini.py samples/rfq_email.txt
    python scripts/smoke_gemini.py samples/rfq_sample.pdf
    python scripts/smoke_gemini.py samples/rfq_sample.jpeg

Rules enforced by this script
------------------------------
- Only paths that resolve inside the samples/ directory are accepted.
  Any attempt to pass a path outside samples/ (e.g. absolute paths,
  path-traversal sequences such as ../../etc/passwd) is refused.
- No real API calls are made unless you explicitly run this script.
- The result is validated with Pydantic — any schema mismatch is reported.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve project root and ensure local imports work regardless of CWD
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_PROJECT_ROOT))

_SAMPLES_DIR = (_PROJECT_ROOT / "samples").resolve()


def _validate_path(raw: str) -> Path:
    """Return an absolute Path inside samples/ or abort with a clear message."""
    candidate = (_PROJECT_ROOT / raw).resolve()
    # Ensure the resolved path is strictly inside samples/
    try:
        candidate.relative_to(_SAMPLES_DIR)
    except ValueError:
        sys.exit(
            f"ERROR: Path '{raw}' resolves outside the samples/ directory.\n"
            f"  Resolved: {candidate}\n"
            f"  Allowed:  {_SAMPLES_DIR}/<anything>\n"
            "Only files under samples/ are accepted."
        )
    if not candidate.exists():
        sys.exit(f"ERROR: File not found: {candidate}")
    if not candidate.is_file():
        sys.exit(f"ERROR: Not a regular file: {candidate}")
    return candidate


def _detect_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        # Fallback for .txt files not caught by mimetypes on some systems
        if path.suffix.lower() in {".txt", ".text"}:
            mime = "text/plain"
        else:
            sys.exit(
                f"ERROR: Cannot determine MIME type for '{path.name}'. "
                "Pass only .txt, .pdf, or image files."
            )
    return mime


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test the Gemini RFQ extraction pipeline with one real API call."
    )
    parser.add_argument(
        "file",
        help="Path to a file under samples/ (e.g. samples/rfq_email.txt)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Path safety check
    # ------------------------------------------------------------------
    file_path = _validate_path(args.file)
    mime_type = _detect_mime(file_path)
    print(f"File   : {file_path}")
    print(f"MIME   : {mime_type}")

    # ------------------------------------------------------------------
    # Lazy imports (after path check so we fail fast on bad paths)
    # ------------------------------------------------------------------
    from app.services.gemini import extract_rfq
    from app.schemas import RfqExtracted
    from pydantic import ValidationError

    # ------------------------------------------------------------------
    # Read file
    # ------------------------------------------------------------------
    file_bytes = file_path.read_bytes()
    print(f"Size   : {len(file_bytes):,} bytes")

    # ------------------------------------------------------------------
    # Call Gemini (one real API call)
    # ------------------------------------------------------------------
    import uuid
    request_id = str(uuid.uuid4())
    print(f"Request: {request_id}")
    print("Calling Gemini API … (this may take several seconds)")

    raw_dict = extract_rfq(
        file_parts=[(file_bytes, mime_type)],
        request_id=request_id,
    )

    # ------------------------------------------------------------------
    # Validate with Pydantic
    # ------------------------------------------------------------------
    try:
        extracted = RfqExtracted.model_validate(raw_dict)
        validation_status = "✅  PASSED"
    except ValidationError as exc:
        validation_status = "❌  FAILED"
        errors = exc.errors(include_url=False)
        print(f"\nValidation result : {validation_status}")
        print(f"  {len(errors)} error(s):")
        for e in errors:
            print(f"    • loc={e['loc']}  type={e['type']}  msg={e['msg']}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Print summary (no raw model output — just counts and status)
    # ------------------------------------------------------------------
    line_items_count = len(extracted.line_items)
    attachments_count = len(extracted.attachments)
    missing_count = len(extracted.missing_or_unclear)

    print(f"\nValidation result : {validation_status}")
    print(f"  Line items        : {line_items_count}")
    print(f"  Attachments       : {attachments_count}")
    print(f"  Missing/unclear   : {missing_count}")
    print(f"  Document language : {extracted.document_language}")
    print(f"  Client            : {extracted.client.company_name}")
    print(f"  RFQ number        : {extracted.rfq_details.rfq_number}")


if __name__ == "__main__":
    main()
