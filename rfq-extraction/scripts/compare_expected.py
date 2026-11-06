#!/usr/bin/env python
"""Manual developer tool to compare extracted RFQ data against expected JSON.

Usage
-----
    python scripts/compare_expected.py <path under samples/> <path to expected JSON under samples/expected/>
    python scripts/compare_expected.py samples/rfq_email.txt samples/expected/rfq_email.json
    python scripts/compare_expected.py samples/rfq_sample.pdf samples/expected/rfq_sample.pdf.json --runs 3

Rules enforced:
- Only paths that resolve inside samples/ are accepted (external paths and traversal rejected).
- Never writes extracted content to files or logs.
- Never prints the API key.
"""

from __future__ import annotations

import argparse
import copy
import json
import mimetypes
import sys
import textwrap
import uuid
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_PROJECT_ROOT))

_SAMPLES_DIR = (_PROJECT_ROOT / "samples").resolve()


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------

def _validate_path(raw: str, must_be_json: bool = False) -> Path:
    """Validate that path resolves strictly inside samples/ and exists."""
    candidate = (_PROJECT_ROOT / raw).resolve()
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
    if must_be_json and candidate.suffix.lower() != ".json":
        sys.exit(f"ERROR: Expected a .json file, got '{candidate.name}'")
    return candidate


def _detect_mime(path: Path) -> str:
    """Detect MIME type for sample input."""
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        if path.suffix.lower() in {".txt", ".text"}:
            mime = "text/plain"
        else:
            sys.exit(
                f"ERROR: Cannot determine MIME type for '{path.name}'. "
                "Pass only .txt, .pdf, or image files."
            )
    return mime


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

_IGNORED_LINE_ITEM_FIELDS = {"source_ref"}


def _diff_dict(
    expected: dict[str, Any],
    actual: dict[str, Any],
    ignored_keys: set[str] | None = None,
) -> list[tuple[str, Any, Any]]:
    """Return list of (field_name, expected_val, actual_val) differences."""
    ignored = ignored_keys or set()
    diffs: list[tuple[str, Any, Any]] = []
    all_keys = sorted(set(expected.keys()) | set(actual.keys()))
    for k in all_keys:
        if k in ignored:
            continue
        exp_v = expected.get(k)
        act_v = actual.get(k)
        if exp_v != act_v:
            diffs.append((k, exp_v, act_v))
    return diffs


def _print_diff_section(
    title: str,
    diffs: list[tuple[str, Any, Any]],
    indent: str = "  ",
) -> None:
    """Print field-by-field differences in a grouped section."""
    print(f"\n[{title}]")
    if not diffs:
        print(f"{indent}✅ No differences (all fields match)")
        return
    for field_name, exp_val, act_val in diffs:
        print(f"{indent}• {field_name}:")
        print(f"{indent}    Expected: {exp_val!r}")
        print(f"{indent}    Actual  : {act_val!r}")


def _compare_line_items(
    expected_items: list[dict[str, Any]],
    actual_items: list[dict[str, Any]],
) -> None:
    """Match line items by line_no and print field-by-field differences."""
    print("\n[line_items]")
    exp_by_no = {item.get("line_no"): item for item in expected_items if "line_no" in item}
    act_by_no = {item.get("line_no"): item for item in actual_items if "line_no" in item}

    all_line_nos = sorted(set(exp_by_no.keys()) | set(act_by_no.keys()))
    if not all_line_nos:
        print("  (No line items present in expected or actual)")
        return

    had_diff = False
    for line_no in all_line_nos:
        exp_item = exp_by_no.get(line_no)
        act_item = act_by_no.get(line_no)

        if exp_item is None:
            had_diff = True
            print(f"  Line Item #{line_no}:")
            print("    • Present in Actual extraction, but NOT in Expected sample")
            continue
        if act_item is None:
            had_diff = True
            print(f"  Line Item #{line_no}:")
            print("    • Present in Expected sample, but NOT in Actual extraction")
            continue

        item_diffs: list[tuple[str, Any, Any]] = []
        all_fields = sorted(set(exp_item.keys()) | set(act_item.keys()))
        for field in all_fields:
            if field in _IGNORED_LINE_ITEM_FIELDS:
                continue
            ev = exp_item.get(field)
            av = act_item.get(field)
            if ev != av:
                item_diffs.append((field, ev, av))

        # Check source_ref.text (ignore file and page)
        exp_ref_text = (exp_item.get("source_ref") or {}).get("text")
        act_ref_text = (act_item.get("source_ref") or {}).get("text")
        if exp_ref_text != act_ref_text:
            item_diffs.append(("source_ref.text", exp_ref_text, act_ref_text))

        if item_diffs:
            had_diff = True
            print(f"  Line Item #{line_no}:")
            for field, ev, av in item_diffs:
                print(f"    • {field}:")
                print(f"        Expected: {ev!r}")
                print(f"        Actual  : {av!r}")

    if not had_diff:
        print(f"  ✅ No differences across all {len(all_line_nos)} line items")


def _compare_attachments(
    expected_att: list[dict[str, Any]],
    actual_att: list[dict[str, Any]],
) -> None:
    """Compare attachments list."""
    print("\n[attachments]")
    if expected_att == actual_att:
        print(f"  ✅ No differences ({len(expected_att)} attachments match)")
        return

    print(f"  Expected count: {len(expected_att)} | Actual count: {len(actual_att)}")
    max_len = max(len(expected_att), len(actual_att))
    for i in range(max_len):
        ev = expected_att[i] if i < len(expected_att) else None
        av = actual_att[i] if i < len(actual_att) else None
        if ev != av:
            print(f"  Attachment #{i + 1}:")
            print(f"    Expected: {ev!r}")
            print(f"    Actual  : {av!r}")


def _print_side_by_side_missing(
    expected_missing: list[str],
    actual_missing: list[str],
    col_width: int = 42,
) -> None:
    """Print expected and actual missing_or_unclear lists side-by-side."""
    print("\n[missing_or_unclear (side by side)]")
    header_exp = "EXPECTED"
    header_act = "ACTUAL"
    sep = " | "
    print(f"  {header_exp:<{col_width}}{sep}{header_act:<{col_width}}")
    print(f"  {'-' * col_width}{sep}{'-' * col_width}")

    max_items = max(len(expected_missing), len(actual_missing))
    if max_items == 0:
        print("  (Both lists are empty)")
        return

    for i in range(max_items):
        item_exp = f"{i + 1}. {expected_missing[i]}" if i < len(expected_missing) else ""
        item_act = f"{i + 1}. {actual_missing[i]}" if i < len(actual_missing) else ""

        wrapped_exp = textwrap.wrap(item_exp, width=col_width) or [""]
        wrapped_act = textwrap.wrap(item_act, width=col_width) or [""]

        sub_rows = max(len(wrapped_exp), len(wrapped_act))
        for r in range(sub_rows):
            line_l = wrapped_exp[r] if r < len(wrapped_exp) else ""
            line_r = wrapped_act[r] if r < len(wrapped_act) else ""
            print(f"  {line_l:<{col_width}}{sep}{line_r:<{col_width}}")
        if i < max_items - 1:
            print(f"  {'.' * col_width}{sep}{'.' * col_width}")


# ---------------------------------------------------------------------------
# Multi-run stability analysis
# ---------------------------------------------------------------------------

def _flatten_for_stability(data: dict[str, Any]) -> dict[str, Any]:
    """Flatten fields of interest into path -> value mapping for stability check."""
    flat: dict[str, Any] = {}

    for sec in ["client", "rfq_details", "delivery", "commercial_terms"]:
        obj = data.get(sec) or {}
        for k, v in obj.items():
            flat[f"{sec}.{k}"] = v

    flat["document_language"] = data.get("document_language")

    for item in data.get("line_items") or []:
        lno = item.get("line_no")
        for k, v in item.items():
            if k == "source_ref":
                ref_text = (v or {}).get("text")
                flat[f"line_items[{lno}].source_ref.text"] = ref_text
            elif k not in _IGNORED_LINE_ITEM_FIELDS:
                flat[f"line_items[{lno}].{k}"] = v

    for idx, att in enumerate(data.get("attachments") or []):
        for k, v in att.items():
            flat[f"attachments[{idx}].{k}"] = v

    return flat


def _report_stability(runs_data: list[dict[str, Any]]) -> None:
    """Compare fields across runs and print stability summary."""
    n = len(runs_data)
    print(f"\n{'=' * 78}")
    print(f"STABILITY REPORT ACROSS {n} RUNS")
    print(f"{'=' * 78}")

    flattened = [_flatten_for_stability(d) for d in runs_data]
    all_keys = sorted({k for f in flattened for k in f.keys()})

    differing_fields: list[str] = []
    for k in all_keys:
        vals = [f.get(k) for f in flattened]
        # Check if all runs produced identical values for this field
        if any(v != vals[0] for v in vals[1:]):
            differing_fields.append(k)

    if not differing_fields:
        print(f"🎉 100% Stable: All {len(all_keys)} fields were identical across all {n} runs.")
        return

    stable_count = len(all_keys) - len(differing_fields)
    pct = (stable_count / len(all_keys)) * 100
    print(f"Stability: {stable_count}/{len(all_keys)} fields stable ({pct:.1f}%)\n")
    print(f"Differences observed across {n} runs ({len(differing_fields)} fields varied):")
    for k in differing_fields:
        print(f"\n  • {k}:")
        for run_idx, f in enumerate(flattened, start=1):
            print(f"      Run {run_idx}: {f.get(k)!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Gemini RFQ extraction against expected JSON sample."
    )
    parser.add_argument(
        "sample",
        help="Path to sample file under samples/ (e.g. samples/rfq_email.txt)",
    )
    parser.add_argument(
        "expected",
        help="Path to expected JSON file under samples/expected/ (e.g. samples/expected/rfq_email.json)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of extraction runs to test output stability (default: 1)",
    )
    args = parser.parse_args()

    if args.runs < 1:
        sys.exit("ERROR: --runs must be >= 1")

    # 1. Enforce path restrictions strictly inside samples/
    sample_path = _validate_path(args.sample)
    expected_path = _validate_path(args.expected, must_be_json=True)
    mime_type = _detect_mime(sample_path)

    # 2. Read expected JSON
    try:
        expected_dict = json.loads(expected_path.read_text(encoding="utf-8"))
    except Exception as exc:
        sys.exit(f"ERROR: Failed to parse expected JSON file: {exc}")

    # 3. Read sample file bytes
    file_bytes = sample_path.read_bytes()

    # 4. Lazy import of service and schema
    from app.services.gemini import extract_rfq
    from app.schemas import RfqExtracted
    from pydantic import ValidationError

    runs_data: list[dict[str, Any]] = []

    print(f"Sample   : {sample_path}")
    print(f"Expected : {expected_path}")
    print(f"MIME     : {mime_type}")
    print(f"Size     : {len(file_bytes):,} bytes")
    print(f"Runs     : {args.runs}")
    print("-" * 78)

    for run_idx in range(1, args.runs + 1):
        req_id = str(uuid.uuid4())
        prefix = f"[Run {run_idx}/{args.runs}]" if args.runs > 1 else "[Extraction]"
        print(f"{prefix} Calling Gemini API (request_id={req_id}) ...")

        raw_dict = extract_rfq([(file_bytes, mime_type)], request_id=req_id)

        try:
            validated = RfqExtracted.model_validate(raw_dict)
            print(f"{prefix} ✅ Schema validation passed")
        except ValidationError as exc:
            print(f"{prefix} ❌ Schema validation failed: {exc}")
            sys.exit(1)

        runs_data.append(validated.model_dump())

    # 5. Field-by-field comparison against Expected JSON (using Run 1)
    run1 = runs_data[0]
    print(f"\n{'=' * 78}")
    print("FIELD-BY-FIELD COMPARISON (Run 1 vs Expected)")
    print(f"{'=' * 78}")

    # Top-level language
    lang_diff = _diff_dict(
        {"document_language": expected_dict.get("document_language")},
        {"document_language": run1.get("document_language")},
    )
    if lang_diff:
        _print_diff_section("document_language", lang_diff)
    else:
        print("\n[document_language]\n  ✅ Matches expected ('" + str(run1.get("document_language")) + "')")

    # Grouped sections
    for section_name in ["client", "rfq_details", "delivery", "commercial_terms"]:
        exp_sec = expected_dict.get(section_name) or {}
        act_sec = run1.get(section_name) or {}
        diffs = _diff_dict(exp_sec, act_sec)
        _print_diff_section(section_name, diffs)

    # Line items
    _compare_line_items(
        expected_dict.get("line_items") or [],
        run1.get("line_items") or [],
    )

    # Attachments
    _compare_attachments(
        expected_dict.get("attachments") or [],
        run1.get("attachments") or [],
    )

    # Missing or unclear (printed side-by-side)
    _print_side_by_side_missing(
        expected_dict.get("missing_or_unclear") or [],
        run1.get("missing_or_unclear") or [],
    )

    # 6. Stability report across runs if --runs > 1
    if args.runs > 1:
        _report_stability(runs_data)


if __name__ == "__main__":
    main()
