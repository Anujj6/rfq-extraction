"""File utilities: content-based MIME type detection, filename sanitization."""

from __future__ import annotations

import os
import re

_SUPPORTED_MIME_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/webp",
    "text/plain",
}


def detect_mime_type(data: bytes) -> str | None:
    """Detect MIME type strictly from file content / magic bytes.

    Supported types:
    - application/pdf: starts with b"%PDF"
    - image/png: starts with b"\x89PNG\r\n\x1a\n"
    - image/jpeg: starts with b"\xff\xd8\xff"
    - image/webp: starts with b"RIFF" and bytes 8..12 are b"WEBP"
    - text/plain: valid UTF-8 without NUL bytes (b"\x00")

    Returns None for any unsupported or corrupt file type.
    """
    if not data:
        return None

    if data.startswith(b"%PDF"):
        return "application/pdf"

    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"

    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"

    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"

    # Plain text check: strictly no NUL bytes and must decode cleanly as UTF-8
    if b"\x00" not in data:
        try:
            data.decode("utf-8")
            return "text/plain"
        except UnicodeDecodeError:
            return None

    return None


def sanitize_filename(filename: str) -> str:
    """Sanitize original filename: base name only, strip path parts and control characters.

    Caps at 255 characters.
    """
    if not filename or not isinstance(filename, str):
        return "unnamed_file"

    # Normalize backslashes and extract base name only
    normalized = filename.replace("\\", "/")
    base = os.path.basename(normalized)

    # Strip control characters (ASCII < 32 or == 127)
    cleaned = "".join(c for c in base if ord(c) >= 32 and ord(c) != 127).strip()

    # Reject empty or relative dot tokens
    if not cleaned or cleaned in {".", ".."}:
        cleaned = "unnamed_file"

    return cleaned[:255]
