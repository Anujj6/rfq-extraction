"""FastAPI application entry point for the RFQ extraction microservice.

Exposes:
- GET  /health          - Liveness probe
- POST /extract         - RFQ extraction pipeline endpoint (storage-backed)
- POST /extract-upload  - Direct multipart upload RFQ extraction endpoint
"""

import hashlib
import logging
import secrets
from typing import Any, Optional

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

try:
    from python_multipart import MultipartParser
    from python_multipart.multipart import parse_options_header
except ImportError:
    from multipart import MultipartParser  # type: ignore[no-redef]
    from multipart.multipart import parse_options_header  # type: ignore[no-redef]

from app.config import get_settings
from app.schemas import ExtractionResult
from app.services.extractor import extract_rfq_from_files
from app.services.file_utils import detect_mime_type, sanitize_filename
from app.services.storage import (
    InvalidStorageKeyError,
    ObjectNotFoundError,
    get_storage,
)

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate limiter configuration
# ---------------------------------------------------------------------------

def get_api_key_hash(request: Request) -> str:
    """Hash the incoming API key to use as the rate limiting identifier."""
    api_key = request.headers.get("x-api-key", "")
    if not api_key:
        return "unauthenticated"
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def is_authenticated(request: Request) -> bool:
    """Return True if the request carries a valid SERVICE_API_KEY."""
    api_key = request.headers.get("x-api-key", "")
    settings = get_settings()
    return bool(
        api_key
        and secrets.compare_digest(api_key, settings.SERVICE_API_KEY.get_secret_value())
    )


def is_unauthenticated(request: Request) -> bool:
    """Return True if the request is not authenticated."""
    return not is_authenticated(request)


limiter = Limiter(key_func=get_api_key_hash)

# ---------------------------------------------------------------------------
# Application initialization
# ---------------------------------------------------------------------------

settings = get_settings()

app = FastAPI(
    title="RFQ Extraction Service",
    description="Extracts structured data from RFQ documents using Gemini.",
    version="0.1.0",
    docs_url="/docs" if settings.ENABLE_DOCS else None,
    redoc_url="/redoc" if settings.ENABLE_DOCS else None,
    openapi_url="/openapi.json" if settings.ENABLE_DOCS else None,
)
app.state.limiter = limiter


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------

class FileReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    storage_key: str = Field(..., min_length=1, max_length=512)
    original_filename: str = Field(..., min_length=1, max_length=512)


class ExtractRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    files: list[FileReference] = Field(..., min_length=1, max_length=5)


# ---------------------------------------------------------------------------
# Error Handlers: Uniform Error Shape {"error": {"code": "...", "message": "..."}}
# ---------------------------------------------------------------------------

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "code": "rate_limit_exceeded",
                "message": "Rate limit exceeded. Please try again later.",
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "invalid_request",
                "message": "The request payload is invalid.",
            }
        },
    )


@app.exception_handler(StarletteHTTPException)
@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request, exc: StarletteHTTPException | HTTPException
) -> JSONResponse:
    headers = getattr(exc, "headers", None)
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail, headers=headers)

    code_map = {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        405: "method_not_allowed",
        413: "payload_too_large",
        415: "unsupported_media_type",
        422: "invalid_request",
        429: "rate_limit_exceeded",
        500: "internal_error",
    }
    error_code = code_map.get(exc.status_code, "error")
    message = str(exc.detail) if exc.detail else "An error occurred."
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": error_code, "message": message}},
        headers=headers,
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Generic 500 error: never leak stack trace or internal details
    _log.error("Unhandled exception on request: %s", type(exc).__name__)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_error",
                "message": "An internal error occurred. Please try again later.",
            }
        },
    )


# ---------------------------------------------------------------------------
# Authentication Dependency
# ---------------------------------------------------------------------------

def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> str:
    """Verify request API key against SERVICE_API_KEY with constant-time comparison."""
    current_settings = get_settings()
    expected_key = current_settings.SERVICE_API_KEY.get_secret_value()
    if not x_api_key or not secrets.compare_digest(x_api_key, expected_key):
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "code": "unauthorized",
                    "message": "Invalid or missing API key.",
                }
            },
        )
    return x_api_key


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    """Liveness probe — returns 200 OK with ``{"status": "ok"}``."""
    return {"status": "ok"}


@app.post("/extract", response_model=ExtractionResult, tags=["extraction"])
@limiter.limit("60/minute", key_func=get_api_key_hash, exempt_when=is_unauthenticated)
@limiter.limit("120/minute", key_func=get_api_key_hash, exempt_when=is_authenticated)
def extract(
    request: Request,
    payload: ExtractRequest = Body(...),
    _auth: str = Depends(verify_api_key),
) -> ExtractionResult:
    """Extract structured data from files referenced in object storage.

    Runs in a worker threadpool (synchronous def) to prevent blocking the event loop.
    """
    current_settings = get_settings()
    storage = get_storage()

    # 1. Enforce file count bounds
    if not (1 <= len(payload.files) <= 5):
        raise HTTPException(
            status_code=422,
            detail={
                "error": {
                    "code": "invalid_request",
                    "message": "Number of files must be between 1 and 5.",
                }
            },
        )

    # 2. Check file sizes BEFORE downloading
    max_file_bytes = current_settings.MAX_FILE_MB * 1024 * 1024
    total_max_bytes = 3 * max_file_bytes
    total_size = 0

    for file_ref in payload.files:
        try:
            size = storage.get_size(file_ref.storage_key)
        except InvalidStorageKeyError:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": {
                        "code": "invalid_request",
                        "message": "Invalid storage key provided.",
                    }
                },
            ) from None
        except ObjectNotFoundError:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": {
                        "code": "not_found",
                        "message": "The requested object was not found in storage.",
                    }
                },
            ) from None

        if size > max_file_bytes:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": {
                        "code": "payload_too_large",
                        "message": "File size exceeds the allowed limit.",
                    }
                },
            )

        total_size += size
        if total_size > total_max_bytes:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": {
                        "code": "payload_too_large",
                        "message": "Total size of all files exceeds the allowed limit.",
                    }
                },
            )

    # 3. Download and inspect media in-memory (never write to disk)
    files_to_extract: list[tuple[bytes, str, str]] = []

    for file_ref in payload.files:
        data = storage.download(file_ref.storage_key)
        detected_mime = detect_mime_type(data)
        if not detected_mime:
            raise HTTPException(
                status_code=415,
                detail={
                    "error": {
                        "code": "unsupported_media_type",
                        "message": (
                            "Unsupported file type. Only PDF, PNG, JPEG, WEBP, "
                            "and UTF-8 plain text are supported."
                        ),
                    }
                },
            )

        clean_filename = sanitize_filename(file_ref.original_filename)
        files_to_extract.append((data, detected_mime, clean_filename))

    # 4. Invoke extraction pipeline
    result = extract_rfq_from_files(files_to_extract)

    # 5. Log non-sensitive summary
    _log.info(
        "Extract request completed",
        extra={
            "request_id": result.request_id,
            "status": result.status,
            "num_files": len(payload.files),
            "num_line_items": len(result.rfq.line_items) if result.rfq else 0,
            "num_warnings": len(result.warnings),
        },
    )

    return result


# ---------------------------------------------------------------------------
# Direct Multipart Upload Extraction (POST /extract-upload)
# ---------------------------------------------------------------------------

MAX_UPLOAD_BYTES = 4 * 1024 * 1024  # 4 MB limit (per-file and total)


async def _parse_multipart_files(request: Request) -> list[tuple[bytes, str]]:
    """Parse multipart/form-data upload strictly in-memory with size bounds enforced while reading."""
    # 1. Enforce total Content-Length check first before reading body stream
    content_length_header = request.headers.get("content-length")
    if content_length_header is not None:
        try:
            content_length = int(content_length_header)
            if content_length > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail={
                        "error": {
                            "code": "payload_too_large",
                            "message": "Total size of all files exceeds the allowed limit.",
                        }
                    },
                )
        except ValueError:
            pass

    # 2. Validate multipart/form-data content type and extract boundary
    content_type_header = request.headers.get("content-type", "")
    content_type, params = parse_options_header(content_type_header)
    if content_type != b"multipart/form-data" or b"boundary" not in params:
        raise HTTPException(
            status_code=422,
            detail={
                "error": {
                    "code": "invalid_request",
                    "message": "Expected multipart/form-data request with boundary.",
                }
            },
        )
    boundary = params[b"boundary"]

    parsed_files: list[tuple[bytes, str]] = []
    total_files_size = 0
    cur_headers: dict[bytes, bytes] = {}
    h_field: list[bytes] = []
    h_value: list[bytes] = []
    cur_file: dict[str, Any] | None = None

    def on_part_begin() -> None:
        nonlocal cur_headers, h_field, h_value, cur_file
        cur_headers = {}
        h_field = []
        h_value = []
        cur_file = None

    def on_header_field(data: bytes, start: int, end: int) -> None:
        h_field.append(data[start:end])

    def on_header_value(data: bytes, start: int, end: int) -> None:
        h_value.append(data[start:end])

    def on_header_end() -> None:
        name = b"".join(h_field).lower()
        val = b"".join(h_value)
        cur_headers[name] = val
        h_field.clear()
        h_value.clear()

    def on_headers_finished() -> None:
        nonlocal cur_file
        cd = cur_headers.get(b"content-disposition", b"")
        disp, p = parse_options_header(cd)
        field_name = p.get(b"name", b"").decode("utf-8", "replace")
        if field_name == "files":
            filename_raw = p.get(b"filename")
            filename = (
                filename_raw.decode("utf-8", "replace")
                if filename_raw is not None
                else "unnamed_file"
            )
            cur_file = {"filename": filename, "data": bytearray()}

    def on_part_data(data: bytes, start: int, end: int) -> None:
        nonlocal total_files_size
        if cur_file is not None:
            chunk = data[start:end]
            if len(cur_file["data"]) + len(chunk) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail={
                        "error": {
                            "code": "payload_too_large",
                            "message": "File size exceeds the allowed limit.",
                        }
                    },
                )
            total_files_size += len(chunk)
            if total_files_size > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail={
                        "error": {
                            "code": "payload_too_large",
                            "message": "Total size of all files exceeds the allowed limit.",
                        }
                    },
                )
            cur_file["data"].extend(chunk)

    def on_part_end() -> None:
        nonlocal cur_file
        if cur_file is not None:
            parsed_files.append((bytes(cur_file["data"]), cur_file["filename"]))
            cur_file = None

    parser = MultipartParser(
        boundary,
        {
            "on_part_begin": on_part_begin,
            "on_header_field": on_header_field,
            "on_header_value": on_header_value,
            "on_header_end": on_header_end,
            "on_headers_finished": on_headers_finished,
            "on_part_data": on_part_data,
            "on_part_end": on_part_end,
        },
    )

    total_stream_bytes = 0
    try:
        async for chunk in request.stream():
            total_stream_bytes += len(chunk)
            if total_stream_bytes > MAX_UPLOAD_BYTES + 65536:
                raise HTTPException(
                    status_code=413,
                    detail={
                        "error": {
                            "code": "payload_too_large",
                            "message": "Total size of all files exceeds the allowed limit.",
                        }
                    },
                )
            parser.write(chunk)
        parser.finalize()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": {
                    "code": "invalid_request",
                    "message": "Failed to parse multipart request body.",
                }
            },
        ) from exc

    return parsed_files


@app.post("/extract-upload", response_model=ExtractionResult, tags=["extraction"])
@limiter.limit("60/minute", key_func=get_api_key_hash, exempt_when=is_unauthenticated)
@limiter.limit("120/minute", key_func=get_api_key_hash, exempt_when=is_authenticated)
async def extract_upload(
    request: Request,
    _auth: str = Depends(verify_api_key),
) -> ExtractionResult:
    """Extract structured data from direct multipart/form-data upload.

    Enforces 4 MB limit (per-file and total), validates media types by magic bytes,
    and runs extraction in threadpool in-memory without disk writes.
    """
    raw_files = await _parse_multipart_files(request)

    # 1. Enforce file count bounds (1 to 5 files in field "files")
    if not (1 <= len(raw_files) <= 5):
        raise HTTPException(
            status_code=422,
            detail={
                "error": {
                    "code": "invalid_request",
                    "message": "Number of files must be between 1 and 5.",
                }
            },
        )

    # 2. Inspect media in-memory (never write to disk, never trust client Content-Type or extension)
    files_to_extract: list[tuple[bytes, str, str]] = []
    for data, raw_filename in raw_files:
        clean_filename = sanitize_filename(raw_filename)
        detected_mime = detect_mime_type(data)
        if not detected_mime or (
            clean_filename.lower().endswith(".pdf") and detected_mime != "application/pdf"
        ):
            raise HTTPException(
                status_code=415,
                detail={
                    "error": {
                        "code": "unsupported_media_type",
                        "message": (
                            "Unsupported file type. Only PDF, PNG, JPEG, WEBP, "
                            "and UTF-8 plain text are supported."
                        ),
                    }
                },
            )

        files_to_extract.append((data, detected_mime, clean_filename))

    # 3. Invoke extraction pipeline in worker threadpool
    result = await run_in_threadpool(extract_rfq_from_files, files_to_extract)

    # 4. Log non-sensitive summary
    _log.info(
        "Extract upload request completed",
        extra={
            "request_id": result.request_id,
            "status": result.status,
            "num_files": len(files_to_extract),
            "num_line_items": len(result.rfq.line_items) if result.rfq else 0,
            "num_warnings": len(result.warnings),
        },
    )

    return result
