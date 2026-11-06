# RFQ Extraction Microservice

An AI-powered document extraction microservice built with **FastAPI** and **Google Gemini** (`gemini-3.8-flash`). It automatically parses, transcribes, and structures Request for Quotation (RFQ) documents—such as engineering inquiries, procurement emails, scanned purchase requests, and spec sheets—into standardized JSON payloads with strict schema validation.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture & Processing Flow](#architecture--processing-flow)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites & Installation](#prerequisites--installation)
- [Configuration](#configuration)
- [Running the Service](#running-the-service)
- [API Reference](#api-reference)
  - [Authentication & Rate Limiting](#authentication--rate-limiting)
  - [Uniform Error Format](#uniform-error-format)
  - [Endpoints](#endpoints)
    - [`GET /health`](#get-health)
    - [`POST /extract`](#post-extract)
    - [`POST /extract-upload`](#post-extract-upload)
- [Schema & Output Model](#schema--output-model)
- [Developer Tools & Scripts](#developer-tools--scripts)
- [Running Tests](#running-tests)
- [Security & Privacy Design](#security--privacy-design)

---

## Overview

Industrial and commercial RFQs often arrive in non-standardized formats: multi-page PDF purchase requisitions, scanned drawing prints, item specification tables, or free-form vendor inquiry emails. 

This service processes these documents by:
1. Validating inputs strictly via content magic bytes (rejecting corrupted or spoofed files).
2. Leveraging Gemini's multimodal reasoning with dynamic low-latency thinking to transcribe raw engineering specifications, commercial terms, delivery requirements, and line items verbatim without hallucination.
3. Validating the LLM output against strict Pydantic models (`extra="forbid"`) and business rules before returning structured data.

---

## Key Features

- **Dual Ingestion Pipelines**:
  - **Storage-Backed (`POST /extract`)**: Ingests files by key from local storage or S3/object storage (supports up to `MAX_FILE_MB`, default 15 MB).
  - **Direct Streaming Upload (`POST /extract-upload`)**: Direct `multipart/form-data` uploads parsed in-memory with strict chunked size bounding (max 4 MB).
- **Strict File Type Verification**: Validates file types by inspecting magic bytes rather than trusting file extensions or client headers (supports PDF, PNG, JPEG, WEBP, and UTF-8 plain text).
- **Zero Disk Footprint**: File buffers are processed entirely in-memory and passed to Gemini via inline bytes; no customer files or intermediate outputs are written to disk.
- **Enterprise-Grade Security**:
  - Constant-time API key validation via `X-API-Key`.
  - Secrets loaded as Pydantic `SecretStr` (redacted in logs, exceptions, and traces).
  - Strict storage key validation preventing directory traversal (`..`), symlink escapes, control characters, and scheme injections.
  - Uniform, sanitized error responses preventing information leakage.
- **Strict Verification & Status Tagging**:
  - Outputs are marked as `ok`, `needs_review`, or `failed`.
  - Audits line items for missing sources, duplicate line numbers, and unquantified items.
  - Automatically correlates single-file sources and strips irrelevant page numbers for text/images.
- **Rate Limiting**: Built-in IP/API-key rate limiting via `slowapi` (60 req/min for unauthenticated requests, 120 req/min for authenticated callers).

---

## Architecture & Processing Flow

```
[ Client Request ]
       |
       v
+-------------------------------------------------------------+
| FastAPI Application (app.main)                             |
|  - Rate Limiter (SlowAPI)                                   |
|  - Authentication Check (Header: X-API-Key)                 |
+-------------------------------------------------------------+
       |
       +-----------------------------------+
       |                                   |
 [ POST /extract ]               [ POST /extract-upload ]
 (Storage Keys)                  (Multipart Streaming)
       |                                   |
       v                                   v
+-----------------------+         +----------------------------+
| Storage Provider      |         | Streaming Multipart Parser |
| (Local / AWS S3)      |         | (In-memory bounded stream) |
| - Key validation      |         +----------------------------+
| - Size pre-check      |                      |
| - In-memory download  |                      |
+-----------------------+                      |
       |                                       |
       +-------------------+-------------------+
                           |
                           v
            +------------------------------+
            | Content Magic Byte Sniffer   |
            | (PDF, PNG, JPEG, WEBP, TXT)  |
            +------------------------------+
                           |
                           v
            +------------------------------+
            | Gemini Extraction Engine     |
            | - google-genai 2.32.0        |
            | - gemini-3.8-flash           |
            | - Low thinking level         |
            | - Response JSON schema       |
            +------------------------------+
                           |
                           v
            +------------------------------+
            | Validation & Post-processing |
            | - Pydantic schema validation |
            | - Business rules audit       |
            | - Source ref sanitization    |
            +------------------------------+
                           |
                           v
                 [ ExtractionResult ]
```

---

## Tech Stack

- **Runtime & Framework**: Python 3.10+ / FastAPI / Starlette
- **Server**: Uvicorn (ASGI)
- **Validation & Settings**: Pydantic v2 & Pydantic-Settings
- **LLM SDK**: Google GenAI Python SDK (`google-genai==2.32.0`)
- **Cloud Storage**: Boto3 (AWS S3 and S3-compatible providers)
- **Rate Limiting**: SlowAPI
- **Streaming Parser**: Python-Multipart
- **Testing**: Pytest

---

## Project Structure

```
rfq-extraction/
├── app/
│   ├── __init__.py
│   ├── auth.py              # API key verification utilities
│   ├── config.py            # Pydantic environment configuration
│   ├── main.py              # FastAPI application & route definitions
│   ├── prompts.py           # Gemini system instructions & prompt templates
│   ├── schemas.py           # Pydantic schemas (ExtractionResult, RfqExtracted)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── extractor.py     # Pipeline coordinator & source_ref logic
│   │   ├── file_utils.py    # Magic-byte MIME detection & filename sanitization
│   │   ├── gemini.py        # Gemini client integration & retry logic
│   │   ├── storage.py       # Local and S3 storage abstractions
│   │   └── validator.py     # Schema and business integrity validator
│   └── tests/
│       ├── test_api.py            # API endpoint integration tests
│       ├── test_extractor.py      # Extractor service & source_ref tests
│       ├── test_gemini_service.py # Gemini service & error handling tests
│       └── test_schemas.py        # Schema validation tests
├── docs/
│   ├── ai_studio_settings.md # Reference model & thinking configuration
│   ├── rfq_schema.json       # JSON schema passed to Gemini response_schema
│   └── system_instructions.md# RFQ transcription engine prompt guidelines
├── samples/                  # Sample RFQ documents & expected JSON outputs
│   ├── expected/             # Golden expected JSON outputs for testing
│   ├── rfq_email.txt
│   ├── rfq_email2.txt
│   ├── rfq_sample.jpeg
│   └── rfq_sample.pdf
├── scripts/
│   ├── compare_expected.py  # Diff CLI tool against expected ground truth
│   └── smoke_gemini.py      # Live Gemini smoke test utility
├── requirements.txt
├── .env.example
└── README.md
```

---

## Prerequisites & Installation

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- A Google Gemini API Key with access to `gemini-3.8-flash`

### 2. Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd rfq-extraction

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Configuration

Copy `.env.example` to `.env` and fill in the required variables:

```bash
cp .env.example .env
```

### Environment Variables

| Variable | Type | Required | Default | Description |
|---|---|:---:|---|---|
| `GEMINI_API_KEY` | String (Secret) | **Yes** | — | API key for Google Gemini API. |
| `GEMINI_MODEL` | String | **Yes** | `gemini-3.8-flash` | Gemini model identifier (e.g. `gemini-3.8-flash`). |
| `SERVICE_API_KEY` | String (Secret) | **Yes** | — | Shared secret required in the `X-API-Key` request header. |
| `MAX_FILE_MB` | Integer | No | `15` | Maximum allowed size (in megabytes) per file for storage extraction. |
| `STORAGE_PROVIDER` | String | No | `local` | Storage backend: `local` or `s3`. |
| `STORAGE_LOCAL_DIR` | String | No | `./samples` | Directory root for `local` storage provider. |
| `STORAGE_BUCKET` | String | Conditional | `""` | S3 bucket name (required when `STORAGE_PROVIDER=s3`). |
| `S3_ENDPOINT_URL` | String | No | `""` | Custom S3 endpoint URL (e.g., MinIO or Cloudflare R2). |
| `S3_REGION` | String | No | `""` | AWS region for S3 client. |
| `S3_ACCESS_KEY_ID` | String | Conditional | `""` | AWS access key ID (required when `STORAGE_PROVIDER=s3`). |
| `S3_SECRET_ACCESS_KEY` | String (Secret) | Conditional | `""` | AWS secret access key (required when `STORAGE_PROVIDER=s3`). |
| `ENABLE_DOCS` | Boolean | No | `false` | Enable Swagger UI (`/docs`), ReDoc (`/redoc`), and `/openapi.json`. |
| `MEDIA_RESOLUTION` | String | No | `default` | Media token resolution: `default`, `low`, `medium`, or `high`. |

> [!NOTE]
> When `ENABLE_DOCS` is `false` (the default for security hardening), OpenAPI docs and schema endpoints return `404 Not Found`. Set `ENABLE_DOCS=true` during development if interactive API exploration is desired.

---

## Running the Service

### Development Mode

Run with `uvicorn` using automatic reload:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Production Mode

Run with multi-worker support behind an ASGI server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4 --proxy-headers
```

---

## API Reference

### Authentication & Rate Limiting

- **Header**: All protected endpoints require `X-API-Key: <SERVICE_API_KEY>`.
- **Rate Limits**:
  - Unauthenticated requests: `60 requests / minute`.
  - Authenticated requests: `120 requests / minute`.
  - Exceeding the quota returns HTTP `429 Too Many Requests`.

### Uniform Error Format

All error responses strictly follow a uniform shape:

```json
{
  "error": {
    "code": "invalid_request",
    "message": "The request payload is invalid."
  }
}
```

Common status codes:
- `400 Bad Request` (`bad_request`)
- `401 Unauthorized` (`unauthorized`)
- `404 Not Found` (`not_found`)
- `413 Payload Too Large` (`payload_too_large`)
- `415 Unsupported Media Type` (`unsupported_media_type`)
- `422 Unprocessable Entity` (`invalid_request`)
- `429 Rate Limit Exceeded` (`rate_limit_exceeded`)
- `500 Internal Server Error` (`internal_error`)

---

### Endpoints

#### `GET /health`
Liveness probe. Does not require authentication.

**Response (`200 OK`)**:
```json
{
  "status": "ok"
}
```

---

#### `POST /extract`
Extracts structured RFQ data from 1 to 5 files located in the configured storage provider (`local` or `s3`).

**Headers**:
- `Content-Type: application/json`
- `X-API-Key: <SERVICE_API_KEY>`

**Request Body**:
```json
{
  "files": [
    {
      "storage_key": "rfq_sample.pdf",
      "original_filename": "inquiry_march_2025.pdf"
    }
  ]
}
```

**cURL Example**:
```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_service_api_key_here" \
  -d '{
    "files": [
      {
        "storage_key": "rfq_email.txt",
        "original_filename": "rfq_email.txt"
      }
    ]
  }'
```

---

#### `POST /extract-upload`
Directly uploads 1 to 5 files via `multipart/form-data`. Uploads are processed purely in-memory.

**Constraints**:
- Max file size: 4 MB (per file and total request payload).
- Allowed MIME types: PDF, PNG, JPEG, WEBP, and UTF-8 plain text.

**Headers**:
- `X-API-Key: <SERVICE_API_KEY>`

**cURL Example**:
```bash
curl -X POST http://localhost:8000/extract-upload \
  -H "X-API-Key: your_service_api_key_here" \
  -F "files=@samples/rfq_sample.pdf;filename=rfq_sample.pdf"
```

---

## Schema & Output Model

The service returns an `ExtractionResult` envelope wrapping the extracted RFQ details:

```json
{
  "schema_version": "1.0",
  "request_id": "c1f7a4b8-936b-4e17-b732-cfbf6d8fbf9e",
  "status": "ok",
  "warnings": [],
  "model_name": "gemini-3.8-flash",
  "prompt_version": "1.0",
  "rfq": {
    "document_language": "en",
    "client": {
      "company_name": "Apex Engineering Ltd",
      "contact_person": "Rajesh Sharma",
      "email": "rajesh@apexeng.in",
      "phone": "+91 98765 43210",
      "website": "www.apexeng.in",
      "address": "Plot 42, Phase II, Peenya Industrial Area, Bengaluru, Karnataka",
      "gst_number": "29ABCDE1234F1Z5",
      "client_reference_number": "AE/PUR/2025/089"
    },
    "rfq_details": {
      "rfq_number": "RFQ-2025-104",
      "rfq_date": "15-Mar-2025",
      "response_due_date": "25-Mar-2025",
      "project_name": "Cooling Water System Expansion",
      "subject_or_summary": "Inquiry for Industrial Valves and Flanges"
    },
    "delivery": {
      "location": "Apex Plant Site, Peenya, Bengaluru",
      "required_date": "30-Apr-2025",
      "instructions": "Unloading by supplier. Deliver between 9 AM and 4 PM."
    },
    "line_items": [
      {
        "line_no": 1,
        "item_name": "Ball Valve",
        "description": "3-piece design, full bore",
        "part_number": "BV-3P-FB-50",
        "drawing_number": null,
        "brand_or_make": "L&T or Audco",
        "material": "Stainless Steel",
        "grade": "SS316",
        "standard": "ASME B16.34",
        "size_raw": "50 NB (2 inch)",
        "other_specs": "Rating: Class 150, Flanged Ends",
        "certifications_raw": "EN 10204 3.1 Test Certificate required",
        "quantity": 10.0,
        "quantity_raw": "10 Nos.",
        "unit_raw": "Nos.",
        "target_price_raw": null,
        "item_delivery_date": null,
        "remarks": null,
        "source_text": "1. Ball Valve, 50 NB, SS316, Class 150, Qty: 10 Nos. Make: L&T or Audco.",
        "source_ref": {
          "file": "inquiry_march_2025.pdf",
          "page": 1
        }
      }
    ],
    "commercial_terms": {
      "payment_terms": "30 days net from receipt of material and approved invoice.",
      "freight_terms": "FOR Destination, inclusive of freight and transit insurance.",
      "warranty": "18 months from supply or 12 months from commissioning.",
      "taxes": "GST extra at actuals (currently 18%).",
      "other_terms": null
    },
    "attachments": [
      {
        "file_name": "spec_sheet_valves.pdf",
        "type": "spec_sheet",
        "description": "Technical specification sheet for valves"
      }
    ],
    "missing_or_unclear": [],
    "unstructured_notes": "Please acknowledge receipt of this inquiry and confirm participation by return email."
  }
}
```

### Status Meanings

- **`ok`**: All line items extracted, required fields validated, and source references properly mapped.
- **`needs_review`**: Extracted successfully, but warnings were detected (e.g., empty line items, line items with unstated quantities, or ambiguity in document sources).
- **`failed`**: Schema validation failed or the model call errored out. `rfq` will be `null`.

---

## Developer Tools & Scripts

Utility scripts are located in `scripts/` to help evaluate and smoke-test the pipeline:

### 1. Gemini Smoke Test (`scripts/smoke_gemini.py`)
Executes an end-to-end extraction against a sample document in `samples/` using the live Gemini API:

```bash
python scripts/smoke_gemini.py samples/rfq_email.txt
python scripts/smoke_gemini.py samples/rfq_sample.pdf
```

### 2. Compare Against Expected (`scripts/compare_expected.py`)
Performs an extraction and compares the output against a golden expected JSON file under `samples/expected/`:

```bash
# Compare a single run
python scripts/compare_expected.py samples/rfq_email.txt samples/expected/rfq_email.json

# Run multiple passes to test extraction consistency
python scripts/compare_expected.py samples/rfq_sample.pdf samples/expected/rfq_sample.pdf.json --runs 3
```

---

## Running Tests

The test suite contains 77 unit and integration tests covering API endpoints, authentication, path traversal guards, media type detection, schemas, and Gemini service integration.

To run the full test suite:

```bash
# Activate your virtual environment, then:
pytest

# Or directly with the virtualenv Python:
.venv\Scripts\pytest -v
```

---

## Security & Privacy Design

1. **Prompt Injection & Untrusted Data**: Documents are treated as untrusted data. The prompt system instructions explicitly prohibit executing embedded instructions found inside user documents.
2. **Path Traversal & Symlink Protection**:
   - `LocalStorage` validates all keys: rejects `..`, absolute paths, backslashes, URL schemes, and checks resolved file targets against base directories.
   - Scripts and endpoints restrict file paths strictly to designated directories.
3. **Information Disclosure Prevention**:
   - Stack traces and raw internal error details are suppressed from HTTP responses.
   - Pydantic validation errors in the validation service sanitize raw field inputs to prevent customer document contents from leaking into logs or warning strings.
   - API keys and secrets are protected via `SecretStr`.
4. **Denial of Service (DoS) Protections**:
   - Direct upload enforces streaming chunk bounds (4 MB limit) before reading entire payloads into memory.
   - Rate limiting prevents brute force and token exhaustion.
   - Max file count enforced at 5 files per request.
