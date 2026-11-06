"""Storage abstraction layer: LocalStorage and S3Storage.

Provides unified get_size and download operations with strict storage key
validation, symlink containment checks, and zero leakage of bucket names
or internal paths.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from app.config import get_settings


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class StorageError(Exception):
    """Base exception for storage errors."""


class InvalidStorageKeyError(StorageError):
    """Raised when a storage key violates security or formatting rules."""


class ObjectNotFoundError(StorageError):
    """Raised when the requested object does not exist in storage."""


# ---------------------------------------------------------------------------
# Key validation
# ---------------------------------------------------------------------------

_URL_SCHEME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def validate_storage_key(key: str) -> None:
    """Validate storage key against path traversal, control chars, schemes, and length.

    Rejects:
    - Empty keys or non-strings
    - Keys longer than 512 characters
    - Keys containing '..'
    - Keys starting with '/'
    - Keys containing backslashes ('\\')
    - Keys containing control characters (ASCII < 32 or == 127)
    - Keys containing URL schemes (e.g. 'http:', 'https:', 's3:', 'file:')
    """
    if not key or not isinstance(key, str):
        raise InvalidStorageKeyError("Storage key cannot be empty")

    if len(key) > 512:
        raise InvalidStorageKeyError("Storage key exceeds 512 characters")

    if ".." in key:
        raise InvalidStorageKeyError("Storage key cannot contain '..'")

    if key.startswith("/"):
        raise InvalidStorageKeyError("Storage key cannot start with '/'")

    if "\\" in key:
        raise InvalidStorageKeyError("Storage key cannot contain backslashes")

    if any(ord(c) < 32 or ord(c) == 127 for c in key):
        raise InvalidStorageKeyError("Storage key cannot contain control characters")

    if _URL_SCHEME_PATTERN.match(key):
        raise InvalidStorageKeyError("Storage key cannot contain a URL scheme")


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class Storage(ABC):
    """Abstract storage interface for object retrieval."""

    @abstractmethod
    def get_size(self, key: str) -> int:
        """Return the size in bytes of the object at key without downloading it."""

    @abstractmethod
    def download(self, key: str) -> bytes:
        """Download and return the bytes of the object at key."""


# ---------------------------------------------------------------------------
# Local Storage
# ---------------------------------------------------------------------------

class LocalStorage(Storage):
    """Storage implementation reading from a local filesystem directory."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir).resolve()

    def get_size(self, key: str) -> int:
        validate_storage_key(key)
        target = (self.base_dir / key).resolve()
        if not target.is_relative_to(self.base_dir):
            raise InvalidStorageKeyError("Storage key escapes allowed base directory")
        if not target.exists() or not target.is_file():
            raise ObjectNotFoundError("Object not found in storage")
        return target.stat().st_size

    def download(self, key: str) -> bytes:
        validate_storage_key(key)
        target = (self.base_dir / key).resolve()
        if not target.is_relative_to(self.base_dir):
            raise InvalidStorageKeyError("Storage key escapes allowed base directory")
        if not target.exists() or not target.is_file():
            raise ObjectNotFoundError("Object not found in storage")
        return target.read_bytes()


# ---------------------------------------------------------------------------
# S3 Storage
# ---------------------------------------------------------------------------

class S3Storage(Storage):
    """Storage implementation using S3-compatible object storage via boto3."""

    def __init__(
        self,
        bucket: str,
        endpoint_url: Optional[str] = None,
        region_name: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
    ):
        self.bucket = bucket
        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url or None,
            region_name=region_name or None,
            aws_access_key_id=access_key_id or None,
            aws_secret_access_key=secret_access_key or None,
        )

    def get_size(self, key: str) -> int:
        validate_storage_key(key)
        try:
            resp = self.s3.head_object(Bucket=self.bucket, Key=key)
            return int(resp["ContentLength"])
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in ("404", "NoSuchKey", "NotFound"):
                raise ObjectNotFoundError("Object not found in storage") from None
            raise StorageError("Storage retrieval failed") from None

    def download(self, key: str) -> bytes:
        validate_storage_key(key)
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=key)
            return resp["Body"].read()
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in ("404", "NoSuchKey", "NotFound"):
                raise ObjectNotFoundError("Object not found in storage") from None
            raise StorageError("Storage download failed") from None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_storage() -> Storage:
    """Return configured Storage instance based on STORAGE_PROVIDER setting."""
    settings = get_settings()
    if settings.STORAGE_PROVIDER == "local":
        return LocalStorage(settings.STORAGE_LOCAL_DIR)
    if settings.STORAGE_PROVIDER == "s3":
        return S3Storage(
            bucket=settings.STORAGE_BUCKET,
            endpoint_url=settings.S3_ENDPOINT_URL or None,
            region_name=settings.S3_REGION or None,
            access_key_id=settings.S3_ACCESS_KEY_ID or None,
            secret_access_key=settings.S3_SECRET_ACCESS_KEY.get_secret_value()
            if settings.S3_SECRET_ACCESS_KEY
            else None,
        )
    raise ValueError(f"Unsupported STORAGE_PROVIDER: {settings.STORAGE_PROVIDER}")
