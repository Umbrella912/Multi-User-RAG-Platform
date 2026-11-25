"""Storage abstraction for uploaded files.

Defines a storage-backend interface with a local-disk implementation and an
S3-ready implementation. The router/ingestion code talks only to
:class:`StorageBackend` so swapping ``local`` -> ``s3`` requires no call-site
changes (set ``STORAGE_BACKEND=s3``).
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from app.core.config import settings

if TYPE_CHECKING:
    from botocore.client import BaseClient

# Characters that are unsafe across filesystems / object stores.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str) -> str:
    """Return a storage-safe version of ``name`` (keeps extension, strips the rest).

    Sanitizes path separators and unsafe characters, and falls back to
    ``"file"`` when the stem sanitizes to empty or to only dots/dashes/dots
    (e.g. ``".."``), so the result is always a plain, safe filename.
    """
    suffix = Path(name).suffix.lower()
    stem = _UNSAFE.sub("_", Path(name).stem)[:80]
    if not stem or set(stem) <= set("._-"):
        stem = "file"
    return f"{stem}{suffix}"


class StorageBackend(ABC):
    """Interface for saving/reading/deleting uploaded blobs."""

    @abstractmethod
    def save(self, user_id: UUID, document_id: UUID, file_name: str, data: bytes) -> str:
        """Persist ``data`` and return the storage key used to reference it."""
        raise NotImplementedError

    @abstractmethod
    def read(self, key: str) -> bytes:
        """Return the bytes stored under ``key``."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove the blob stored under ``key`` (no error if absent)."""
        raise NotImplementedError


class LocalStorageBackend(StorageBackend):
    """Store blobs on the local filesystem under ``STORAGE_LOCAL_PATH``."""

    def _path(self, key: str) -> Path:
        """Resolve a storage key to an absolute, contained filesystem path."""
        root = Path(settings.STORAGE_LOCAL_PATH).resolve()
        target = (root / key).resolve()
        # Prevent path traversal outside the storage root.
        if not str(target).startswith(str(root)):
            raise ValueError("invalid storage key")
        return target

    def save(self, user_id: UUID, document_id: UUID, file_name: str, data: bytes) -> str:
        key = f"{user_id}/{document_id}/{safe_filename(file_name)}"
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return key

    def read(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def delete(self, key: str) -> None:
        target = self._path(key)
        if target.exists():
            target.unlink()


class S3StorageBackend(StorageBackend):
    """Store blobs in an S3 (or S3-compatible) bucket via ``boto3``."""

    def __init__(self) -> None:
        self._bucket = settings.S3_BUCKET
        self._prefix = settings.S3_PREFIX.strip("/")
        self._client: BaseClient = self._build_client()

    def _build_client(self) -> BaseClient:
        """Build a boto3 S3 client from settings (lazy import)."""
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "S3 storage requires the 'boto3' package; "
                "install the project extras (see requirements.txt)"
            ) from exc
        kwargs = {
            "region_name": settings.S3_REGION or None,
            "aws_access_key_id": settings.S3_ACCESS_KEY or None,
            "aws_secret_access_key": settings.S3_SECRET_KEY or None,
        }
        if settings.S3_ENDPOINT_URL:
            kwargs["endpoint_url"] = settings.S3_ENDPOINT_URL
        return boto3.client("s3", **kwargs)

    def _key(self, user_id: UUID, document_id: UUID, file_name: str) -> str:
        """Build the object key: ``{prefix}/{user_id}/{document_id}/{name}``."""
        parts = [p for p in (self._prefix, str(user_id), str(document_id)) if p]
        parts.append(safe_filename(file_name))
        return "/".join(parts)

    def save(self, user_id: UUID, document_id: UUID, file_name: str, data: bytes) -> str:
        key = self._key(user_id, document_id, file_name)
        self._client.put_object(
            Bucket=self._bucket, Key=key, Body=data, ContentType="application/octet-stream"
        )
        return key

    def read(self, key: str) -> bytes:
        response = self._client.get_object(Bucket=self._bucket, Key=key)
        return response["Body"].read()

    def delete(self, key: str) -> None:
        try:
            self._client.delete_object(Bucket=self._bucket, Key=key)
        except Exception:
            # Best-effort delete: a missing object is not an error for us.
            pass


def _build_storage() -> StorageBackend:
    """Construct the storage backend selected by ``STORAGE_BACKEND``."""
    if settings.STORAGE_BACKEND == "s3":
        return S3StorageBackend()
    return LocalStorageBackend()


# Module-level singleton, built from the configured backend.
storage: StorageBackend = _build_storage()
