"""Upload endpoint tests: validation, dedup, listing, status, delete.

The heavy services (storage, embedding, vector store) are mocked so the test
only exercises the router/validation logic. DB-backed: auto-skips without a
reachable Postgres (the same pattern as the other integration tests).
"""
from __future__ import annotations

import io
from uuid import UUID

import pytest
from httpx import AsyncClient

from app.api.routers import documents as documents_router
from app.services.storage import StorageBackend

API = "/api"


class _FakeStorage(StorageBackend):
    """In-memory storage backend capturing saved bytes without touching disk."""

    def __init__(self) -> None:
        self._blobs: dict[str, bytes] = {}

    def save(self, user_id: UUID, document_id: UUID, file_name: str, data: bytes) -> str:
        key = f"fake/{document_id}/{file_name}"
        self._blobs[key] = data
        return key

    def read(self, key: str) -> bytes:
        return self._blobs[key]

    def delete(self, key: str) -> None:
        self._blobs.pop(key, None)


@pytest.fixture
def fake_storage(monkeypatch) -> _FakeStorage:
    """Replace the router's storage backend with an in-memory fake."""
    storage = _FakeStorage()
    monkeypatch.setattr(documents_router, "storage", storage)
    return storage


@pytest.fixture
def no_ingest(monkeypatch) -> None:
    """Prevent the real (heavy) ingestion background task from running."""

    async def _noop(document_id: UUID, kb_id: UUID) -> None:
        return None

    monkeypatch.setattr(documents_router, "_ingestion_task", _noop)


async def _make_kb(client: AsyncClient) -> tuple[str, UUID]:
    """Register a user + create a KB; return (auth_header, kb_id)."""
    reg = await client.post(
        f"{API}/auth/register",
        json={"email": "uploader@example.com", "password": "secret123"},
    )
    token = reg.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    kb = await client.post(
        f"{API}/knowledge-bases", json={"name": "Docs KB"}, headers=headers
    )
    return headers, UUID(kb.json()["id"])


async def test_upload_list_status_delete(
    client: AsyncClient, fake_storage: _FakeStorage, no_ingest
) -> None:
    headers, kb_id = await _make_kb(client)

    # Upload a valid .txt file.
    upload = await client.post(
        f"{API}/knowledge-bases/{kb_id}/documents",
        files={"file": ("notes.txt", io.BytesIO(b"hello world"), "text/plain")},
        headers=headers,
    )
    assert upload.status_code == 201, upload.text
    body = upload.json()
    assert body["status"] == "pending"
    assert body["file_type"] == "txt"
    doc_id = body["id"]

    # List documents.
    listing = await client.get(f"{API}/knowledge-bases/{kb_id}/documents", headers=headers)
    assert listing.status_code == 200
    assert len(listing.json()) == 1

    # Status endpoint.
    status_resp = await client.get(f"{API}/documents/{doc_id}", headers=headers)
    assert status_resp.status_code == 200
    assert status_resp.json()["id"] == doc_id

    # Delete removes it.
    delete = await client.delete(f"{API}/documents/{doc_id}", headers=headers)
    assert delete.status_code == 204
    gone = await client.get(f"{API}/documents/{doc_id}", headers=headers)
    assert gone.status_code == 404


async def test_duplicate_upload_conflicts(
    client: AsyncClient, fake_storage: _FakeStorage, no_ingest
) -> None:
    headers, kb_id = await _make_kb(client)
    data = b"the same content"
    first = await client.post(
        f"{API}/knowledge-bases/{kb_id}/documents",
        files={"file": ("a.txt", io.BytesIO(data), "text/plain")},
        headers=headers,
    )
    assert first.status_code == 201
    # Same content, different name -> still a duplicate (hash check).
    second = await client.post(
        f"{API}/knowledge-bases/{kb_id}/documents",
        files={"file": ("b.txt", io.BytesIO(data), "text/plain")},
        headers=headers,
    )
    assert second.status_code == 409


async def test_upload_rejects_bad_extension(
    client: AsyncClient, fake_storage: _FakeStorage, no_ingest
) -> None:
    headers, kb_id = await _make_kb(client)
    resp = await client.post(
        f"{API}/knowledge-bases/{kb_id}/documents",
        files={"file": ("evil.exe", io.BytesIO(b"MZ"), "application/octet-stream")},
        headers=headers,
    )
    assert resp.status_code == 400


async def test_upload_rejects_mismatched_mime(
    client: AsyncClient, fake_storage: _FakeStorage, no_ingest
) -> None:
    headers, kb_id = await _make_kb(client)
    # Declared as .pdf but content is not a PDF.
    resp = await client.post(
        f"{API}/knowledge-bases/{kb_id}/documents",
        files={"file": ("fake.pdf", io.BytesIO(b"not a pdf"), "application/pdf")},
        headers=headers,
    )
    assert resp.status_code == 400


async def test_upload_rejects_empty_file(
    client: AsyncClient, fake_storage: _FakeStorage, no_ingest
) -> None:
    headers, kb_id = await _make_kb(client)
    resp = await client.post(
        f"{API}/knowledge-bases/{kb_id}/documents",
        files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
        headers=headers,
    )
    assert resp.status_code == 400


async def test_upload_requires_auth(client: AsyncClient) -> None:
    # Auth is enforced before any KB lookup, so a bogus KB id still yields 401.
    resp = await client.post(
        f"{API}/knowledge-bases/00000000-0000-0000-0000-000000000000/documents",
        files={"file": ("x.txt", io.BytesIO(b"data"), "text/plain")},
    )
    assert resp.status_code == 401
