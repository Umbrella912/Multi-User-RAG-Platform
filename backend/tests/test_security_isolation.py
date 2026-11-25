"""Security test suite: authentication hardening and data isolation.

Two layers:

* **Unit tests (no DB)** — storage path traversal, filename sanitization,
  JWT tampering / expiry / type confusion, rate-limiter behavior, and
  retrieval-level user scoping.
* **API tests (DB-backed)** — cross-user access to knowledge bases,
  documents and chat sessions is always denied, unauthenticated requests
  are rejected, upload validation is enforced, and rate limits trigger.

The DB-backed tests auto-skip when no test database is reachable (the same
pattern as the other integration tests).
"""
from __future__ import annotations

import io
import time
from uuid import UUID

import jwt
import pytest
from httpx import AsyncClient

from app.core.config import settings
from app.core.rate_limit import InMemoryRateLimiter
from app.core.security import (
    REFRESH_TOKEN_TYPE,
    create_access_token,
    create_refresh_token,
    decode_token,
)
from app.services.storage import LocalStorageBackend, safe_filename

API = "/api"


# ---------------------------------------------------------------------------
# Unit: storage path traversal & filename sanitization
# ---------------------------------------------------------------------------


def test_safe_filename_strips_path_components() -> None:
    """Filenames must never carry path separators or traversal sequences."""
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("../..") == "file"
    assert "/" not in safe_filename("a/b/c.pdf")
    assert "../" not in safe_filename("../evil.txt")
    assert safe_filename("weird/name with spaces.pdf") == "name_with_spaces.pdf"


def test_local_storage_rejects_path_traversal(tmp_path, monkeypatch) -> None:
    """Storage keys that resolve outside the root must be rejected."""
    monkeypatch.setattr(
        "app.core.config.settings.STORAGE_LOCAL_PATH", str(tmp_path)
    )
    backend = LocalStorageBackend()
    with pytest.raises(ValueError, match="invalid storage key"):
        backend._path("../secret.txt")  # noqa: SLF001 - security-critical
    with pytest.raises(ValueError, match="invalid storage key"):
        backend.read("../../etc/passwd")  # noqa: SLF001
    # A well-formed key inside the root works.
    key = backend.save(UUID(int=1), UUID(int=2), "ok.txt", b"data")
    assert "../" not in key
    assert backend.read(key) == b"data"


# ---------------------------------------------------------------------------
# Unit: JWT hardening
# ---------------------------------------------------------------------------


def test_jwt_access_roundtrip() -> None:
    payload = decode_token(create_access_token(UUID(int=5)), expected_type="access")
    assert payload["sub"] == str(UUID(int=5))
    assert payload["type"] == "access"


def test_jwt_rejects_tampered_signature() -> None:
    token = create_access_token(UUID(int=5))
    header, payload, signature = token.split(".")
    tampered = f"{header}.{payload}.{signature[:-2]}XX"
    with pytest.raises(jwt.PyJWTError):
        decode_token(tampered)


def test_jwt_rejects_expired_token() -> None:
    from datetime import timedelta

    from app.core.security import _create_token

    expired = _create_token(UUID(int=5), "access", timedelta(seconds=-60))
    with pytest.raises(jwt.PyJWTError):
        decode_token(expired)


def test_jwt_rejects_wrong_token_type() -> None:
    refresh = create_refresh_token(UUID(int=5))
    # A refresh token must never be accepted as an access token.
    with pytest.raises(jwt.PyJWTError):
        decode_token(refresh, expected_type="access")
    # The refresh type claim is honored when expected.
    payload = decode_token(refresh, expected_type=REFRESH_TOKEN_TYPE)
    assert payload["type"] == REFRESH_TOKEN_TYPE


def test_jwt_rejects_algorithm_confusion() -> None:
    """Tokens signed with a different HS algorithm must be rejected."""
    token = jwt.encode(
        {"sub": str(UUID(int=5)), "type": "access"},
        settings.JWT_SECRET,
        algorithm="HS512",
    )
    with pytest.raises(jwt.PyJWTError):
        decode_token(token)


def test_jwt_rejects_garbage() -> None:
    with pytest.raises(jwt.PyJWTError):
        decode_token("not-a-jwt")


# ---------------------------------------------------------------------------
# Unit: rate limiter
# ---------------------------------------------------------------------------


def test_rate_limiter_blocks_after_limit() -> None:
    limiter = InMemoryRateLimiter()
    assert all(limiter.is_allowed("k", limit=3, period=60) for _ in range(3))
    assert limiter.is_allowed("k", limit=3, period=60) is False
    # Reset restores the allowance.
    limiter.reset("k")
    assert limiter.is_allowed("k", limit=3, period=60) is True


def test_rate_limiter_keys_are_isolated() -> None:
    limiter = InMemoryRateLimiter()
    for _ in range(3):
        limiter.is_allowed("user-a", limit=3, period=60)
    assert limiter.is_allowed("user-a", limit=3, period=60) is False
    assert limiter.is_allowed("user-b", limit=3, period=60) is True


def test_rate_limiter_window_expires() -> None:
    limiter = InMemoryRateLimiter()
    assert limiter.is_allowed("k", limit=1, period=0.05) is True
    assert limiter.is_allowed("k", limit=1, period=0.05) is False
    time.sleep(0.06)
    assert limiter.is_allowed("k", limit=1, period=0.05) is True


# ---------------------------------------------------------------------------
# Unit: retrieval-level data isolation
# ---------------------------------------------------------------------------


class _IsolationVectorStore:
    """Records queries and enforces per-user filtering like the real store."""

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], list[dict]] = {}
        self.queries: list[dict] = []
        self.fetches: list[dict] = []

    async def add_chunks(self, kb_id, chunks, embeddings) -> int:
        for chunk in chunks:
            self._data.setdefault((str(kb_id), chunk.user_id), []).append(
                {
                    "id": chunk.id,
                    "text": chunk.text,
                    "metadata": {
                    "file_name": chunk.file_name,
                    "page": chunk.page,
                    "user_id": chunk.user_id,
                },
                    "distance": 0.1,
                }
            )
        return len(chunks)

    async def query(self, kb_id, query_embedding, *, user_id, n_results=20):
        self.queries.append({"kb_id": str(kb_id), "user_id": str(user_id)})
        chunks = self._data.get((str(kb_id), str(user_id)), [])
        return [
            {
                "id": c["id"],
                "text": c["text"],
                "metadata": c["metadata"],
                "distance": c["distance"],
            }
            for c in chunks[:n_results]
        ]

    async def fetch_all(self, kb_id, *, user_id):
        self.fetches.append({"kb_id": str(kb_id), "user_id": str(user_id)})
        return list(self._data.get((str(kb_id), str(user_id)), []))


async def test_retriever_scopes_vector_and_bm25_by_user(monkeypatch) -> None:
    """Vector and BM25 lookups must carry the caller's user id as a filter."""
    from app.services import embeddings, vector_store
    from app.services.retrieval import Retriever

    store = _IsolationVectorStore()
    monkeypatch.setattr(vector_store, "vector_store", store)
    monkeypatch.setattr(
        embeddings, "embed_queries", lambda texts, **kw: [[0.0] for _ in texts]
    )
    monkeypatch.setattr("app.services.retrieval.settings.RERANK_ENABLED", False)

    user_a = UUID(int=1)
    user_b = UUID(int=2)
    kb_id = UUID(int=3)
    from app.services.chunking import Chunk

    for user in (user_a, user_b):
        await store.add_chunks(
            kb_id,
            [
                Chunk(
                    text="user a secret content",
                    chunk_index=0,
                    document_id="d",
                    kb_id=str(kb_id),
                    user_id=str(user),
                    file_name="f.txt",
                    page=1,
                    page_end=1,
                )
            ],
            [[0.0]],
        )

    retriever = Retriever()
    result_a = await retriever.retrieve("secret content", kb_id, user_a)

    # The vector query was scoped to user A.
    assert store.queries[-1]["user_id"] == str(user_a)
    assert store.queries[-1]["kb_id"] == str(kb_id)
    # The BM25 index build was scoped to user A.
    assert store.fetches[-1]["user_id"] == str(user_a)
    # User A only ever sees their own chunks.
    assert all(c.metadata.get("user_id") == str(user_a) for c in result_a.chunks)


# ---------------------------------------------------------------------------
# API tests (DB-backed; skip without a reachable test database)
# ---------------------------------------------------------------------------


async def _register_and_token(client: AsyncClient, email: str) -> tuple[dict, dict[str, str]]:
    body = await client.post(
        f"{API}/auth/register",
        json={"email": email, "password": "secret123", "full_name": "U"},
    )
    assert body.status_code == 201, body.text
    data = body.json()
    return data, {"Authorization": f"Bearer {data['access_token']}"}


@pytest.fixture
def clean_limiter() -> None:
    """Reset the shared rate limiter around each test for determinism."""
    from app.core.rate_limit import limiter

    limiter.reset()
    yield
    limiter.reset()


@pytest.fixture
def fake_storage(monkeypatch) -> None:
    from app.api.routers import documents as documents_router
    from app.services.storage import StorageBackend

    class _Fake(StorageBackend):
        def save(self, user_id, document_id, file_name, data):
            return f"fake/{document_id}/{file_name}"

        def read(self, key):
            return b"fake content"

        def delete(self, key):
            return None

    monkeypatch.setattr(documents_router, "storage", _Fake())


@pytest.fixture
def no_ingest(monkeypatch) -> None:
    from app.api.routers import documents as documents_router

    async def _noop(document_id: UUID, kb_id: UUID) -> None:
        return None

    monkeypatch.setattr(documents_router, "_ingestion_task", _noop)


async def test_cross_user_document_access_denied(
    client: AsyncClient, fake_storage: None, no_ingest: None, clean_limiter: None
) -> None:
    """User B must get 404 on user A's document (existence must not leak)."""
    _, headers_a = await _register_and_token(client, "sec-doc-a@example.com")
    kb = await client.post(
        f"{API}/knowledge-bases", json={"name": "A KB"}, headers=headers_a
    )
    kb_id = UUID(kb.json()["id"])
    upload = await client.post(
        f"{API}/knowledge-bases/{kb_id}/documents",
        files={"file": ("notes.txt", io.BytesIO(b"secret content"), "text/plain")},
        headers=headers_a,
    )
    doc_id = UUID(upload.json()["id"])

    _, headers_b = await _register_and_token(client, "sec-doc-b@example.com")

    fetch = await client.get(f"{API}/documents/{doc_id}", headers=headers_b)
    assert fetch.status_code == 404

    delete = await client.delete(f"{API}/documents/{doc_id}", headers=headers_b)
    assert delete.status_code == 404

    # B cannot list documents inside A's KB.
    listing = await client.get(f"{API}/knowledge-bases/{kb_id}/documents", headers=headers_b)
    assert listing.status_code == 404

    # A can still access it.
    own = await client.get(f"{API}/documents/{doc_id}", headers=headers_a)
    assert own.status_code == 200


async def test_cross_user_chat_session_access_denied(
    client: AsyncClient, clean_limiter: None
) -> None:
    """User B must not read, ask or delete user A's chat session."""
    _, headers_a = await _register_and_token(client, "sec-chat-a@example.com")
    kb = await client.post(
        f"{API}/knowledge-bases", json={"name": "Chat KB"}, headers=headers_a
    )
    kb_id = UUID(kb.json()["id"])
    session = await client.post(
        f"{API}/chat/sessions",
        json={"kb_id": str(kb_id), "title": "A session"},
        headers=headers_a,
    )
    session_id = UUID(session.json()["id"])

    _, headers_b = await _register_and_token(client, "sec-chat-b@example.com")

    messages = await client.get(f"{API}/chat/sessions/{session_id}/messages", headers=headers_b)
    assert messages.status_code == 404

    ask = await client.post(
        f"{API}/chat/sessions/{session_id}/ask-sync",
        json={"question": "hi"},
        headers=headers_b,
    )
    assert ask.status_code == 404

    delete = await client.delete(f"{API}/chat/sessions/{session_id}", headers=headers_b)
    assert delete.status_code == 404

    # A can still use the session.
    own = await client.get(f"{API}/chat/sessions/{session_id}/messages", headers=headers_a)
    assert own.status_code == 200


async def test_unauthenticated_access_rejected(client: AsyncClient) -> None:
    """Protected endpoints must require a valid bearer token."""
    assert (await client.get(f"{API}/knowledge-bases")).status_code == 401
    assert (await client.post(f"{API}/knowledge-bases", json={"name": "x"})).status_code == 401
    assert (await client.get(f"{API}/chat/sessions")).status_code == 401
    bad = await client.get(
        f"{API}/knowledge-bases",
        headers={"Authorization": "Bearer not-a-token"},
    )
    assert bad.status_code == 401


async def test_upload_validation_enforced(
    client: AsyncClient, fake_storage: None, no_ingest: None, clean_limiter: None, monkeypatch
) -> None:
    """Uploads must reject bad extensions, empty files, bad magic and size."""
    from app.api.routers import documents as documents_router

    monkeypatch.setattr(documents_router, "_MAX_BYTES", 64)
    _, headers = await _register_and_token(client, "sec-upload@example.com")
    kb = await client.post(
        f"{API}/knowledge-bases", json={"name": "Upload KB"}, headers=headers
    )
    kb_id = UUID(kb.json()["id"])

    # Unsupported extension.
    bad_ext = await client.post(
        f"{API}/knowledge-bases/{kb_id}/documents",
        files={"file": ("evil.exe", io.BytesIO(b"MZ..."), "application/octet-stream")},
        headers=headers,
    )
    assert bad_ext.status_code == 400

    # Empty file.
    empty = await client.post(
        f"{API}/knowledge-bases/{kb_id}/documents",
        files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
        headers=headers,
    )
    assert empty.status_code == 400

    # Bad magic bytes (fake PDF content).
    bad_magic = await client.post(
        f"{API}/knowledge-bases/{kb_id}/documents",
        files={"file": ("fake.pdf", io.BytesIO(b"not a pdf at all"), "application/pdf")},
        headers=headers,
    )
    assert bad_magic.status_code == 400

    # Oversized file (limit monkeypatched to 64 bytes).
    oversized = await client.post(
        f"{API}/knowledge-bases/{kb_id}/documents",
        files={
            "file": ("big.txt", io.BytesIO(b"x" * 200), "text/plain")
        },
        headers=headers,
    )
    assert oversized.status_code == 413


@pytest.fixture
def fake_pipeline(app) -> None:
    """Override the pipeline dependency with a fake-LLM pipeline."""
    from app.api.dependencies import get_rag_pipeline
    from app.services.llm import GenerationResult, LLMProvider
    from app.services.rag_pipeline import RAGPipeline

    class _FakeLLM(LLMProvider):
        name = "fake"

        async def generate_with_usage(self, messages, **kwargs):
            return GenerationResult(text="answer", usage={"total_tokens": 1})

        async def generate_stream(self, messages, **kwargs):
            yield "answer"

    app.dependency_overrides[get_rag_pipeline] = lambda: RAGPipeline(_FakeLLM())


async def test_chat_rate_limit_returns_429(
    client: AsyncClient, fake_pipeline: None, monkeypatch
) -> None:
    """Asking beyond the per-user chat limit must yield 429.

    The limiter is swapped for a fresh instance so the test is deterministic
    regardless of what other tests requested (the dependency resolves the
    ``limiter`` global at request time).
    """
    from app.core.rate_limit import InMemoryRateLimiter

    monkeypatch.setattr("app.core.rate_limit.limiter", InMemoryRateLimiter())

    _, headers = await _register_and_token(client, "sec-rate@example.com")
    kb = await client.post(
        f"{API}/knowledge-bases", json={"name": "Rate KB"}, headers=headers
    )
    kb_id = UUID(kb.json()["id"])
    session = await client.post(
        f"{API}/chat/sessions",
        json={"kb_id": str(kb_id), "title": "Rate session"},
        headers=headers,
    )
    session_id = UUID(session.json()["id"])

    statuses = []
    for _ in range(settings.RATE_LIMIT_CHAT_PER_MIN + 1):
        resp = await client.post(
            f"{API}/chat/sessions/{session_id}/ask-sync",
            json={"question": "question"},
            headers=headers,
        )
        statuses.append(resp.status_code)

    assert statuses[: settings.RATE_LIMIT_CHAT_PER_MIN] == [200] * settings.RATE_LIMIT_CHAT_PER_MIN
    assert statuses[-1] == 429


async def test_kb_name_collision_conflicts(
    client: AsyncClient, clean_limiter: None
) -> None:
    """Duplicate KB names for the same user are rejected with 409."""
    _, headers = await _register_and_token(client, "sec-dup@example.com")
    first = await client.post(
        f"{API}/knowledge-bases", json={"name": "Duplicate KB"}, headers=headers
    )
    assert first.status_code == 201
    second = await client.post(
        f"{API}/knowledge-bases", json={"name": "Duplicate KB"}, headers=headers
    )
    assert second.status_code == 409
