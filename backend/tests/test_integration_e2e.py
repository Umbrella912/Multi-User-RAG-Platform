"""End-to-end integration tests against a real PostgreSQL test database.

These tests exercise the full user journey through the real API routes and
the real ingestion/retrieval pipeline:

    register -> create KB -> upload document -> ingest -> ask -> persisted messages

Heavy external services are replaced with deterministic fakes (in-memory
vector store, canned embeddings, fake LLM, in-memory storage) so the tests
run fast and offline; only PostgreSQL is real (auto-skipped when no test
database is reachable, like the other DB-backed tests).
"""
from __future__ import annotations

import io
import re
from typing import Any
from uuid import UUID

import pytest
from httpx import AsyncClient

from app.api.dependencies import get_rag_pipeline
from app.api.routers import documents as documents_router
from app.services import embeddings, vector_store
from app.services.llm import GenerationResult, LLMProvider
from app.services.rag_pipeline import RAGPipeline
from app.services.storage import StorageBackend

API = "/api"


class _FakeVectorStore:
    """In-memory Chroma stand-in; ranks chunks by query-token overlap.

    The fake embeddings record the last query text, so the store can score
    chunks by keyword overlap without a real embedding model.
    """

    def __init__(self) -> None:
        self._chunks: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self._last_query: str = ""

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def set_query(self, query: str) -> None:
        """Record the query text used by the next :meth:`query` call."""
        self._last_query = query

    async def add_chunks(self, kb_id: UUID | str, chunks: list, embeddings: list) -> int:
        key = (str(kb_id), chunks[0].user_id)
        bucket = self._chunks.setdefault(key, [])
        for chunk in chunks:
            bucket = [c for c in bucket if c["id"] != chunk.id]  # upsert semantics
            bucket.append(
                {
                    "id": chunk.id,
                    "text": chunk.text,
                    "metadata": {
                        "file_name": chunk.file_name,
                        "page": chunk.page,
                        "page_end": chunk.page_end,
                        "doc_id": chunk.document_id,
                        "kb_id": chunk.kb_id,
                        "user_id": chunk.user_id,
                    },
                }
            )
        self._chunks[key] = bucket
        return len(chunks)

    async def query(
        self,
        kb_id: UUID | str,
        query_embedding: list[float],
        *,
        user_id: UUID | str,
        n_results: int = 20,
    ) -> list[dict[str, Any]]:
        del query_embedding  # ranking uses token overlap, not the vector
        chunks = self._chunks.get((str(kb_id), str(user_id)), [])
        query_tokens = self._tokens(self._last_query)
        scored: list[tuple[dict[str, Any], float]] = []
        for chunk in chunks:
            overlap = query_tokens & self._tokens(chunk["text"])
            score = len(overlap) / len(query_tokens) if query_tokens else 0.0
            scored.append((chunk, score))
        scored.sort(key=lambda item: -item[1])
        return [
            {
                "id": chunk["id"],
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "distance": 1.0 - score,
            }
            for chunk, score in scored[:n_results]
        ]

    async def fetch_all(self, kb_id: UUID | str, *, user_id: UUID | str) -> list[dict[str, Any]]:
        return list(self._chunks.get((str(kb_id), str(user_id)), []))

    async def delete_by_document(self, kb_id: UUID | str, document_id: UUID | str) -> None:
        for (kb, _user), bucket in list(self._chunks.items()):
            if kb == str(kb_id):
                self._chunks[(kb, _user)] = [
                    c for c in bucket if c["metadata"].get("doc_id") != str(document_id)
                ]


class _FakeLLM(LLMProvider):
    """Canned LLM that cites the first retrieved chunk deterministically."""

    name = "fake"

    async def generate_with_usage(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> GenerationResult:
        return GenerationResult(
            text="The document states [notes.txt, p.1].",
            usage={"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        )

    async def generate_stream(self, messages: list[dict[str, str]], **kwargs: Any):
        for token in "The document states [notes.txt, p.1].":
            yield token
        self.last_usage = {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}


class _FakeStorage(StorageBackend):
    """In-memory storage backend."""

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
def fake_vector_store(monkeypatch) -> _FakeVectorStore:
    store = _FakeVectorStore()
    monkeypatch.setattr(vector_store, "vector_store", store)
    return store


@pytest.fixture
def fake_embeddings(fake_vector_store: _FakeVectorStore, monkeypatch) -> None:
    """Canned embeddings; query text is recorded for keyword-overlap ranking."""

    def _embed_documents(texts: list[str], **kw: Any) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def _embed_queries(texts: list[str], **kw: Any) -> list[list[float]]:
        if texts:
            fake_vector_store.set_query(texts[0])
        return [[0.0] for _ in texts]

    monkeypatch.setattr(embeddings, "embed_documents", _embed_documents)
    monkeypatch.setattr(embeddings, "embed_queries", _embed_queries)


@pytest.fixture
def fake_storage(monkeypatch) -> _FakeStorage:
    storage = _FakeStorage()
    monkeypatch.setattr(documents_router, "storage", storage)
    return storage


@pytest.fixture
def fake_pipeline(app) -> None:
    """Override the pipeline dependency with a fake-LLM pipeline."""
    app.dependency_overrides[get_rag_pipeline] = lambda: RAGPipeline(_FakeLLM())


@pytest.fixture
def ingest_on_test_db(app, monkeypatch) -> None:
    """Route background ingestion through the test database session factory."""
    monkeypatch.setattr(documents_router, "async_session_factory", app.state.test_factory)


@pytest.fixture
def no_rerank(monkeypatch) -> None:
    """Disable cross-encoder reranking (no model download in tests)."""
    monkeypatch.setattr("app.services.retrieval.settings.RERANK_ENABLED", False)


async def _register_and_token(client: AsyncClient, email: str) -> tuple[dict, dict[str, str]]:
    body = await client.post(
        f"{API}/auth/register",
        json={"email": email, "password": "secret123", "full_name": "User"},
    )
    assert body.status_code == 201, body.text
    token = body.json()["access_token"]
    return body.json(), {"Authorization": f"Bearer {token}"}


async def _make_kb(client: AsyncClient, headers: dict[str, str]) -> UUID:
    body = await client.post(
        f"{API}/knowledge-bases", json={"name": "E2E KB"}, headers=headers
    )
    assert body.status_code == 201, body.text
    return UUID(body.json()["id"])


async def _upload_txt(
    client: AsyncClient, kb_id: UUID, headers: dict[str, str], content: str
) -> dict:
    body = await client.post(
        f"{API}/knowledge-bases/{kb_id}/documents",
        files={"file": ("notes.txt", io.BytesIO(content.encode()), "text/plain")},
        headers=headers,
    )
    assert body.status_code == 201, body.text
    return body.json()


async def _poll_document_status(
    client: AsyncClient, doc_id: UUID, headers: dict[str, str], expected: str
) -> dict:
    """Wait for ingestion to settle, then assert the final status."""
    status = ""
    for _ in range(20):
        body = await client.get(f"{API}/documents/{doc_id}", headers=headers)
        assert body.status_code == 200
        status = body.json()["status"]
        if status in {expected, "failed"}:
            break
        import asyncio

        await asyncio.sleep(0.1)
    assert status == expected, f"document status was {status!r}, expected {expected!r}"
    return body.json()


async def test_full_user_journey_end_to_end(
    client: AsyncClient,
    fake_vector_store: _FakeVectorStore,
    fake_embeddings: None,
    fake_storage: _FakeStorage,
    fake_pipeline: None,
    ingest_on_test_db: None,
    no_rerank: None,
) -> None:
    """register -> KB -> upload -> ingest -> ask -> messages persisted."""
    user, headers = await _register_and_token(client, "e2e-user@example.com")
    kb_id = await _make_kb(client, headers)

    content = (
        "The Multi-User RAG Platform uses PostgreSQL to store users, "
        "knowledge bases, documents and chat sessions."
    )
    uploaded = await _upload_txt(client, kb_id, headers, content)
    document = await _poll_document_status(client, uploaded["id"], headers, "ready")
    assert document["file_name"] == "notes.txt"
    assert document["file_type"] == "txt"

    # The vector store received chunks scoped to the KB and the user.
    indexed = await fake_vector_store.fetch_all(kb_id, user_id=user["id"])
    assert len(indexed) >= 1
    assert indexed[0]["metadata"]["file_name"] == "notes.txt"

    # Create a chat session and ask a question.
    session = await client.post(
        f"{API}/chat/sessions",
        json={"kb_id": str(kb_id), "title": "E2E session"},
        headers=headers,
    )
    assert session.status_code == 201, session.text
    session_id = UUID(session.json()["id"])

    answer = await client.post(
        f"{API}/chat/sessions/{session_id}/ask-sync",
        json={"question": "What does the platform use to store data?"},
        headers=headers,
    )
    assert answer.status_code == 200, answer.text
    payload = answer.json()
    assert payload["session_id"] == str(session_id)
    assert payload["skipped"] is False
    assert "notes.txt" in payload["answer"]
    assert payload["citations"], "expected at least one citation"
    assert payload["retrieved_chunks"], "expected retrieved chunks"

    # Both messages (user + assistant) and the usage log are persisted.
    messages = await client.get(f"{API}/chat/sessions/{session_id}/messages", headers=headers)
    assert messages.status_code == 200
    assert len(messages.json()) == 2
    roles = {m["role"] for m in messages.json()}
    assert roles == {"user", "assistant"}


async def test_ingest_marks_document_failed_on_empty_content(
    client: AsyncClient,
    fake_vector_store: _FakeVectorStore,
    fake_embeddings: None,
    fake_storage: _FakeStorage,
    ingest_on_test_db: None,
    no_rerank: None,
) -> None:
    """A document with no extractable text transitions to ``failed``."""
    _, headers = await _register_and_token(client, "e2e-fail@example.com")
    kb_id = await _make_kb(client, headers)

    uploaded = await _upload_txt(client, kb_id, headers, "   \n\t  ")
    document = await _poll_document_status(client, uploaded["id"], headers, "failed")
    assert document["status"] == "failed"
    assert document.get("error")

    # Nothing was indexed for this document.
    indexed = await fake_vector_store.fetch_all(kb_id, user_id=UUID(document["user_id"]))
    assert indexed == []


async def test_delete_document_removes_chunks(
    client: AsyncClient,
    fake_vector_store: _FakeVectorStore,
    fake_embeddings: None,
    fake_storage: _FakeStorage,
    fake_pipeline: None,
    ingest_on_test_db: None,
    no_rerank: None,
) -> None:
    """Deleting a document removes its chunks from the vector store."""
    _, headers = await _register_and_token(client, "e2e-delete@example.com")
    kb_id = await _make_kb(client, headers)

    uploaded = await _upload_txt(
        client, kb_id, headers, "Deletion removes the indexed chunks entirely."
    )
    doc_id = UUID(uploaded["id"])
    await _poll_document_status(client, doc_id, headers, "ready")

    indexed_before = await fake_vector_store.fetch_all(kb_id, user_id=UUID(uploaded["user_id"]))
    assert indexed_before

    deleted = await client.delete(f"{API}/documents/{doc_id}", headers=headers)
    assert deleted.status_code == 204

    indexed_after = await fake_vector_store.fetch_all(kb_id, user_id=UUID(uploaded["user_id"]))
    assert indexed_after == []


async def test_refresh_token_rotation(
    client: AsyncClient,
    fake_pipeline: None,
    no_rerank: None,
) -> None:
    """A refresh token yields a new working access token (rotation flow)."""
    user, headers = await _register_and_token(client, "e2e-refresh@example.com")

    refreshed = await client.post(
        f"{API}/auth/refresh",
        json={"refresh_token": user["refresh_token"]},
    )
    assert refreshed.status_code == 200, refreshed.text
    new_tokens = refreshed.json()
    assert new_tokens["access_token"]
    assert new_tokens["refresh_token"] != user["refresh_token"]

    me = await client.get(
        f"{API}/auth/me",
        headers={"Authorization": f"Bearer {new_tokens['access_token']}"},
    )
    assert me.status_code == 200
    assert me.json()["email"] == "e2e-refresh@example.com"
