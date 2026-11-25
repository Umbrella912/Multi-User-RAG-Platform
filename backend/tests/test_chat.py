"""Chat endpoint tests: session CRUD, ask-sync, SSE, persistence, isolation.

The RAG pipeline and background persistence are mocked (fake LLM + fake
retriever + test session factory) so the tests exercise the router/persistence
logic without heavy deps or a real LLM. DB-backed: auto-skips without Postgres.
"""
from __future__ import annotations

import json
from uuid import UUID

import pytest_asyncio
from httpx import AsyncClient

from app.api.dependencies import get_rag_pipeline
from app.api.routers import chat as chat_router
from app.api.routers.chat import _format_sse
from app.services.llm import GenerationResult, LLMProvider
from app.services.rag_pipeline import RAGPipeline
from app.services.retrieval import RetrievalResult, RetrievedChunk

API = "/api"

ANSWER = "Refunds are 30 days per [policy.pdf, p.1]."


def test_format_sse_wraps_json_frame() -> None:
    frame = _format_sse({"type": "token", "text": "hello"})
    assert frame.startswith("data: ")
    assert frame.endswith("\n\n")
    assert json.loads(frame[len("data: "):]) == {"type": "token", "text": "hello"}


class _FakeLLM(LLMProvider):
    name = "fake"

    def __init__(self, answer: str = ANSWER) -> None:
        self.answer = answer
        self.last_usage: dict[str, int] | None = None

    async def generate_with_usage(self, messages, **kwargs):  # type: ignore[override]
        self.last_usage = {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}
        return GenerationResult(text=self.answer, usage=self.last_usage)

    async def generate_stream(self, messages, **kwargs):  # type: ignore[override]
        for token in self.answer.split(" "):
            yield token + " "
        self.last_usage = {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}


class _FakeRetriever:
    async def retrieve(self, query, kb_id, user_id, **kwargs):  # type: ignore[override]
        chunk = RetrievedChunk(
            id="doc:0",
            text="Refunds are 30 days.",
            metadata={"file_name": "policy.pdf", "page": 1, "page_end": 1},
            score=0.9,
            source="vector",
        )
        return RetrievalResult(chunks=[chunk], best_score=0.9)


@pytest_asyncio.fixture
def fake_pipeline(app, monkeypatch, test_session_factory) -> RAGPipeline:
    """Wire the app to a pipeline with a fake LLM/retriever and test factory."""
    pipeline = RAGPipeline(_FakeLLM(), _FakeRetriever())
    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline
    # Redirect background persistence to the test DB.
    monkeypatch.setattr(chat_router, "_session_factory", test_session_factory)
    return pipeline


async def _make_session(client: AsyncClient, email: str = "chat@example.com") -> tuple[dict, UUID]:
    """Register a user, create a KB, and a chat session; return (headers, session_id)."""
    reg = await client.post(
        f"{API}/auth/register",
        json={"email": email, "password": "secret123"},
    )
    headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}
    kb = await client.post(
        f"{API}/knowledge-bases", json={"name": "Chat KB"}, headers=headers
    )
    kb_id = kb.json()["id"]
    sess = await client.post(
        f"{API}/chat/sessions", json={"kb_id": kb_id, "title": "S1"}, headers=headers
    )
    return headers, UUID(sess.json()["id"])


async def test_session_crud(client: AsyncClient, fake_pipeline) -> None:
    headers, session_id = await _make_session(client)

    # List sessions.
    listing = await client.get(f"{API}/chat/sessions", headers=headers)
    assert listing.status_code == 200
    assert len(listing.json()) == 1

    # Messages (empty initially).
    msgs = await client.get(f"{API}/chat/sessions/{session_id}/messages", headers=headers)
    assert msgs.status_code == 200
    assert msgs.json() == []

    # Delete.
    deleted = await client.delete(f"{API}/chat/sessions/{session_id}", headers=headers)
    assert deleted.status_code == 204
    gone = await client.get(f"{API}/chat/sessions/{session_id}/messages", headers=headers)
    assert gone.status_code == 404


async def test_ask_sync_persists_messages_and_usage(
    client: AsyncClient, fake_pipeline, test_session_factory
) -> None:
    headers, session_id = await _make_session(client)

    resp = await client.post(
        f"{API}/chat/sessions/{session_id}/ask-sync",
        json={"question": "What is the refund policy?"},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["answer"] == ANSWER
    assert body["skipped"] is False
    assert body["token_usage"]["total_tokens"] == 20
    assert body["citations"]
    assert body["citations"][0]["file_name"] == "policy.pdf"

    # Both user and assistant messages are persisted.
    msgs = await client.get(f"{API}/chat/sessions/{session_id}/messages", headers=headers)
    roles = [m["role"] for m in msgs.json()]
    assert roles == ["user", "assistant"]
    assistant = msgs.json()[1]
    assert assistant["citations"] is not None
    assert assistant["token_usage"]["total_tokens"] == 20

    # A UsageLog row was written.
    async with test_session_factory() as db:
        from sqlalchemy import select

        from app.models.usage_log import UsageLog

        rows = list((await db.scalars(select(UsageLog))).all())
        assert len(rows) == 1
        assert rows[0].action == "chat_completion"
        assert rows[0].total_tokens == 20
        assert rows[0].retrieved_chunks == 1


async def test_ask_sse_streams_events(client: AsyncClient, fake_pipeline) -> None:
    headers, session_id = await _make_session(client)

    resp = await client.post(
        f"{API}/chat/sessions/{session_id}/ask",
        json={"question": "What is the refund policy?"},
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    # Parse SSE frames.
    frames = [
        json.loads(line[len("data: "):])
        for line in resp.text.splitlines()
        if line.startswith("data: ")
    ]
    types = [f["type"] for f in frames]
    assert "rewrite" in types
    assert "citation" in types
    assert "token" in types
    assert types[-1] == "done"

    done = frames[-1]
    assert done["answer"] == ANSWER
    assert done["token_usage"]["total_tokens"] == 20
    assert done["retrieved_chunks"]

    # Assistant message persisted after streaming.
    msgs = await client.get(f"{API}/chat/sessions/{session_id}/messages", headers=headers)
    roles = [m["role"] for m in msgs.json()]
    assert roles == ["user", "assistant"]


async def test_ask_requires_session_ownership(client: AsyncClient, fake_pipeline) -> None:
    headers_a, session_id = await _make_session(client, email="owner@example.com")
    # Different user cannot access the session.
    headers_b, _ = await _make_session(client, email="other@example.com")
    resp = await client.get(
        f"{API}/chat/sessions/{session_id}/messages", headers=headers_b
    )
    assert resp.status_code == 404

    resp = await client.post(
        f"{API}/chat/sessions/{session_id}/ask-sync",
        json={"question": "hi"},
        headers=headers_b,
    )
    assert resp.status_code == 404


async def test_create_session_requires_owned_kb(client: AsyncClient, fake_pipeline) -> None:
    reg = await client.post(
        f"{API}/auth/register",
        json={"email": "kbowner@example.com", "password": "secret123"},
    )
    headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}
    # KB owned by someone else -> 404.
    resp = await client.post(
        f"{API}/chat/sessions",
        json={"kb_id": UUID("00000000-0000-0000-0000-000000000000")},
        headers=headers,
    )
    assert resp.status_code == 404
