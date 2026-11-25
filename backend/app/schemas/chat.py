"""Chat session, message, and ask request/response schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ChatSessionCreate(BaseModel):
    """Payload for creating a chat session bound to a knowledge base."""

    kb_id: UUID
    title: str | None = Field(default=None, max_length=255)


class ChatSessionOut(BaseModel):
    """Public representation of a chat session."""

    id: UUID
    kb_id: UUID
    user_id: UUID
    title: str | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class MessageOut(BaseModel):
    """Public representation of a persisted chat message."""

    id: UUID
    session_id: UUID
    user_id: UUID
    role: str
    content: str
    citations: list[dict[str, Any]] | None
    token_usage: dict[str, Any] | None
    latency_ms: int | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AskRequest(BaseModel):
    """Payload for asking a question in a session."""

    question: str = Field(..., min_length=1, max_length=8000)
    use_history: bool = True


class AskSyncResponse(BaseModel):
    """Non-streaming RAG answer returned by the ``ask-sync`` endpoint."""

    session_id: UUID
    answer: str
    citations: list[dict[str, Any]]
    retrieved_chunks: list[dict[str, Any]]
    latency_ms: int
    token_usage: dict[str, Any] | None
    skipped: bool
    model: str
