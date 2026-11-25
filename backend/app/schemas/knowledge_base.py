"""Knowledge base request/response schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeBaseCreate(BaseModel):
    """Payload for creating a knowledge base."""

    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    embedding_model: str | None = None
    chunk_size: int | None = Field(default=None, ge=64, le=4000)
    chunk_overlap: int | None = Field(default=None, ge=0, le=1000)


class KnowledgeBaseUpdate(BaseModel):
    """Payload for updating a knowledge base (all fields optional)."""

    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = None
    embedding_model: str | None = None
    chunk_size: int | None = Field(default=None, ge=64, le=4000)
    chunk_overlap: int | None = Field(default=None, ge=0, le=1000)
    status: Literal["active", "archived"] | None = None


class KnowledgeBaseOut(BaseModel):
    """Public representation of a knowledge base."""

    id: UUID
    user_id: UUID
    name: str
    description: str | None
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
