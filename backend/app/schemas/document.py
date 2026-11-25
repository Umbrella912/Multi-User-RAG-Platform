"""Document request/response schemas."""
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class DocumentOut(BaseModel):
    """Public representation of a document and its ingestion state."""

    id: UUID
    kb_id: UUID
    user_id: UUID
    file_name: str
    file_type: str
    status: str
    error_message: str | None
    chunk_count: int
    char_count: int | None
    page_count: int | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
