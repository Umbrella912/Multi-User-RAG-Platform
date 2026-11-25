"""Chat session model."""
from __future__ import annotations

import uuid

from sqlalchemy import Index, String, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin


class ChatSession(Base, TimestampMixin):
    """A conversation thread bound to one user and one knowledge base."""

    __tablename__ = "chat_sessions"
    __table_args__ = (Index("idx_sess_user_kb", "user_id", "kb_id"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    kb_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
