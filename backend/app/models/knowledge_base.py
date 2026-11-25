"""Knowledge base model (one vector collection per KB)."""
from __future__ import annotations

import uuid

from sqlalchemy import CheckConstraint, Integer, String, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin


class KnowledgeBase(Base, TimestampMixin):
    """A user-owned knowledge base. Maps to a Chroma collection ``kb_{id}``."""

    __tablename__ = "knowledge_bases"
    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_kb_user_name"),
        CheckConstraint("status IN ('active', 'archived')", name="ck_kb_status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
        comment="Owner; also enforced on every Chroma query.",
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding_model: Mapped[str] = mapped_column(
        String(255), nullable=False, server_default="BAAI/bge-small-en-v1.5"
    )
    chunk_size: Mapped[int] = mapped_column(Integer, nullable=False, server_default="800")
    chunk_overlap: Mapped[int] = mapped_column(Integer, nullable=False, server_default="160")
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default="active")
