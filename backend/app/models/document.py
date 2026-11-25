"""Uploaded document model with ingestion status tracking."""
from __future__ import annotations

import uuid

from sqlalchemy import CheckConstraint, Integer, String, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin


class Document(Base, TimestampMixin):
    """A document uploaded into a knowledge base, tracked through ingestion."""

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("kb_id", "file_name", name="uq_doc_kb_filename"),
        CheckConstraint(
            "file_type IN ('pdf', 'docx', 'txt', 'md')", name="ck_doc_file_type"
        ),
        CheckConstraint(
            "status IN ('pending', 'processing', 'ready', 'failed')", name="ck_doc_status"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    kb_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    file_name: Mapped[str] = mapped_column(String(512), nullable=False)
    file_type: Mapped[str] = mapped_column(String(8), nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True,
        comment="sha256 of the uploaded bytes; used for duplicate detection.",
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default="pending")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    char_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
