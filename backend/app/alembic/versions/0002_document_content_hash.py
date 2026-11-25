"""Add ``content_hash`` to the ``documents`` table for duplicate detection.

Revision ID: 0002_document_content_hash
Revises: 0001_initial
Create Date: 2024-01-02 00:00:00.000000

"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers
revision: str = "0002_document_content_hash"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the ``content_hash`` column and an index for dedup lookups."""
    op.add_column(
        "documents",
        sa.Column("content_hash", sa.String(64), nullable=True),
    )
    op.create_index("ix_documents_content_hash", "documents", ["content_hash"])


def downgrade() -> None:
    """Remove the ``content_hash`` column and its index."""
    op.drop_index("ix_documents_content_hash", table_name="documents")
    op.drop_column("documents", "content_hash")
