"""Ingestion pipeline orchestration: load -> clean -> chunk -> embed -> store.

Updates a :class:`~app.models.document.Document` through the states
``pending -> processing -> ready`` (or ``-> failed`` with an error message).
Blocking work (file parsing, embedding, vector writes) is dispatched to a
threadpool so the call-path stays async.
"""
from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.document import Document
from app.services import embeddings, loaders, vector_store
from app.services.chunking import chunk_pages
from app.services.retrieval import default_bm25_indexer
from app.services.storage import storage

logger = get_logger(__name__)

# Collapse runs of whitespace into a single space while preserving newlines.
_WS = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalize whitespace: trim trailing spaces, collapse blank-line runs."""
    text = _WS.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


async def ingest_document(
    session: AsyncSession,
    document: Document,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> Document:
    """Run the full ingestion pipeline for ``document`` and persist its state.

    The document is moved to ``processing`` at the start and to ``ready`` on
    success (with ``chunk_count``/``char_count``/``page_count`` populated) or
    ``failed`` on any error, which is stored in ``error_message``.

    Args:
        session: Async DB session.
        document: The document row to ingest (must already be persisted).
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        The updated document.
    """
    await _set_status(session, document, "processing", error=None)

    try:
        # 1) Load: read stored bytes and extract (text, page) segments.
        data = await run_in_threadpool(storage.read, document.storage_path)
        pages = await run_in_threadpool(loaders.load_file, document.file_type, data)

        # 2) Clean: normalize whitespace on each segment (LoadedPage is frozen,
        #    so rebuild each instance rather than mutating it).
        pages = [replace(page, text=clean_text(page.text)) for page in pages]
        pages = [page for page in pages if page.text.strip()]
        if not pages:
            raise loaders.DocumentLoadError("document contains no extractable text")

        # 3) Chunk with full provenance metadata.
        chunks = chunk_pages(
            pages,
            document_id=str(document.id),
            kb_id=str(document.kb_id),
            user_id=str(document.user_id),
            file_name=document.file_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if not chunks:
            raise loaders.DocumentLoadError("document yielded zero chunks")

        # 4) Embed the chunk texts (batched, normalized).
        embeddings_list = await run_in_threadpool(
            embeddings.embed_documents, [chunk.text for chunk in chunks]
        )

        # 5) Store in Chroma with a mandatory user/kb metadata filter on writes.
        written = await vector_store.vector_store.add_chunks(
            document.kb_id, chunks, embeddings_list
        )

        await _set_status(
            session,
            document,
            "ready",
            error=None,
            chunk_count=written,
            char_count=sum(len(chunk.text) for chunk in chunks),
            page_count=max((page.page for page in pages), default=0),
        )
        logger.info(
            "ingested document",
            extra={
                "document_id": str(document.id),
                "kb_id": str(document.kb_id),
                "user_id": str(document.user_id),
                "chunks": written,
            },
        )
        # Invalidate the cached BM25 index so the next search sees new chunks.
        default_bm25_indexer.invalidate(document.kb_id)
    except Exception as exc:
        await _set_status(session, document, "failed", error=str(exc))
        logger.warning(
            "document ingestion failed",
            extra={
                "document_id": str(document.id),
                "kb_id": str(document.kb_id),
                "error": str(exc),
            },
        )
    return document


async def _set_status(
    session: AsyncSession,
    document: Document,
    status: str,
    *,
    error: str | None = None,
    chunk_count: int | None = None,
    char_count: int | None = None,
    page_count: int | None = None,
) -> None:
    """Apply a status transition and commit."""
    document.status = status
    document.error_message = error
    if chunk_count is not None:
        document.chunk_count = chunk_count
    if char_count is not None:
        document.char_count = char_count
    if page_count is not None:
        document.page_count = page_count
    await session.commit()
    await session.refresh(document)


def serialize_chunk(chunk: Any) -> dict[str, Any]:
    """Return a plain dict for a chunk (handy for debugging/tests)."""
    return {
        "id": chunk.id,
        "text": chunk.text,
        "chunk_index": chunk.chunk_index,
        "file_name": chunk.file_name,
        "page": chunk.page,
        "page_end": chunk.page_end,
        "document_id": chunk.document_id,
        "kb_id": chunk.kb_id,
        "user_id": chunk.user_id,
    }
