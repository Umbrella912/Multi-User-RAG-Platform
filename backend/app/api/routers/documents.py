"""Document upload, listing, status, and deletion endpoints.

Uploads are validated (extension, MIME sniff, max size), hash-checked for
duplicates within a KB, persisted via the storage abstraction, then ingested in
a background task. All queries are scoped to the authenticated user.
"""
from __future__ import annotations

import hashlib
from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user, get_session
from app.core.config import settings
from app.core.logging import get_logger
from app.core.rate_limit import rate_limit
from app.db.session import async_session_factory
from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from app.models.user import User
from app.schemas.document import DocumentOut
from app.services.ingestion import ingest_document
from app.services.retrieval import default_bm25_indexer
from app.services.storage import storage
from app.services.tasks import task_backend
from app.services.vector_store import vector_store

logger = get_logger(__name__)

router = APIRouter(tags=["documents"])

# Magic-byte signatures for supported types.
_MAGIC = {
    "pdf": (b"%PDF",),
    "docx": (b"PK\x03\x04",),  # DOCX is a ZIP container
}

_MAX_BYTES = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024


def _file_type(filename: str) -> str:
    """Return the lowercase extension (without dot) for ``filename``."""
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[1].lower()


def _sniff_mime(file_type: str, head: bytes) -> bool:
    """Return ``True`` if the file's magic bytes match ``file_type``."""
    signatures = _MAGIC.get(file_type)
    if signatures is None:
        # txt/md have no magic bytes; accept them if decodable as text.
        return True
    return any(head.startswith(sig) for sig in signatures)


async def _read_limited(file: UploadFile, max_bytes: int) -> bytes:
    """Read ``file`` fully, rejecting if it exceeds ``max_bytes``."""
    buffer = bytearray()
    while chunk := await file.read(1024 * 1024):
        buffer.extend(chunk)
        if len(buffer) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"file exceeds the {settings.MAX_UPLOAD_SIZE_MB} MB upload limit"
                ),
            )
    return bytes(buffer)


async def _get_owned_kb(session: AsyncSession, kb_id: UUID, user_id: UUID) -> KnowledgeBase:
    """Fetch a KB owned by ``user_id``; 404 if missing or not owned."""
    kb = await session.get(KnowledgeBase, kb_id)
    if kb is None or kb.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="knowledge base not found",
        )
    return kb


async def _get_owned_document(
    session: AsyncSession, doc_id: UUID, user_id: UUID
) -> Document:
    """Fetch a document owned by ``user_id``; 404 if missing or not owned."""
    doc = await session.get(Document, doc_id)
    if doc is None or doc.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="document not found",
        )
    return doc


async def _ingestion_task(document_id: UUID, kb_id: UUID) -> None:
    """Background entry point: open a fresh session and ingest ``document_id``."""
    async with async_session_factory() as session:
        document = await session.get(Document, document_id)
        if document is None:
            return
        kb = await session.get(KnowledgeBase, kb_id)
        if kb is None:
            return
        await ingest_document(
            session,
            document,
            chunk_size=kb.chunk_size,
            chunk_overlap=kb.chunk_overlap,
        )


@router.post(
    "/knowledge-bases/{kb_id}/documents",
    response_model=DocumentOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rate_limit(settings.RATE_LIMIT_UPLOAD_PER_MIN, 60, key_prefix="upload"))],
)
async def upload_document(
    kb_id: UUID,
    background: BackgroundTasks,
    file: Annotated[UploadFile, File()],
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> Document:
    """Upload a document into an owned KB and trigger background ingestion."""
    await _get_owned_kb(session, kb_id, user.id)

    file_type = _file_type(file.filename or "")
    if file_type not in settings.ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"unsupported file type; allowed: "
                f"{', '.join(settings.ALLOWED_FILE_TYPES)}"
            ),
        )

    data = await _read_limited(file, _MAX_BYTES)
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="empty file: nothing was uploaded",
        )

    if not _sniff_mime(file_type, data[:8]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="file contents do not match the declared type",
        )

    content_hash = hashlib.sha256(data).hexdigest()
    existing = await session.scalar(
        select(Document).where(
            Document.kb_id == kb_id,
            Document.content_hash == content_hash,
        )
    )
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="an identical document has already been uploaded to this KB",
        )

    # Use a placeholder id-independent key, then persist the row first so the
    # storage key can reference the real document id.
    document = Document(
        kb_id=kb_id,
        user_id=user.id,
        file_name=file.filename or "document",
        file_type=file_type,
        storage_path="",
        content_hash=content_hash,
        status="pending",
    )
    session.add(document)
    await session.commit()
    await session.refresh(document)

    storage_path = storage.save(user.id, document.id, document.file_name, data)
    document.storage_path = storage_path
    await session.commit()
    await session.refresh(document)

    logger.info(
        "document uploaded",
        extra={
            "document_id": str(document.id),
            "kb_id": str(kb_id),
            "user_id": str(user.id),
            "file_type": file_type,
            "bytes": len(data),
        },
    )
    task_backend.enqueue(background, _ingestion_task, document.id, kb_id)
    return document


@router.get("/knowledge-bases/{kb_id}/documents", response_model=list[DocumentOut])
async def list_documents(
    kb_id: UUID,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[Document]:
    """List documents in an owned KB, optionally filtered by status."""
    await _get_owned_kb(session, kb_id, user.id)
    query = (
        select(Document)
        .where(Document.kb_id == kb_id, Document.user_id == user.id)
        .order_by(Document.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    if status_filter:
        query = query.where(Document.status == status_filter)
    result = await session.scalars(query)
    return list(result.all())


@router.get("/documents/{doc_id}", response_model=DocumentOut)
async def get_document(
    doc_id: UUID,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> Document:
    """Return a single document's status/metadata (only if owned)."""
    return await _get_owned_document(session, doc_id, user.id)


@router.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def delete_document(
    doc_id: UUID,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> None:
    """Delete a document, its stored file, and its vectors (scoped to owner)."""
    doc = await _get_owned_document(session, doc_id, user.id)

    await vector_store.delete_by_document(doc.kb_id, doc.id)
    if doc.storage_path:
        storage.delete(doc.storage_path)

    # Invalidate the BM25 cache for this KB (chunks changed).
    default_bm25_indexer.invalidate(doc.kb_id)

    await session.delete(doc)
    await session.commit()
