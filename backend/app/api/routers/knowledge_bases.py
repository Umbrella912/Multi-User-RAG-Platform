"""Knowledge base CRUD. Every query is scoped to the authenticated user."""
from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user, get_session
from app.core.config import settings
from app.core.rate_limit import rate_limit
from app.models.knowledge_base import KnowledgeBase
from app.models.user import User
from app.schemas.knowledge_base import (
    KnowledgeBaseCreate,
    KnowledgeBaseOut,
    KnowledgeBaseUpdate,
)

router = APIRouter(
    prefix="/knowledge-bases",
    tags=["knowledge-bases"],
    dependencies=[Depends(get_current_user)],
)


async def _get_owned_kb(
    session: AsyncSession, kb_id: UUID, user_id: UUID
) -> KnowledgeBase:
    """Fetch a KB owned by ``user_id``; 404 if missing or not owned (no leak)."""
    kb = await session.get(KnowledgeBase, kb_id)
    if kb is None or kb.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="knowledge base not found",
        )
    return kb


@router.get("", response_model=list[KnowledgeBaseOut])
async def list_knowledge_bases(
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> list[KnowledgeBase]:
    """List the current user's knowledge bases, newest first."""
    result = await session.scalars(
        select(KnowledgeBase)
        .where(KnowledgeBase.user_id == user.id)
        .order_by(KnowledgeBase.created_at.desc())
    )
    return list(result.all())


@router.post(
    "",
    response_model=KnowledgeBaseOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rate_limit(30, 60, key_prefix="kb_create"))],
)
async def create_knowledge_base(
    body: KnowledgeBaseCreate,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> KnowledgeBase:
    """Create a knowledge base for the current user."""
    kb = KnowledgeBase(
        user_id=user.id,
        name=body.name,
        description=body.description,
        embedding_model=body.embedding_model or settings.EMBEDDING_MODEL,
        chunk_size=body.chunk_size or settings.CHUNK_SIZE,
        chunk_overlap=body.chunk_overlap or settings.CHUNK_OVERLAP,
    )
    session.add(kb)
    try:
        await session.commit()
    except IntegrityError as err:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="knowledge base name already exists",
        ) from err
    await session.refresh(kb)
    return kb


@router.get("/{kb_id}", response_model=KnowledgeBaseOut)
async def get_knowledge_base(
    kb_id: UUID,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> KnowledgeBase:
    """Get a single knowledge base (only if owned by the caller)."""
    return await _get_owned_kb(session, kb_id, user.id)


@router.patch("/{kb_id}", response_model=KnowledgeBaseOut)
async def update_knowledge_base(
    kb_id: UUID,
    body: KnowledgeBaseUpdate,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> KnowledgeBase:
    """Update editable fields of an owned knowledge base."""
    kb = await _get_owned_kb(session, kb_id, user.id)
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(kb, field, value)
    try:
        await session.commit()
    except IntegrityError as err:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="knowledge base name already exists",
        ) from err
    await session.refresh(kb)
    return kb


@router.delete(
    "/{kb_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
)
async def delete_knowledge_base(
    kb_id: UUID,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    """Delete an owned knowledge base (cascades documents and chunks)."""
    kb = await _get_owned_kb(session, kb_id, user.id)
    await session.delete(kb)
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
