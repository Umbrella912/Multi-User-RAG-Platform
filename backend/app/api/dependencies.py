"""Shared FastAPI dependencies: DB session and current-user resolution."""
from __future__ import annotations

from typing import Annotated, Any
from uuid import UUID

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import decode_token
from app.db.session import get_session
from app.models.user import User

# No-error bearer extractor: returns None when no/invalid Authorization header.
bearer_scheme = HTTPBearer(auto_error=False)


async def get_optional_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> User | None:
    """Resolve the authenticated user, or ``None`` if absent/invalid (never raises)."""
    if credentials is None or not credentials.credentials:
        return None
    try:
        payload = decode_token(credentials.credentials, expected_type="access")
    except jwt.PyJWTError:
        return None
    user_id = payload.get("sub")
    if not user_id:
        return None
    try:
        user = await session.get(User, UUID(user_id))
    except Exception:
        return None
    if user is None or not user.is_active:
        return None
    return user


async def get_current_user(
    user: Annotated[User | None, Depends(get_optional_current_user)],
) -> User:
    """Require an authenticated, active user; raise 401 otherwise."""
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


_pipeline: Any | None = None


def get_rag_pipeline():
    """Return the process-wide :class:`RAGPipeline` (built lazily, cached).

    Constructs the configured LLM provider on first use. Tests override this
    dependency to inject a pipeline with a fake LLM/retriever.
    """
    global _pipeline
    if _pipeline is None:
        from app.services.llm import get_provider
        from app.services.rag_pipeline import RAGPipeline

        _pipeline = RAGPipeline(get_provider())
    return _pipeline
