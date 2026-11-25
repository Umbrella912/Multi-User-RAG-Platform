"""Health-check endpoints used by orchestrators and load balancers."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import get_session

logger = logging.getLogger("app.health")

router = APIRouter(tags=["health"])


async def _check_database(session: AsyncSession) -> bool:
    """Return ``True`` if the database responds to a trivial query."""
    try:
        await session.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # pragma: no cover - depends on live DB
        logger.error("health_db_check_failed", extra={"error": str(exc)})
        return False


@router.get("/health")
async def health(session: AsyncSession = Depends(get_session)) -> dict:  # noqa: B008
    """Liveness + readiness probe. Returns 503 if the database is unreachable."""
    db_ok = await _check_database(session)
    if not db_ok:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="database unavailable",
        )
    return {
        "status": "ok",
        "db": "ok",
        "chroma": "n/a",
        "version": "0.1.0",
        "environment": settings.ENVIRONMENT,
    }


@router.get("/health/live")
async def health_live() -> dict:
    """Liveness probe: always 200 if the process is up."""
    return {"status": "ok"}


@router.get("/health/ready")
async def health_ready(session: AsyncSession = Depends(get_session)) -> dict:  # noqa: B008
    """Readiness probe: same checks as ``/health``."""
    return await health(session)
