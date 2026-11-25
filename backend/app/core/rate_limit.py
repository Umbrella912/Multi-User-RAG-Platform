"""Per-user rate limiting with a swappable backend.

The default :class:`InMemoryRateLimiter` is process-local and fine for
development / single-worker use. For multi-worker or multi-instance deployments,
implement :class:`RateLimiter` against Redis (same interface) and assign it to
``limiter`` — no call-site changes required.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from app.api.dependencies import get_optional_current_user
from app.models.user import User


class RateLimiter(ABC):
    """Abstract fixed/sliding-window limiter keyed by an arbitrary string."""

    @abstractmethod
    def is_allowed(self, key: str, limit: int, period: int) -> bool:
        """Return ``True`` if ``key`` is within ``limit`` hits per ``period`` seconds."""
        raise NotImplementedError


class InMemoryRateLimiter(RateLimiter):
    """Sliding-window counter held in process memory (not shared across workers)."""

    def __init__(self) -> None:
        self._hits: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str, limit: int, period: int) -> bool:
        now = time.monotonic()
        window = self._hits[key]
        # Evict timestamps outside the rolling window.
        while window and now - window[0] >= period:
            window.pop(0)
        if len(window) >= limit:
            return False
        window.append(now)
        return True

    def reset(self, key: str | None = None) -> None:
        """Drop all recorded hits, or just the hits for ``key``."""
        if key is None:
            self._hits.clear()
        else:
            self._hits.pop(key, None)


# Shared singleton. Swap this for a Redis-backed implementation in production.
limiter: RateLimiter = InMemoryRateLimiter()


def rate_limit(limit: int, period: int, *, key_prefix: str = "global"):
    """Build a FastAPI dependency enforcing ``limit`` requests / ``period`` seconds.

    The rate-limit key is the authenticated user id when present, otherwise the
    client IP — so limits are per-user while public endpoints stay protected.
    """

    async def _dependency(
        request: Request,
        user: Annotated[User | None, Depends(get_optional_current_user)],
    ) -> None:
        if user is not None:
            key = f"{key_prefix}:user:{user.id}"
        else:
            host = request.client.host if request.client else "unknown"
            key = f"{key_prefix}:ip:{host}"
        if not limiter.is_allowed(key, limit, period):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="rate limit exceeded",
            )

    return _dependency
