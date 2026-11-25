"""Password hashing and JWT (access + refresh) helpers."""
from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import jwt
from passlib.context import CryptContext

from app.core.config import settings

# bcrypt is the only supported password scheme.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ACCESS_TOKEN_TYPE: Literal["access"] = "access"
REFRESH_TOKEN_TYPE: Literal["refresh"] = "refresh"


def hash_password(password: str) -> str:
    """Hash ``password`` with bcrypt for safe storage."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Return ``True`` if ``plain_password`` matches ``hashed_password``."""
    return pwd_context.verify(plain_password, hashed_password)


def _create_token(
    subject: str | uuid.UUID, token_type: str, expires_delta: timedelta
) -> str:
    """Build a signed JWT carrying ``sub``, ``type``, ``iat``, ``exp`` and ``jti``."""
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": str(subject),
        "type": token_type,
        "iat": now,
        "exp": now + expires_delta,
        "jti": str(uuid.uuid4()),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def create_access_token(subject: str | uuid.UUID) -> str:
    """Create a short-lived access token (default 15 minutes)."""
    return _create_token(
        subject,
        ACCESS_TOKEN_TYPE,
        timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )


def create_refresh_token(subject: str | uuid.UUID) -> str:
    """Create a long-lived refresh token (default 7 days)."""
    return _create_token(
        subject,
        REFRESH_TOKEN_TYPE,
        timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
    )


def decode_token(token: str, expected_type: str | None = None) -> dict[str, Any]:
    """Decode and validate ``token``; raise ``jwt.PyJWTError`` on failure.

    Args:
        token: The JWT string.
        expected_type: If provided, the token's ``type`` claim must match.

    Returns:
        The decoded JWT payload.
    """
    payload = jwt.decode(
        token,
        settings.JWT_SECRET,
        algorithms=[settings.JWT_ALGORITHM],
    )
    if expected_type is not None and payload.get("type") != expected_type:
        raise jwt.InvalidTokenError(f"expected {expected_type} token")
    return payload
