"""Unit tests for security helpers (no database required)."""
from __future__ import annotations

from uuid import uuid4

import jwt

from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)


def test_password_hash_and_verify() -> None:
    hashed = hash_password("secret123")
    assert hashed != "secret123"
    assert verify_password("secret123", hashed) is True
    assert verify_password("wrong-password", hashed) is False


def test_access_token_roundtrip() -> None:
    user_id = uuid4()
    token = create_access_token(user_id)
    payload = decode_token(token, expected_type="access")
    assert payload["sub"] == str(user_id)
    assert payload["type"] == "access"


def test_refresh_token_type() -> None:
    token = create_refresh_token("user-123")
    payload = decode_token(token, expected_type="refresh")
    assert payload["type"] == "refresh"


def test_wrong_token_type_rejected() -> None:
    token = create_access_token("user-123")
    try:
        decode_token(token, expected_type="refresh")
    except jwt.InvalidTokenError:
        return
    raise AssertionError("expected InvalidTokenError")
