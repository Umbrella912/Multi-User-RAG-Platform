"""Unit tests for the in-memory rate limiter (no database required)."""
from __future__ import annotations

from app.core.rate_limit import InMemoryRateLimiter


def test_allows_up_to_limit() -> None:
    limiter = InMemoryRateLimiter()
    for _ in range(5):
        assert limiter.is_allowed("k", limit=5, period=60) is True
    # 6th exceeds the limit.
    assert limiter.is_allowed("k", limit=5, period=60) is False


def test_different_keys_are_independent() -> None:
    limiter = InMemoryRateLimiter()
    assert limiter.is_allowed("a", limit=2, period=60) is True
    assert limiter.is_allowed("a", limit=2, period=60) is True
    assert limiter.is_allowed("a", limit=2, period=60) is False  # exceeded
    assert limiter.is_allowed("b", limit=2, period=60) is True
    assert limiter.is_allowed("b", limit=2, period=60) is True
    assert limiter.is_allowed("b", limit=2, period=60) is False  # exceeded
