"""Unit tests for the retry/backoff helpers."""
from __future__ import annotations

import asyncio

import pytest

from app.services.llm import LLMError, LLMRateLimitError, LLMTimeoutError
from app.services.retry import retry_async, retry_stream


async def test_retry_async_succeeds_after_transient_failures() -> None:
    calls = {"n": 0}

    async def _fn() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise LLMRateLimitError("429")
        return "ok"

    result = await retry_async(_fn, max_attempts=5)
    assert result == "ok"
    assert calls["n"] == 3


async def test_retry_async_does_not_retry_non_retryable() -> None:
    calls = {"n": 0}

    async def _fn() -> str:
        calls["n"] += 1
        raise LLMError("auth failed")

    with pytest.raises(LLMError):
        await retry_async(_fn, max_attempts=5)
    assert calls["n"] == 1


async def test_retry_async_exhausts_attempts() -> None:
    calls = {"n": 0}

    async def _fn() -> str:
        calls["n"] += 1
        raise LLMTimeoutError("timed out")

    with pytest.raises(LLMTimeoutError):
        await retry_async(_fn, max_attempts=3)
    assert calls["n"] == 3


async def test_retry_stream_retries_before_first_token() -> None:
    calls = {"n": 0}

    async def _gen():
        calls["n"] += 1
        if calls["n"] < 2:
            raise LLMRateLimitError("429")
        yield "a"
        yield "b"

    out = [t async for t in retry_stream(_gen, max_attempts=3)]
    assert out == ["a", "b"]
    assert calls["n"] == 2


async def test_retry_stream_does_not_retry_after_tokens_emitted() -> None:
    calls = {"n": 0}

    async def _gen():
        calls["n"] += 1
        yield "partial"
        raise LLMRateLimitError("429 mid-stream")

    with pytest.raises(LLMRateLimitError):
        async for _ in retry_stream(_gen, max_attempts=3):
            pass
    # Emitted tokens then failed -> no retry.
    assert calls["n"] == 1


async def test_retry_async_catches_asyncio_timeout() -> None:
    calls = {"n": 0}

    async def _fn() -> str:
        calls["n"] += 1
        raise TimeoutError()

    with pytest.raises(asyncio.TimeoutError):
        await retry_async(_fn, max_attempts=2)
    assert calls["n"] == 2
