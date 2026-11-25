"""Unit tests for the LLM provider abstraction using a fake provider."""
from __future__ import annotations

import pytest

from app.services.llm import (
    GenerationResult,
    LLMError,
    LLMProvider,
    get_provider,
)


class _FakeProvider(LLMProvider):
    """A deterministic in-memory provider for tests."""

    name = "fake"

    def __init__(self, reply: str = "ok", raises: bool = False) -> None:
        self.reply = reply
        self.raises = raises

    async def generate_with_usage(self, messages, **kwargs):  # type: ignore[override]
        if self.raises:
            raise LLMError("boom")
        return GenerationResult(
            text=self.reply,
            usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        )

    async def generate_stream(self, messages, **kwargs):  # type: ignore[override]
        if self.raises:
            raise LLMError("boom")
        for piece in self.reply.split(" "):
            yield piece + " "


async def test_provider_generate() -> None:
    provider = _FakeProvider(reply="hello world")
    assert await provider.generate([{"role": "user", "content": "hi"}]) == "hello world"


async def test_provider_generate_with_usage() -> None:
    provider = _FakeProvider(reply="hello")
    result = await provider.generate_with_usage([{"role": "user", "content": "hi"}])
    assert result.text == "hello"
    assert result.usage["total_tokens"] == 3


async def test_provider_stream() -> None:
    provider = _FakeProvider(reply="a b")
    tokens = [t async for t in provider.generate_stream([{"role": "user", "content": "hi"}])]
    assert "".join(tokens) == "a b "


def test_get_provider_unknown_raises() -> None:
    with pytest.raises(LLMError):
        get_provider("does-not-exist")


def test_get_provider_groq_without_sdk() -> None:
    # Groq SDK is not installed here; constructing should raise a clear error.
    import app.services.llm as llm_mod

    try:
        import groq  # noqa: F401
        pytest.skip("groq installed; cannot test missing-sdk path")
    except ImportError:
        pass
    with pytest.raises(LLMError):
        llm_mod.GroqProvider(api_key="x")


def test_normalize_error_maps_rate_limit_and_timeout() -> None:

    from app.services.llm import LLMRateLimitError, LLMTimeoutError, _normalize_llm_error

    class _RateLimit(Exception):
        status_code = 429

    class _Timeout(Exception):
        pass

    assert isinstance(_normalize_llm_error(_RateLimit("429 slow down")), LLMRateLimitError)
    assert isinstance(_normalize_llm_error(TimeoutError()), LLMTimeoutError)
    assert isinstance(_normalize_llm_error(_Timeout("timed out")), LLMTimeoutError)
    assert not _normalize_llm_error(ValueError("bad")).retryable
