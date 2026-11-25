"""Retry helpers with exponential backoff for transient LLM failures.

Retries timeouts and rate limits (and anything marked ``retryable``). Backoff is
exponential: ``base_delay * 2**attempt``, capped at ``max_delay``, with a small
jitter to avoid thundering-herd retries. For streaming, a retry only happens if
no token has been emitted yet — once output has started, a mid-stream failure
propagates (restarting would duplicate tokens).
"""
from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TypeVar

from app.core.config import settings
from app.core.logging import get_logger
from app.services.llm import LLMError

logger = get_logger(__name__)

T = TypeVar("T")


def _is_retryable(exc: BaseException) -> bool:
    """Return ``True`` if ``exc`` is a transient, retryable failure."""
    if isinstance(exc, asyncio.TimeoutError):
        return True
    if isinstance(exc, LLMError):
        return exc.retryable
    return False


def _backoff(attempt: int) -> float:
    """Compute the (jittered) sleep duration for a 0-based ``attempt``."""
    delay = min(
        settings.LLM_RETRY_MAX_DELAY,
        settings.LLM_RETRY_BASE_DELAY * (2 ** attempt),
    )
    # Full-jitter: random in [0, delay] to spread retries.
    return random.uniform(0, delay)


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int | None = None,
    retry_predicate: Callable[[BaseException], bool] | None = None,
) -> T:
    """Call ``fn`` and retry transient failures with exponential backoff.

    Args:
        fn: An async callable returning ``T``.
        max_attempts: Total attempts (default ``settings.LLM_MAX_RETRIES``).
        retry_predicate: Override the default retryability test.

    Returns:
        The result of ``fn``.

    Raises:
        The last exception when all attempts are exhausted or the error is not
        retryable.
    """
    attempts = max_attempts or settings.LLM_MAX_RETRIES
    predicate = retry_predicate or _is_retryable

    last_exc: BaseException | None = None
    for attempt in range(attempts):
        try:
            return await fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised after retries
            last_exc = exc
            if not predicate(exc) or attempt == attempts - 1:
                raise
            delay = _backoff(attempt)
            logger.warning(
                "retrying after transient error",
                extra={
                    "attempt": attempt + 1,
                    "max_attempts": attempts,
                    "delay": round(delay, 3),
                    "error": str(exc),
                },
            )
            await asyncio.sleep(delay)

    assert last_exc is not None  # pragma: no cover - loop always returns or raises
    raise last_exc  # pragma: no cover - loop always returns or raises


async def retry_stream(
    fn: Callable[[], AsyncIterator[T]],
    *,
    max_attempts: int | None = None,
    retry_predicate: Callable[[BaseException], bool] | None = None,
) -> AsyncIterator[T]:
    """Stream from ``fn``, retrying only before the first item is emitted.

    Once a token has been yielded, the stream is committed: a later failure is
    raised rather than retried (restarting would duplicate already-sent tokens).
    """
    attempts = max_attempts or settings.LLM_MAX_RETRIES
    predicate = retry_predicate or _is_retryable

    last_exc: BaseException | None = None
    for attempt in range(attempts):
        emitted = False
        try:
            async for item in fn():
                emitted = True
                yield item
            return
        except BaseException as exc:  # noqa: BLE001 - re-raised after retries
            last_exc = exc
            # If we already streamed tokens, do not retry (would duplicate output).
            if emitted or not predicate(exc) or attempt == attempts - 1:
                raise
            delay = _backoff(attempt)
            logger.warning(
                "retrying stream after transient error",
                extra={
                    "attempt": attempt + 1,
                    "max_attempts": attempts,
                    "delay": round(delay, 3),
                    "error": str(exc),
                },
            )
            await asyncio.sleep(delay)

    assert last_exc is not None  # pragma: no cover - loop always returns or raises
    raise last_exc  # pragma: no cover - loop always returns or raises
