"""LLM provider abstraction.

A single :class:`LLMProvider` interface with:
- ``generate_with_usage`` — buffered generation returning text *and* token usage.
- ``generate_stream`` — async generator yielding tokens (usage captured in
  ``last_usage`` once the stream completes).
- ``generate`` — convenience wrapper returning just the text.

Implemented for Groq, Gemini and Ollama. Providers are selected via
``LLM_PROVIDER``. SDK imports are lazy so the module loads without any provider
package installed; a clear error is raised on first use when the chosen
provider's SDK is missing. SDK timeout / rate-limit errors are normalized into
:class:`LLMTimeoutError` / :class:`LLMRateLimitError`, both marked retryable.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from app.core.config import settings


class LLMError(Exception):
    """Raised when a provider cannot be initialized or a generation call fails.

    ``retryable`` is ``True`` for transient failures (timeouts, rate limits)
    that the pipeline should retry with backoff.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class LLMRateLimitError(LLMError):
    """Rate-limited by the provider (HTTP 429); retryable."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=True)


class LLMTimeoutError(LLMError):
    """The provider call timed out; retryable."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=True)


@dataclass
class GenerationResult:
    """A completed generation with its token usage."""

    text: str
    usage: dict[str, int] = field(default_factory=dict)


def _normalize_llm_error(exc: Exception) -> LLMError:
    """Convert an arbitrary SDK exception into a typed, retryable-if-transient error."""
    name = type(exc).__name__.lower()
    message = str(exc)
    lowered = message.lower()
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "status", None)

    if (
        status == 429
        or "ratelimit" in name
        or "rate_limit" in name
        or "rate limit" in lowered
        or "429" in lowered
    ):
        return LLMRateLimitError(message)
    if (
        isinstance(exc, asyncio.TimeoutError)
        or "timeout" in name
        or "timed out" in lowered
    ):
        return LLMTimeoutError(message)
    return LLMError(message)


class LLMProvider(ABC):
    """Interface for text generation, both buffered and streaming."""

    #: Human-readable provider name.
    name: str = "base"

    #: Token usage captured from the most recent streaming call (best-effort).
    last_usage: dict[str, int] | None = None

    @abstractmethod
    async def generate_with_usage(
        self, messages: Sequence[dict[str, str]], **kwargs: Any
    ) -> GenerationResult:
        """Return the full response and its token usage."""
        raise NotImplementedError

    @abstractmethod
    def generate_stream(
        self, messages: Sequence[dict[str, str]], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Yield the assistant response token-by-token (async generator)."""
        raise NotImplementedError  # pragma: no cover - abstract

    async def generate(
        self, messages: Sequence[dict[str, str]], **kwargs: Any
    ) -> str:
        """Return just the assistant response text."""
        return (await self.generate_with_usage(messages, **kwargs)).text

    def _params(self, **overrides: Any) -> dict[str, Any]:
        """Merge default generation params with per-call overrides."""
        params = {
            "temperature": settings.LLM_TEMPERATURE,
            "max_tokens": settings.LLM_MAX_TOKENS,
        }
        params.update(overrides)
        return params


class GroqProvider(LLMProvider):
    """Groq-hosted OpenAI-compatible chat completions (async)."""

    name = "groq"

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        try:
            from groq import AsyncGroq
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise LLMError(
                "the Groq provider requires the 'groq' package; "
                "install the project extras (see requirements.txt)"
            ) from exc
        self.model = model or settings.LLM_MODEL
        self._client = AsyncGroq(api_key=api_key or settings.GROQ_API_KEY)

    async def generate_with_usage(
        self, messages: Sequence[dict[str, str]], **kwargs: Any
    ) -> GenerationResult:
        params = self._params(**kwargs)
        try:
            response = await self._client.chat.completions.create(
                model=self.model, messages=cast(Any, list(messages)), **params
            )
        except Exception as exc:
            raise _normalize_llm_error(exc) from exc

        usage = getattr(response, "usage", None)
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        total = getattr(usage, "total_tokens", 0) or (prompt + completion)
        return GenerationResult(
            text=response.choices[0].message.content or "",
            usage={
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": total,
            },
        )

    async def generate_stream(
        self, messages: Sequence[dict[str, str]], **kwargs: Any
    ) -> AsyncIterator[str]:
        params = self._params(**kwargs)
        try:
            # stream_options is not in the groq SDK's typed signature; build
            # the kwargs dict and bridge it to the client.
            call_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": list(messages),
                "stream": True,
                "stream_options": {"include_usage": True},
                **params,
            }
            stream = await self._client.chat.completions.create(
                **cast(Any, call_kwargs),
            )
            async for chunk in stream:
                if chunk.usage:
                    self.last_usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens or 0,
                        "completion_tokens": chunk.usage.completion_tokens or 0,
                        "total_tokens": chunk.usage.total_tokens or 0,
                    }
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as exc:
            raise _normalize_llm_error(exc) from exc


class GeminiProvider(LLMProvider):
    """Google Gemini chat generation (async via a threadpool bridge)."""

    name = "gemini"

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        try:
            import google.generativeai as genai
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise LLMError(
                "the Gemini provider requires the 'google-generativeai' package; "
                "install the project extras (see requirements.txt)"
            ) from exc
        genai.configure(api_key=api_key or settings.GEMINI_API_KEY)
        self.model_name = model or settings.LLM_MODEL
        self._genai = genai

    def _contents(self, messages: Sequence[dict[str, str]]) -> str:
        return "\n".join(m["content"] for m in messages if m.get("role") == "user")

    def _config(self, params: dict[str, Any]) -> Any:
        return self._genai.types.GenerationConfig(
            temperature=params["temperature"],
            max_output_tokens=params["max_tokens"],
        )

    async def generate_with_usage(
        self, messages: Sequence[dict[str, str]], **kwargs: Any
    ) -> GenerationResult:
        import asyncio

        params = self._params(**kwargs)
        model = self._genai.GenerativeModel(self.model_name)
        try:
            response = await asyncio.to_thread(
                model.generate_content,
                self._contents(messages),
                generation_config=self._config(params),
            )
        except Exception as exc:
            raise _normalize_llm_error(exc) from exc

        usage_meta = getattr(response, "usage_metadata", None) or {}
        prompt = int(usage_meta.get("prompt_token_count", 0) or 0)
        completion = int(usage_meta.get("candidates_token_count", 0) or 0)
        total = int(usage_meta.get("total_token_count", 0) or (prompt + completion))
        return GenerationResult(
            text=response.text or "",
            usage={
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": total,
            },
        )

    async def generate_stream(
        self, messages: Sequence[dict[str, str]], **kwargs: Any
    ) -> AsyncIterator[str]:
        import asyncio

        model = self._genai.GenerativeModel(self.model_name)
        try:
            stream = await asyncio.to_thread(
                model.generate_content, self._contents(messages), stream=True
            )
        except Exception as exc:
            raise _normalize_llm_error(exc) from exc

        async for chunk in self._stream_sync(stream):
            yield chunk

    async def _stream_sync(self, stream: Any) -> AsyncIterator[str]:
        """Yield text chunks from a sync Gemini stream via a threadpool."""
        import asyncio

        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def _consume() -> None:
            try:
                for chunk in stream:
                    if chunk.text:
                        queue.put_nowait(chunk.text)
                    usage_meta = getattr(chunk, "usage_metadata", None)
                    if usage_meta:
                        self.last_usage = {
                            "prompt_tokens": int(usage_meta.get("prompt_token_count", 0) or 0),
                            "completion_tokens": int(
                                usage_meta.get("candidates_token_count", 0) or 0
                            ),
                            "total_tokens": int(
                                usage_meta.get("total_token_count", 0) or 0
                            ),
                        }
            except Exception as exc:  # pragma: no cover - provider specific
                queue.put_nowait(None)
                raise LLMError(f"gemini stream failed: {exc}") from exc
            finally:
                queue.put_nowait(None)

        loop = asyncio.get_event_loop()
        task = loop.run_in_executor(None, _consume)
        while True:
            item = await queue.get()
            if item is None:
                await task
                break
            yield item


class OllamaProvider(LLMProvider):
    """Local Ollama chat generation (async)."""

    name = "ollama"

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        try:
            from ollama import AsyncClient
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise LLMError(
                "the Ollama provider requires the 'ollama' package; "
                "install the project extras (see requirements.txt)"
            ) from exc
        self.model = model or settings.LLM_MODEL
        self._client = AsyncClient(host=base_url or settings.OLLAMA_BASE_URL)

    def _options(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "temperature": params["temperature"],
            "num_predict": params["max_tokens"],
        }

    async def generate_with_usage(
        self, messages: Sequence[dict[str, str]], **kwargs: Any
    ) -> GenerationResult:
        params = self._params(**kwargs)
        try:
            response = await self._client.chat(
                model=self.model,
                messages=[
                    {"role": m["role"], "content": m["content"]} for m in messages
                ],
                options=self._options(params),
            )
        except Exception as exc:
            raise _normalize_llm_error(exc) from exc

        prompt = getattr(response, "prompt_eval_count", 0) or 0
        completion = getattr(response, "eval_count", 0) or 0
        return GenerationResult(
            text=response.message.content or "",
            usage={
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
            },
        )

    async def generate_stream(
        self, messages: Sequence[dict[str, str]], **kwargs: Any
    ) -> AsyncIterator[str]:
        params = self._params(**kwargs)
        try:
            stream = await self._client.chat(
                model=self.model,
                messages=[
                    {"role": m["role"], "content": m["content"]} for m in messages
                ],
                stream=True,
                options=self._options(params),
            )
            async for chunk in stream:
                if chunk.done and chunk.eval_count is not None:
                    self.last_usage = {
                        "prompt_tokens": chunk.prompt_eval_count or 0,
                        "completion_tokens": chunk.eval_count or 0,
                        "total_tokens": (chunk.prompt_eval_count or 0) + (chunk.eval_count or 0),
                    }
                if chunk.message and chunk.message.content:
                    yield chunk.message.content
        except Exception as exc:
            raise _normalize_llm_error(exc) from exc


def get_provider(name: str | None = None, **kwargs: Any) -> LLMProvider:
    """Construct the provider selected by ``LLM_PROVIDER`` (or ``name``).

    Args:
        name: Optional provider name override; defaults to settings.
        **kwargs: Forwarded to the provider constructor (model/api_key/...).

    Returns:
        An initialized :class:`LLMProvider`.

    Raises:
        LLMError: if the provider name is unknown or its SDK is missing.
    """
    chosen = (name or settings.LLM_PROVIDER).lower()
    if chosen == "groq":
        return GroqProvider(**kwargs)
    if chosen == "gemini":
        return GeminiProvider(**kwargs)
    if chosen == "ollama":
        return OllamaProvider(**kwargs)
    raise LLMError(f"unknown LLM provider: {chosen!r}")
