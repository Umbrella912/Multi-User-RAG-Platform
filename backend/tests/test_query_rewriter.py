"""Unit tests for query rewriting with a fake LLM."""
from __future__ import annotations

from app.services.llm import GenerationResult, LLMProvider
from app.services.query_rewriter import QueryRewriter


class _FakeLLM(LLMProvider):
    name = "fake"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    async def generate_with_usage(self, messages, **kwargs):  # type: ignore[override]
        self.calls += 1
        return GenerationResult(text=self.reply)

    async def generate_stream(self, messages, **kwargs):  # type: ignore[override]
        yield self.reply


async def test_rewrite_uses_llm_when_history_present() -> None:
    llm = _FakeLLM("refund policy for electronics")
    rewriter = QueryRewriter(llm)
    history = [{"role": "user", "content": "What is the refund policy?"}]
    query = await rewriter.rewrite("Does it apply to electronics?", history)
    assert query == "refund policy for electronics"
    assert llm.calls == 1


async def test_rewrite_returns_original_without_history() -> None:
    llm = _FakeLLM("should-not-be-used")
    rewriter = QueryRewriter(llm)
    assert await rewriter.rewrite("standalone question", []) == "standalone question"
    assert llm.calls == 0


async def test_rewrite_falls_back_on_llm_error() -> None:
    from app.services.llm import LLMError

    class _FailingLLM(LLMProvider):
        name = "failing"

        async def generate_with_usage(self, messages, **kwargs):  # type: ignore[override]
            raise LLMError("boom")

        async def generate_stream(self, messages, **kwargs):  # type: ignore[override]
            yield ""

    rewriter = QueryRewriter(_FailingLLM())
    history = [{"role": "user", "content": "prior"}]
    assert await rewriter.rewrite("follow up", history) == "follow up"


def test_build_history_text_caps_to_last_n() -> None:
    rewriter = QueryRewriter(_FakeLLM("x"), history_turns=2)
    history = [{"role": "user", "content": str(i)} for i in range(5)]
    text = rewriter.build_history_text(history)
    # Only the last 2 messages should appear.
    assert "3" not in text.splitlines()
    assert "4" in text
