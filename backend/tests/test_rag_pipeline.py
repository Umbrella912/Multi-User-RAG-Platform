"""Unit tests for the RAG pipeline orchestration (fake LLM + fake retriever)."""
from __future__ import annotations

from app.services.llm import GenerationResult, LLMProvider
from app.services.rag_pipeline import RAGPipeline
from app.services.retrieval import RetrievalResult, RetrievedChunk


class _FakeLLM(LLMProvider):
    name = "fake"

    def __init__(self, reply: str = "The answer is in [doc.pdf, p.1].") -> None:
        self.reply = reply
        self.stream_tokens = list(reply)
        self.last_usage: dict[str, int] | None = None

    async def generate_with_usage(self, messages, **kwargs):  # type: ignore[override]
        self.last_usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        return GenerationResult(
            text=self.reply,
            usage=self.last_usage,
        )

    async def generate_stream(self, messages, **kwargs):  # type: ignore[override]
        for token in self.stream_tokens:
            yield token
        self.last_usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }


class _FakeRetriever:
    """Stand-in retriever returning a canned RetrievalResult."""

    def __init__(self, result: RetrievalResult) -> None:
        self._result = result

    async def retrieve(self, query, kb_id, user_id, **kwargs):  # type: ignore[override]
        return self._result


def _chunk(file_name: str, page: int) -> RetrievedChunk:
    return RetrievedChunk(
        id="doc:0",
        text="relevant context",
        metadata={"file_name": file_name, "page": page, "page_end": page},
        score=0.9,
        source="vector",
    )


async def test_pipeline_full_flow() -> None:
    llm = _FakeLLM()
    retriever = _FakeRetriever(
        RetrievalResult(chunks=[_chunk("doc.pdf", 1)], best_score=0.9)
    )
    pipeline = RAGPipeline(llm, retriever)

    result = await pipeline.answer(
        "What does the doc say?", "kb-1", "user-1", history=[]
    )
    assert result.skipped is False
    assert result.answer == llm.reply
    assert result.token_usage is not None
    assert result.token_usage["total_tokens"] == 15
    assert result.citations
    assert result.citations[0].file_name == "doc.pdf"


async def test_pipeline_gate_short_circuits_without_llm_call() -> None:
    llm = _FakeLLM()
    # Empty chunks -> gate rejects -> no LLM call.
    retriever = _FakeRetriever(RetrievalResult(chunks=[], best_score=0.0))
    pipeline = RAGPipeline(llm, retriever)

    result = await pipeline.answer("unanswerable", "kb-1", "user-1")
    assert result.skipped is True
    assert result.answer == "I couldn't find this in your documents."
    assert result.citations == []


async def test_pipeline_gate_rejects_low_score() -> None:
    llm = _FakeLLM()
    retriever = _FakeRetriever(
        RetrievalResult(chunks=[_chunk("doc.pdf", 1)], best_score=0.1)
    )
    pipeline = RAGPipeline(llm, retriever)

    result = await pipeline.answer("q", "kb-1", "user-1")
    assert result.skipped is True
    assert "couldn't find" in result.answer.lower()


async def test_pipeline_stream_emits_events() -> None:
    llm = _FakeLLM(reply="Answer [doc.pdf, p.1].")
    retriever = _FakeRetriever(
        RetrievalResult(chunks=[_chunk("doc.pdf", 1)], best_score=0.9)
    )
    pipeline = RAGPipeline(llm, retriever)

    events = [
        event
        async for event in pipeline.answer_stream("q", "kb-1", "user-1")
    ]
    types = [e["type"] for e in events]
    assert "rewrite" in types
    assert "citation" in types
    assert "token" in types
    assert types[-1] == "done"
    done = events[-1]
    assert done["skipped"] is False
    assert done["token_usage"]["total_tokens"] == 15
    assert "retrieved_chunks" in done


async def test_pipeline_stream_gate_emits_done_only() -> None:
    llm = _FakeLLM()
    retriever = _FakeRetriever(RetrievalResult(chunks=[], best_score=0.0))
    pipeline = RAGPipeline(llm, retriever)

    events = [
        event
        async for event in pipeline.answer_stream("q", "kb-1", "user-1")
    ]
    # Rewrite event first, then a single short-circuit done.
    assert events[0]["type"] == "rewrite"
    assert events[-1]["type"] == "done"
    assert events[-1]["skipped"] is True
    assert events[-1]["answer"] == "I couldn't find this in your documents."
