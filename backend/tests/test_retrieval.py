"""Unit tests for retrieval primitives that do not require heavy deps."""
from __future__ import annotations

from app.services.retrieval import (
    NO_CONTEXT_MESSAGE,
    RetrievedChunk,
    reciprocal_rank_fusion,
)


def _chunk(chunk_id: str, text: str = "text") -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        text=text,
        metadata={"file_name": f"{chunk_id}.pdf", "page": 1},
        score=1.0,
    )


def test_rrf_merges_and_ranks() -> None:
    vector = [_chunk("a"), _chunk("b")]
    bm25 = [_chunk("b"), _chunk("c")]
    fused = reciprocal_rank_fusion([vector, bm25], k=60)
    ids = [c.id for c in fused]
    # "b" appears in both lists, so it should outrank single-list entries.
    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c"}


def test_rrf_scores_are_positive_and_ordered() -> None:
    fused = reciprocal_rank_fusion([[_chunk("a")], [_chunk("b")]], k=60)
    assert all(c.score > 0 for c in fused)
    assert fused[0].score >= fused[-1].score


def test_rrf_empty_lists() -> None:
    assert reciprocal_rank_fusion([[], []], k=60) == []


def test_retrieval_result_gate() -> None:
    from app.services.retrieval import RetrievalResult

    empty = RetrievalResult()
    assert empty.has_context is False
    assert empty.no_context_message is None

    below = RetrievalResult(chunks=[_chunk("a")], best_score=0.1)
    assert below.has_context is False
    assert below.no_context_message == NO_CONTEXT_MESSAGE

    above = RetrievalResult(chunks=[_chunk("a")], best_score=0.8)
    assert above.has_context is True
    assert above.no_context_message is None


def test_cosine_similarity_mapping() -> None:
    from app.services.retrieval import _cosine_similarity

    assert _cosine_similarity(0.0) == 1.0
    assert _cosine_similarity(1.0) == 0.0
    assert 0.0 <= _cosine_similarity(0.5) <= 1.0


async def test_retriever_gate_uses_vector_score_without_rerank(monkeypatch) -> None:
    """The gate must use vector cosine similarity (not RRF scores) when rerank is off.

    RRF scores are rank-based (~1/(k+rank), always below any sane threshold),
    so using them for the gate would reject every result. With reranking
    disabled the gate must instead reflect the best vector similarity.
    """
    from app.services.retrieval import RetrievedChunk, Retriever

    monkeypatch.setattr("app.services.retrieval.settings.RERANK_ENABLED", False)
    monkeypatch.setattr("app.services.retrieval.settings.SIMILARITY_THRESHOLD", 0.35)

    chunk = RetrievedChunk(
        id="c1",
        text="relevant text",
        metadata={"file_name": "doc.txt", "page": 1},
        score=0.9,
        source="vector",
    )

    class _FakeVectorStore:
        async def query(self, kb_id, query_embedding, *, user_id, n_results):
            return [
                {"id": chunk.id, "text": chunk.text, "metadata": chunk.metadata,
                 "distance": 0.1},
            ]

        async def fetch_all(self, kb_id, *, user_id):
            return []

    monkeypatch.setattr("app.services.vector_store.vector_store", _FakeVectorStore())
    monkeypatch.setattr(
        "app.services.embeddings.embed_queries", lambda texts, **kw: [[0.0] for _ in texts]
    )

    retriever = Retriever()
    result = await retriever.retrieve("query", kb_id="kb", user_id="u")
    # Cosine similarity 1 - 0.1 = 0.9 clears the 0.35 threshold.
    assert result.has_context is True
    assert result.reranked is False
