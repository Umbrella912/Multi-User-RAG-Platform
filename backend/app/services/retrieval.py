"""Retrieval: vector search, BM25 keyword search, RRF fusion, reranking, gate.

The :class:`Retriever` orchestrates the query pipeline:
1. Vector search (Chroma, cosine similarity, top-k configurable).
2. BM25 keyword search (``rank_bm25``) over a per-KB index built lazily from
   stored chunks and cached in memory, invalidated on document changes.
3. Reciprocal Rank Fusion (RRF) to merge the two ranked lists.
4. Optional cross-encoder reranking (toggleable via ``RERANK_ENABLED``).
5. Similarity gate: below ``SIMILARITY_THRESHOLD`` the result carries the
   ``NO_CONTEXT_MESSAGE`` signal so the pipeline can skip the LLM call.

All vector-store access filters by ``user_id`` and ``kb_id`` (data isolation).
``rank_bm25`` and ``sentence-transformers`` are imported lazily so the module
loads without them; clear errors are raised on first use.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from fastapi.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.logging import get_logger
from app.services import embeddings, vector_store

logger = get_logger(__name__)

#: Message returned when retrieval finds nothing relevant.
NO_CONTEXT_MESSAGE = "I couldn't find this in your documents."


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with its score and source."""

    id: str
    text: str
    metadata: dict[str, Any]
    score: float
    source: str = "vector"  # "vector" | "bm25" | "rrf" | "rerank"

    @property
    def file_name(self) -> str:
        return str(self.metadata.get("file_name", "unknown"))

    @property
    def page(self) -> int:
        return int(self.metadata.get("page", 0))

    @property
    def page_end(self) -> int:
        return int(self.metadata.get("page_end", self.page))


@dataclass
class RetrievalResult:
    """Outcome of a multi-stage retrieval pass."""

    chunks: list[RetrievedChunk] = field(default_factory=list)
    best_score: float = 0.0
    vector_results: list[RetrievedChunk] = field(default_factory=list)
    bm25_results: list[RetrievedChunk] = field(default_factory=list)
    fused: list[RetrievedChunk] = field(default_factory=list)
    reranked: bool = False

    @property
    def has_context(self) -> bool:
        """``True`` when the best score clears the similarity threshold."""
        return bool(self.chunks) and self.best_score >= settings.SIMILARITY_THRESHOLD

    @property
    def no_context_message(self) -> str | None:
        """The ``NO_CONTEXT_MESSAGE`` when the gate rejects the context."""
        return None if self.has_context else (NO_CONTEXT_MESSAGE if self.chunks else None)


def _cosine_similarity(distance: float) -> float:
    """Convert a Chroma cosine distance in [0, 2] to a similarity in [0, 1]."""
    return max(0.0, min(1.0, 1.0 - distance))


class Reranker:
    """Optional cross-encoder reranker (``ms-marco-MiniLM-L-6-v2``)."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "reranker requires the 'sentence-transformers' package; "
                "install the project extras (see requirements.txt)"
            ) from exc
        self._model = CrossEncoder(self._model_name)
        return self._model

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Return sigmoid-normalized relevance scores for (query, passage) pairs."""
        if not passages:
            return []
        model = self._load()
        pairs = [(query, passage) for passage in passages]
        raw = model.predict(pairs)
        return [_sigmoid(float(score)) for score in raw]


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid mapping a logit to a probability in [0, 1]."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class BM25Indexer:
    """Per-KB BM25 index over chunk texts, cached in memory.

    The index is built lazily from the Chroma collection (scoped to the user)
    and cached by ``kb_id``. :meth:`invalidate` drops a cached index so it is
    rebuilt after a document is added or deleted.
    """

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def invalidate(self, kb_id: UUID | str) -> None:
        """Drop the cached index for ``kb_id`` (call after ingest/delete)."""
        self._cache.pop(str(kb_id), None)

    def clear(self) -> None:
        """Drop every cached index."""
        self._cache.clear()

    async def get_index(self, kb_id: UUID | str, user_id: UUID | str) -> Any:
        """Return the BM25 index for ``kb_id``, building it on first access."""
        key = str(kb_id)
        if key in self._cache:
            return self._cache[key]

        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "BM25 search requires the 'rank_bm25' package; "
                "install the project extras (see requirements.txt)"
            ) from exc

        chunks = await vector_store.vector_store.fetch_all(kb_id, user_id=user_id)
        # Cache the chunk payload (text + metadata) for mapping ranks back.
        self._cache[f"{key}:chunks"] = chunks
        if not chunks:
            index = None
        else:
            tokenized = [self._tokenize(c["text"]) for c in chunks]
            index = BM25Okapi(tokenized)
        self._cache[key] = index
        return index

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase, split on non-alphanumeric, drop empty tokens."""
        import re

        return re.findall(r"[a-z0-9]+", text.lower())

    async def search(
        self, kb_id: UUID | str, user_id: UUID | str, query: str, top_k: int
    ) -> list[RetrievedChunk]:
        """Return top-k chunks ranked by BM25 score (0 if no index)."""
        key = str(kb_id)
        index = await self.get_index(kb_id, user_id)
        chunks: list[dict[str, Any]] = self._cache.get(f"{key}:chunks", [])
        if index is None or not chunks:
            return []

        query_tokens = self._tokenize(query)
        scores = index.get_scores(query_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results: list[RetrievedChunk] = []
        for idx in ranked[:top_k]:
            if scores[idx] <= 0:
                continue
            chunk = chunks[idx]
            results.append(
                RetrievedChunk(
                    id=chunk["id"],
                    text=chunk["text"],
                    metadata=chunk.get("metadata", {}),
                    score=float(scores[idx]),
                    source="bm25",
                )
            )
        return results


#: Shared, process-wide BM25 index cache. Invalidate after ingest/delete.
default_bm25_indexer = BM25Indexer()


def reciprocal_rank_fusion(
    lists: Iterable[list[RetrievedChunk]], k: int = 60
) -> list[RetrievedChunk]:
    """Merge ranked lists with RRF: ``score = Σ 1/(k + rank)``.

    Args:
        lists: Iterable of ranked :class:`RetrievedChunk` lists (best first).
        k: RRF smoothing constant (default 60).

    Returns:
        Merged chunks sorted by fused score, descending.
    """
    fused: dict[str, RetrievedChunk] = {}
    for ranked in lists:
        for rank, chunk in enumerate(ranked, start=1):
            contribution = 1.0 / (k + rank)
            existing = fused.get(chunk.id)
            if existing is None:
                fused[chunk.id] = RetrievedChunk(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    score=contribution,
                    source="rrf",
                )
            else:
                existing.score += contribution
                # Prefer the richer text/metadata from whichever list had it.
                if not existing.text and chunk.text:
                    existing.text = chunk.text
                existing.metadata = existing.metadata or chunk.metadata
    return sorted(fused.values(), key=lambda c: c.score, reverse=True)


class Retriever:
    """Orchestrates vector + BM25 retrieval, RRF fusion, reranking and gating."""

    def __init__(
        self,
        bm25_indexer: BM25Indexer | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.bm25 = bm25_indexer or default_bm25_indexer
        self._reranker = reranker

    @property
    def rerank_enabled(self) -> bool:
        """Whether cross-encoder reranking is active."""
        return settings.RERANK_ENABLED

    async def _vector_search(
        self,
        query: str,
        kb_id: UUID | str,
        user_id: UUID | str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Embed the query (with BGE prefix) and query Chroma by similarity."""
        embedding = await run_in_threadpool(embeddings.embed_queries, [query])
        hits = await vector_store.vector_store.query(
            kb_id, embedding[0], user_id=user_id, n_results=top_k
        )
        return [
            RetrievedChunk(
                id=hit["id"],
                text=hit["text"],
                metadata=hit["metadata"],
                score=_cosine_similarity(hit["distance"]),
                source="vector",
            )
            for hit in hits
        ]

    async def retrieve(
        self,
        query: str,
        kb_id: UUID | str,
        user_id: UUID | str,
        *,
        top_k_vector: int | None = None,
        top_k_bm25: int | None = None,
        rrf_k: int | None = None,
        context_chunks: int | None = None,
    ) -> RetrievalResult:
        """Run the full retrieval pass and apply the similarity gate.

        Args:
            query: Standalone search query.
            kb_id: Knowledge base id.
            user_id: Caller id (mandatory isolation filter).
            top_k_vector: Vector results to fetch (default ``TOP_K_VECTOR``).
            top_k_bm25: BM25 results to fetch (default ``TOP_K_BM25``).
            rrf_k: RRF smoothing constant (default ``RRF_K``).
            context_chunks: How many fused chunks to keep (default
                ``CONTEXT_CHUNKS``).

        Returns:
            A :class:`RetrievalResult` with the final (reranked) context and a
            ``has_context`` / ``no_context_message`` gate signal.
        """
        top_k_vector = top_k_vector if top_k_vector is not None else settings.TOP_K_VECTOR
        top_k_bm25 = top_k_bm25 if top_k_bm25 is not None else settings.TOP_K_BM25
        rrf_k = rrf_k if rrf_k is not None else settings.RRF_K
        context_chunks = context_chunks if context_chunks is not None else settings.CONTEXT_CHUNKS

        vector_results = await self._vector_search(
            query, kb_id, user_id, top_k_vector
        )
        bm25_results = await self.bm25.search(kb_id, user_id, query, top_k_bm25)

        fused = reciprocal_rank_fusion([vector_results, bm25_results], k=rrf_k)
        top_fused = fused[: max(context_chunks * 5, 30)]  # rerank from a wider pool

        # Optional reranking.
        if self.rerank_enabled and top_fused:
            reranker = self._reranker or Reranker()
            passages = [c.text for c in top_fused if c.text]
            scores = await run_in_threadpool(reranker.score, query, passages)
            keep = [c for c in top_fused if c.text]
            # Re-score only text-bearing chunks; carry textless ones through.
            textless = [c for c in top_fused if not c.text]
            for chunk, score in zip(keep, scores, strict=False):
                chunk.score = float(score)
                chunk.source = "rerank"
            reranked = keep + textless
            reranked.sort(key=lambda c: c.score, reverse=True)
            final = reranked[:context_chunks]
            reranked_flag = True
            best_score = final[0].score if final else 0.0
        else:
            final = fused[:context_chunks]
            reranked_flag = False
            # Gate on the best vector cosine similarity when the reranker is
            # off: RRF scores are rank-based (~1/(k+rank)) and always tiny,
            # so they are not a meaningful threshold signal.
            best_score = max((c.score for c in vector_results), default=0.0)

        return RetrievalResult(
            chunks=final,
            best_score=best_score,
            vector_results=vector_results,
            bm25_results=bm25_results,
            fused=fused,
            reranked=reranked_flag,
        )
