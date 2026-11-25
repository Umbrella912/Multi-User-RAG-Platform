"""RAG pipeline orchestration.

Flow: rewrite (optional) -> retrieve -> rerank -> build prompt -> generate ->
extract citations. Supports both buffered and streaming generation. Citations are
parsed from the model's ``[file_name, p.X]`` references and resolved to the
retrieved chunks' metadata.
"""
from __future__ import annotations

import re
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from app.core.config import settings
from app.core.logging import get_logger
from app.services.llm import LLMProvider
from app.services.prompts import SYSTEM_PROMPT, build_rag_prompt
from app.services.query_rewriter import QueryRewriter
from app.services.retrieval import (
    NO_CONTEXT_MESSAGE,
    RetrievalResult,
    RetrievedChunk,
    Retriever,
)
from app.services.retry import retry_async, retry_stream

logger = get_logger(__name__)

# Matches citations like [quarterly_report.pdf, p.3] or [notes.txt, p.2-4].
_CITATION_RE = re.compile(
    r"\[(?P<file>[^\],]+),\s*p\.(?P<pages>\d+(?:-\d+)?)\]",
    re.IGNORECASE,
)


@dataclass
class Citation:
    """A resolved source citation."""

    file_name: str
    page: int
    page_end: int
    chunk_id: str
    snippet: str


@dataclass
class RAGResult:
    """Result of a non-streaming RAG run."""

    answer: str
    citations: list[Citation] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    query: str = ""
    model: str = ""
    token_usage: dict[str, Any] | None = None
    latency_ms: int | None = None
    skipped: bool = False  # True when the similarity gate short-circuited.


class RAGPipeline:
    """Orchestrates the full retrieval-augmented generation flow."""

    def __init__(
        self,
        llm: LLMProvider,
        retriever: Retriever | None = None,
        rewriter: QueryRewriter | None = None,
    ) -> None:
        self._llm = llm
        self._retriever = retriever or Retriever()
        self._rewriter = rewriter or QueryRewriter(llm)

    @property
    def model(self) -> str:
        """The name of the underlying LLM provider."""
        return self._llm.name

    def _history_messages(self, history: Sequence[dict[str, str]]) -> list[dict[str, str]]:
        """Return the last N history messages, capped for prompt size."""
        recent = list(history)[-settings.HISTORY_TURNS :]
        return [{"role": m["role"], "content": m["content"]} for m in recent]

    async def _prepare(
        self,
        question: str,
        kb_id: UUID | str,
        user_id: UUID | str,
        history: Sequence[dict[str, str]],
        use_history: bool,
    ) -> tuple[str, RetrievalResult]:
        """Rewrite the query (if needed) and run retrieval."""
        if use_history and history:
            query = await self._rewriter.rewrite(question, history)
        else:
            query = question
        result = await self._retriever.retrieve(query, kb_id, user_id)
        return query, result

    async def answer(
        self,
        question: str,
        kb_id: UUID | str,
        user_id: UUID | str,
        *,
        history: Sequence[dict[str, str]] | None = None,
        use_history: bool = True,
    ) -> RAGResult:
        """Run a full non-streaming RAG answer.

        Returns a :class:`RAGResult`. When retrieval finds no relevant context,
        the answer is the ``NO_CONTEXT_MESSAGE`` and ``skipped`` is ``True``
        (no LLM call is made).
        """
        started = time.perf_counter()
        history = history or []

        query, retrieval = await self._prepare(question, kb_id, user_id, history, use_history)

        if not retrieval.has_context:
            return RAGResult(
                answer=NO_CONTEXT_MESSAGE,
                citations=[],
                chunks=retrieval.chunks,
                query=query,
                model=self._llm.name,
                latency_ms=_elapsed_ms(started),
                skipped=True,
            )

        prompt = build_rag_prompt(query, retrieval.chunks)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        gen = await retry_async(lambda: self._llm.generate_with_usage(messages))
        answer_text = gen.text
        citations = self.extract_citations(answer_text, retrieval.chunks)
        return RAGResult(
            answer=answer_text,
            citations=citations,
            chunks=retrieval.chunks,
            query=query,
            model=self._llm.name,
            token_usage=gen.usage or None,
            latency_ms=_elapsed_ms(started),
        )

    async def answer_stream(
        self,
        question: str,
        kb_id: UUID | str,
        user_id: UUID | str,
        *,
        history: Sequence[dict[str, str]] | None = None,
        use_history: bool = True,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream the RAG answer as event dicts.

        Yields ``{"type": "rewrite", "query": ...}``, ``{"type": "citation", ...}``,
        ``{"type": "token", "text": ...}`` and a final ``{"type": "done", ...}``
        event. If the similarity gate rejects the context, yields a single
        ``{"type": "done", "answer": NO_CONTEXT_MESSAGE, "skipped": True}``.
        """
        started = time.perf_counter()
        history = history or []

        query, retrieval = await self._prepare(question, kb_id, user_id, history, use_history)
        yield {"type": "rewrite", "query": query}

        if not retrieval.has_context:
            yield {
                "type": "done",
                "answer": NO_CONTEXT_MESSAGE,
                "citations": [],
                "retrieved_chunks": [],
                "skipped": True,
                "latency_ms": _elapsed_ms(started),
                "token_usage": None,
            }
            return

        # Emit citations as their context chunks become relevant (by order).
        for chunk in retrieval.chunks:
            yield {
                "type": "citation",
                "file_name": chunk.file_name,
                "page": chunk.page,
                "chunk_id": chunk.id,
                "snippet": chunk.text[:160],
            }

        prompt = build_rag_prompt(query, retrieval.chunks)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        answer_text = ""
        stream = retry_stream(lambda: self._llm.generate_stream(messages))
        async for token in stream:
            answer_text += token
            yield {"type": "token", "text": token}

        token_usage = getattr(self._llm, "last_usage", None)
        citations = self.extract_citations(answer_text, retrieval.chunks)
        yield {
            "type": "done",
            "answer": answer_text,
            "citations": [c.__dict__ for c in citations],
            "retrieved_chunks": [
                {
                    "id": c.id,
                    "file_name": c.file_name,
                    "page": c.page,
                    "score": c.score,
                }
                for c in retrieval.chunks
            ],
            "skipped": False,
            "latency_ms": _elapsed_ms(started),
            "token_usage": token_usage,
        }

    def extract_citations(
        self, answer_text: str, chunks: list[RetrievedChunk]
    ) -> list[Citation]:
        """Parse ``[file, p.X]`` citations from the answer and resolve them.

        Each citation's ``file_name``/``page`` are matched to a retrieved chunk;
        when a referenced page is not among the retrieved chunks, the citation
        is kept with ``chunk_id=""`` so the source is still surfaced to the user.
        """
        citations: list[Citation] = []
        seen: set[tuple[str, int, int]] = set()

        for match in _CITATION_RE.finditer(answer_text):
            file_name = match.group("file").strip()
            pages_str = match.group("pages")
            if "-" in pages_str:
                page_start, page_end = (int(p) for p in pages_str.split("-", 1))
            else:
                page_start = page_end = int(pages_str)

            key = (file_name, page_start, page_end)
            if key in seen:
                continue
            seen.add(key)

            chunk = _find_chunk(chunks, file_name, page_start)
            citations.append(
                Citation(
                    file_name=file_name,
                    page=page_start,
                    page_end=page_end,
                    chunk_id=chunk.id if chunk else "",
                    snippet=(chunk.text[:160] if chunk else ""),
                )
            )
        return citations


def _find_chunk(
    chunks: list[RetrievedChunk], file_name: str, page: int
) -> RetrievedChunk | None:
    """Find the chunk whose file and page range contains ``page``."""
    for chunk in chunks:
        if chunk.file_name == file_name and chunk.page <= page <= chunk.page_end:
            return chunk
    # Fall back to file match only (page may be a range boundary).
    for chunk in chunks:
        if chunk.file_name == file_name:
            return chunk
    return None


def _elapsed_ms(started: float) -> int:
    """Return milliseconds elapsed since ``started`` (a ``perf_counter`` value)."""
    return int((time.perf_counter() - started) * 1000)
