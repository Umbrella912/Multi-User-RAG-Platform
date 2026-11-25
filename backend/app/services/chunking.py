"""Recursive character text splitter with metadata-preserving chunking.

Pure-Python (no heavy deps) so it is fully unit-testable. Chunks carry the full
provenance needed for retrieval and citation: ``file_name``, ``page`` (the
min-max page span the chunk spans), ``chunk_index``, ``document_id``, ``kb_id``
and ``user_id``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Preferred split separators, in order of preference.
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    """A single chunk of text with full provenance metadata."""

    text: str
    chunk_index: int
    document_id: str
    kb_id: str
    user_id: str
    file_name: str
    page: int  # first page the chunk touches
    page_end: int  # last page the chunk touches (== page for single-page)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Stable Chroma-style id: ``{document_id}:{chunk_index}``."""
        return f"{self.document_id}:{self.chunk_index}"


class RecursiveCharacterTextSplitter:
    """Split text into chunks up to ``chunk_size`` chars with ``chunk_overlap``.

    Splitting is recursive: the splitter tries the least-destructive separator
    first and recurses on pieces that are still too large.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 160,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be in [0, chunk_size)")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or _SEPARATORS

    def split_text(self, text: str) -> list[str]:
        """Split ``text`` into a list of chunk strings."""
        return self._split(text, self.separators)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split ``text`` using the first applicable separator."""
        if len(text) <= self.chunk_size:
            return [text] if text else []

        if not separators:
            # Hard split on character count as a last resort.
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

        separator = separators[0]
        rest = separators[1:]
        pieces = text.split(separator) if separator else list(text)

        # Re-join with the separator to keep it attached where possible.
        if separator:
            pieces = [piece + separator for piece in pieces[:-1]] + [pieces[-1]]

        result: list[str] = []
        current: list[str] = []
        current_len = 0
        for piece in pieces:
            if not piece:
                continue
            if len(piece) > self.chunk_size:
                # Flush the buffer, then recurse into the oversized piece.
                if current:
                    result.append("".join(current))
                    current, current_len = [], 0
                result.extend(self._split(piece, rest))
                continue
            added_len = len(piece) + (1 if current else 0)
            if current_len + added_len > self.chunk_size and current:
                result.append("".join(current))
                # Carry an overlap window from the tail into the next chunk.
                tail = self._overlap_tail(current)
                current = ([tail] if tail else []) + [piece]
                current_len = len(tail) + (len(piece) + (1 if tail else 0))
            else:
                current.append(piece)
                current_len += added_len
        if current:
            result.append("".join(current))
        return result

    def _overlap_tail(self, pieces: list[str]) -> str:
        """Return the trailing string (up to ``chunk_overlap`` chars) of ``pieces``."""
        if self.chunk_overlap <= 0 or not pieces:
            return ""
        tail: list[str] = []
        length = 0
        for piece in reversed(pieces):
            if length + len(piece) > self.chunk_overlap:
                # Take a partial slice of this piece to fill the overlap.
                remaining = self.chunk_overlap - length
                if remaining > 0:
                    tail.insert(0, piece[-remaining:])
                break
            tail.insert(0, piece)
            length += len(piece)
        return "".join(tail)


def chunk_pages(
    pages: list,
    *,
    document_id: str,
    kb_id: str,
    user_id: str,
    file_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Chunk a sequence of :class:`~app.services.loaders.LoadedPage` objects.

    Each returned :class:`Chunk` preserves metadata and the page span it covers
    (``page`` = first page, ``page_end`` = last page of its source segments).

    Args:
        pages: Loaded page segments (any objects with ``.text`` and ``.page``).
        document_id: Owning document id (string form).
        kb_id: Owning knowledge-base id (string form).
        user_id: Owning user id (string form).
        file_name: Original file name.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks in characters.

    Returns:
        List of :class:`Chunk` with ``chunk_index`` assigned sequentially.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    chunks: list[Chunk] = []
    index = 0
    for page in pages:
        for text in splitter.split_text(page.text):
            if not text.strip():
                continue
            chunks.append(
                Chunk(
                    text=text,
                    chunk_index=index,
                    document_id=document_id,
                    kb_id=kb_id,
                    user_id=user_id,
                    file_name=file_name,
                    page=page.page,
                    page_end=page.page,
                )
            )
            index += 1
    return chunks
