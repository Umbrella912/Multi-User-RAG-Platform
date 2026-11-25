"""Unit tests for the recursive splitter and metadata-preserving chunking."""
from __future__ import annotations

from app.services.chunking import RecursiveCharacterTextSplitter, chunk_pages
from app.services.loaders import LoadedPage


def test_split_small_text_returns_single_chunk() -> None:
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    assert splitter.split_text("short text") == ["short text"]


def test_split_respects_chunk_size() -> None:
    splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=5)
    text = "word " * 100  # 500 chars
    chunks = splitter.split_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= 50 for c in chunks)


def test_split_invalid_overlap_rejected() -> None:
    try:
        RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=100)
    except ValueError:
        return
    raise AssertionError("expected ValueError for overlap >= chunk_size")


def test_chunk_pages_preserves_metadata_and_page() -> None:
    pages = [
        LoadedPage(text="alpha", page=1),
        LoadedPage(text="beta", page=2),
    ]
    chunks = chunk_pages(
        pages,
        document_id="doc-1",
        kb_id="kb-1",
        user_id="user-1",
        file_name="a.pdf",
        chunk_size=100,
        chunk_overlap=0,
    )
    assert len(chunks) == 2
    first, second = chunks
    assert first.chunk_index == 0
    assert second.chunk_index == 1
    assert first.page == 1 and first.page_end == 1
    assert second.page == 2 and second.page_end == 2
    for chunk in chunks:
        assert chunk.document_id == "doc-1"
        assert chunk.kb_id == "kb-1"
        assert chunk.user_id == "user-1"
        assert chunk.file_name == "a.pdf"
    assert first.id == "doc-1:0"


def test_chunk_pages_drops_empty_segments() -> None:
    pages = [LoadedPage(text="   \n  ", page=1), LoadedPage(text="real", page=1)]
    chunks = chunk_pages(
        pages,
        document_id="doc",
        kb_id="kb",
        user_id="u",
        file_name="x.txt",
        chunk_size=100,
        chunk_overlap=0,
    )
    assert len(chunks) == 1
    assert chunks[0].text == "real"
