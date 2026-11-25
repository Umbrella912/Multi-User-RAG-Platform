"""Unit tests for loaders that do not require heavy dependencies (txt/md)."""
from __future__ import annotations

import pytest

from app.services.ingestion import clean_text
from app.services.loaders import DocumentLoadError, load_file


def test_load_txt_returns_single_page() -> None:
    pages = load_file("txt", b"hello world")
    assert len(pages) == 1
    assert pages[0].text == "hello world"
    assert pages[0].page == 1


def test_load_txt_empty_raises() -> None:
    with pytest.raises(DocumentLoadError, match="empty text file"):
        load_file("txt", b"   \n  ")


def test_load_markdown_splits_on_page_breaks() -> None:
    md = b"# Page one\n---\n# Page two"
    pages = load_file("md", md)
    assert len(pages) == 2
    assert "Page one" in pages[0].text
    assert "Page two" in pages[1].text
    assert [p.page for p in pages] == [1, 2]


def test_load_markdown_empty_raises() -> None:
    with pytest.raises(DocumentLoadError, match="empty markdown"):
        load_file("md", b"")


def test_unsupported_type_raises() -> None:
    with pytest.raises(DocumentLoadError, match="unsupported file type"):
        load_file("exe", b"MZ")


def test_clean_text_normalizes_whitespace() -> None:
    assert clean_text("a   b\t\tc\n\n\n\nd") == "a b c\n\nd"
    assert clean_text("  leading and trailing  ") == "leading and trailing"
