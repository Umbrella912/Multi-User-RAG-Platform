"""Unit tests for prompts and citation extraction."""
from __future__ import annotations

from app.services.prompts import build_rag_prompt, format_context
from app.services.rag_pipeline import RAGPipeline
from app.services.retrieval import RetrievedChunk


def _chunk(file_name: str, page: int, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        id="doc:0",
        text=text,
        metadata={"file_name": file_name, "page": page, "page_end": page},
        score=0.9,
    )


def test_format_context_tags_source() -> None:
    chunks = [_chunk("report.pdf", 3, "some content")]
    rendered = format_context(chunks)
    assert "[file: report.pdf, page: 3]" in rendered
    assert "some content" in rendered


def test_build_rag_prompt_contains_question_and_context() -> None:
    chunks = [_chunk("a.pdf", 1, "alpha")]
    prompt = build_rag_prompt("What is alpha?", chunks)
    assert "alpha" in prompt
    assert "What is alpha?" in prompt
    assert "I couldn't find this in your documents." in prompt


def test_build_rag_prompt_empty_context() -> None:
    prompt = build_rag_prompt("question", [])
    assert "no passages" in prompt.lower() or "no passages" in prompt


def test_system_prompt_covers_contract() -> None:
    from app.services.prompts import SYSTEM_PROMPT

    assert "only" in SYSTEM_PROMPT.lower()
    assert "cite" in SYSTEM_PROMPT.lower()
    assert "I couldn't find this in your documents" in SYSTEM_PROMPT
    assert "untrusted" in SYSTEM_PROMPT.lower()


def test_extract_citations_resolves_to_chunks() -> None:
    pipeline = RAGPipeline.__new__(RAGPipeline)  # bypass __init__ (no llm needed)
    chunks = [
        _chunk("report.pdf", 3, "refund policy text"),
        _chunk("notes.txt", 2, "shipping text"),
    ]
    answer = "Per [report.pdf, p.3], refunds are allowed. See also [notes.txt, p.2]."
    citations = pipeline.extract_citations(answer, chunks)
    names = {c.file_name for c in citations}
    assert names == {"report.pdf", "notes.txt"}
    # Resolved chunks point at the matching chunk ids.
    report = next(c for c in citations if c.file_name == "report.pdf")
    assert report.chunk_id == "doc:0"


def test_extract_citations_handles_unknown_page() -> None:
    pipeline = RAGPipeline.__new__(RAGPipeline)
    chunks = [_chunk("report.pdf", 3, "x")]
    citations = pipeline.extract_citations("See [report.pdf, p.9] for details.", chunks)
    assert len(citations) == 1
    # Unknown page falls back to the file match (still resolved).
    assert citations[0].file_name == "report.pdf"
