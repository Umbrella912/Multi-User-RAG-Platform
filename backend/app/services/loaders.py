"""File loaders: extract (text, page) segments from PDF/DOCX/TXT/MD.

Each loader returns a list of :class:`LoadedPage` segments in reading order,
with a 1-based ``page`` number where pagination exists (PDF). DOCX files have
no native pagination and use ``page=0`` ("not paginated"); TXT/MD are treated
as a single page unless the markdown uses ``---`` page breaks.

Heavy dependencies (``PyMuPDF``, ``python-docx``) are imported lazily inside
each loader so the module imports cleanly without them and produces a clear
error only when the relevant loader is actually used.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Matches a markdown horizontal-rule / page-break line on its own.
_MD_PAGE_BREAK = re.compile(r"^\s*---+\s*$", re.MULTILINE)


class DocumentLoadError(Exception):
    """Raised when a file cannot be loaded or yields no usable text."""


@dataclass(frozen=True)
class LoadedPage:
    """A contiguous segment of extracted text and its page number.

    Attributes:
        text: The extracted (raw) text for the segment.
        page: 1-based page number; ``0`` means "not paginated".
    """

    text: str
    page: int


def _require(module_name: str, import_name: str | None = None) -> Any:
    """Import ``module_name`` (or ``import_name``) and raise a clear error if absent."""
    try:
        return __import__(module_name)
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise DocumentLoadError(
            f"loader for this file type requires the '{module_name}' package; "
            "install the project extras (see requirements.txt)"
        ) from exc


def load_pdf(data: bytes) -> list[LoadedPage]:
    """Extract text page-by-page from a PDF using PyMuPDF (``fitz``).

    Raises:
        DocumentLoadError: if the PDF has no pages, cannot be opened, or has no
            extractable text (a scanned/image PDF).
    """
    fitz = _require("fitz", "PyMuPDF")
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise DocumentLoadError(f"failed to open PDF: {exc}") from exc

    try:
        pages: list[LoadedPage] = []
        for index, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append(LoadedPage(text=text, page=index))

        if not pages:
            if doc.page_count == 0:
                raise DocumentLoadError("empty PDF: the document has no pages")
            raise DocumentLoadError(
                "scanned PDF: no extractable text found (the file likely contains "
                "only images; OCR is required to index it)"
            )
        return pages
    finally:
        doc.close()


def load_docx(data: bytes) -> list[LoadedPage]:
    """Extract paragraphs from a DOCX using ``python-docx``.

    DOCX has no reliable pagination, so every segment carries ``page=0``.
    """
    import io

    docx = _require("docx", "python-docx")
    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise DocumentLoadError(f"failed to open DOCX: {exc}") from exc

    pages: list[LoadedPage] = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            pages.append(LoadedPage(text=text, page=0))

    if not pages:
        raise DocumentLoadError("empty DOCX: the document has no text paragraphs")
    return pages


def load_text(data: bytes) -> list[LoadedPage]:
    """Return the whole text as a single 1-based page, decoding defensively."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception as exc:  # pragma: no cover - defensive
            raise DocumentLoadError(f"failed to decode text file: {exc}") from exc

    if not text.strip():
        raise DocumentLoadError("empty text file: no content to index")
    return [LoadedPage(text=text, page=1)]


def load_markdown(data: bytes) -> list[LoadedPage]:
    """Load markdown, splitting on ``---`` page-break lines into pages."""
    if not data.strip():
        raise DocumentLoadError("empty markdown file: no content to index")
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover - defensive
        raise DocumentLoadError(f"failed to decode markdown file: {exc}") from exc
    parts = _MD_PAGE_BREAK.split(text)
    pages: list[LoadedPage] = []
    for index, part in enumerate(parts, start=1):
        if part.strip():
            pages.append(LoadedPage(text=part, page=index))
    if not pages:
        raise DocumentLoadError("empty markdown file: no content to index")
    return pages


# Map file_type -> loader callable.
LOADERS = {
    "pdf": load_pdf,
    "docx": load_docx,
    "txt": load_text,
    "md": load_markdown,
}


def load_file(file_type: str, data: bytes) -> list[LoadedPage]:
    """Dispatch to the correct loader for ``file_type`` and return its pages.

    Args:
        file_type: One of ``pdf``, ``docx``, ``txt``, ``md``.
        data: Raw file bytes.

    Returns:
        Ordered list of :class:`LoadedPage` segments.

    Raises:
        DocumentLoadError: if the type is unsupported or extraction fails.
    """
    loader = LOADERS.get(file_type)
    if loader is None:
        raise DocumentLoadError(f"unsupported file type: {file_type!r}")
    return loader(data)
