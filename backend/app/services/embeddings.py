"""Embedding provider: singleton BGE-small embedder with batching.

The default model is ``BAAI/bge-small-en-v1.5`` (384-dim). BGE models require a
query-instruction prefix for *query* embeddings (not for document chunks), so
:func:`embed_documents` embeds as-is while :func:`embed_queries` prefixes each
query with the required instruction. Embeddings are normalized to unit length.

``sentence-transformers`` is a heavy, optional dependency and is imported lazily
so the module imports cleanly without it; a clear error is raised on first use
when it is missing.
"""
from __future__ import annotations

from functools import lru_cache

from app.core.config import settings

# BGE's mandated instruction prefix for English queries.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class EmbeddingError(Exception):
    """Raised when the embedder cannot be initialized or run."""


@lru_cache(maxsize=1)
def _get_model() -> object:
    """Load (and cache) the sentence-transformers model singleton."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise EmbeddingError(
            "embeddings require the 'sentence-transformers' package; "
            "install the project extras (see requirements.txt)"
        ) from exc
    try:
        return SentenceTransformer(settings.EMBEDDING_MODEL)
    except Exception as exc:  # pragma: no cover - model download failures
        raise EmbeddingError(f"failed to load embedding model: {exc}") from exc


def embed_documents(
    texts: list[str], *, batch_size: int = 32
) -> list[list[float]]:
    """Embed document chunks (no prefix) in batches; returns normalized vectors."""
    if not texts:
        return []
    model = _get_model()
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        embeddings.extend(
            model.encode(  # type: ignore[attr-defined]
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )
    return [list(map(float, vector)) for vector in embeddings]


def embed_queries(
    queries: list[str], *, batch_size: int = 32
) -> list[list[float]]:
    """Embed search queries, prepending the BGE query instruction prefix."""
    if not queries:
        return []
    prefixed = [QUERY_INSTRUCTION + q for q in queries]
    return embed_documents(prefixed, batch_size=batch_size)
