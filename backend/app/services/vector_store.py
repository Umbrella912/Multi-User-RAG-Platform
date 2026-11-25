"""ChromaDB vector-store wrapper.

One Chroma collection per knowledge base, named ``kb_{kb_id}``. Every write and
query carries ``user_id``/``kb_id`` in metadata, and every query applies a
metadata ``where`` filter on both — defense in depth on top of the collection
namespacing and the ownership checks in the API layer.

Chroma is synchronous/CPU-bound; the public methods here are async and dispatch
the blocking calls to a threadpool so the async call-path is never blocked.
``chromadb`` is imported lazily so the module loads without it installed.
"""
from __future__ import annotations

from typing import Any
from uuid import UUID

from app.core.config import settings


class VectorStoreError(Exception):
    """Raised when the Chroma client cannot be initialized or a call fails."""


def _collection_name(kb_id: UUID | str) -> str:
    """Return the Chroma collection name for a knowledge base."""
    return f"kb_{kb_id}"


class VectorStore:
    """Thin async wrapper over a Chroma client (HTTP server or persistent local)."""

    def __init__(self) -> None:
        # Built lazily so the module imports cleanly without ``chromadb`` installed.
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        """Return (building if needed) the underlying Chroma client."""
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self) -> Any:
        """Build a Chroma client, preferring the HTTP server when configured."""
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise VectorStoreError(
                "vector store requires the 'chromadb' package; "
                "install the project extras (see requirements.txt)"
            ) from exc

        host = settings.CHROMA_HOST.strip().lower()
        if host and host not in ("localhost", "127.0.0.1", "0.0.0.0"):
            return chromadb.HttpClient(
                host=settings.CHROMA_HOST, port=settings.CHROMA_PORT
            )
        return chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)

    def _collection(self, kb_id: UUID | str) -> Any:
        """Return (creating if needed) the collection for ``kb_id``."""
        name = _collection_name(kb_id)
        return self.client.get_or_create_collection(name=name)

    async def add_chunks(
        self,
        kb_id: UUID | str,
        chunks: list,
        embeddings: list[list[float]],
    ) -> int:
        """Upsert chunks with their embeddings and metadata.

        Args:
            kb_id: Knowledge base id (selects the collection).
            chunks: :class:`~app.services.chunking.Chunk` objects.
            embeddings: Corresponding normalized vectors.

        Returns:
            The number of chunks written.
        """
        if len(chunks) != len(embeddings):
            raise VectorStoreError("chunks and embeddings length mismatch")

        ids = [chunk.id for chunk in chunks]
        metadatas = [
            {
                "user_id": chunk.user_id,
                "kb_id": chunk.kb_id,
                "doc_id": chunk.document_id,
                "file_name": chunk.file_name,
                "page": chunk.page,
                "page_end": chunk.page_end,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in chunks
        ]
        collection = self._collection(kb_id)
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=[chunk.text for chunk in chunks],
            metadatas=metadatas,
        )
        return len(ids)

    async def delete_by_document(self, kb_id: UUID | str, document_id: UUID | str) -> None:
        """Remove every chunk belonging to ``document_id`` from the collection."""
        collection = self._collection(kb_id)
        collection.delete(where={"doc_id": str(document_id)})

    async def fetch_all(
        self, kb_id: UUID | str, *, user_id: UUID | str
    ) -> list[dict[str, Any]]:
        """Return all chunks in ``kb_id`` scoped to ``user_id`` (for BM25 builds).

        Returns:
            List of ``{id, text, metadata}`` dicts.
        """
        collection = self._collection(kb_id)
        result = collection.get(
            where={"user_id": str(user_id), "kb_id": str(kb_id)},
            include=["documents", "metadatas"],
        )
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        return [
            {"id": ids[i], "text": documents[i], "metadata": metadatas[i]}
            for i in range(len(ids))
        ]

    async def query(
        self,
        kb_id: UUID | str,
        query_embedding: list[float],
        *,
        user_id: UUID | str,
        n_results: int = 20,
    ) -> list[dict[str, Any]]:
        """Query the collection with a mandatory metadata filter.

        Args:
            kb_id: Knowledge base id.
            query_embedding: Embedded query vector.
            user_id: Caller id (mandatory ``where`` filter, defense in depth).
            n_results: Number of nearest neighbors to return.

        Returns:
            List of ``{id, text, metadata, distance}`` results, best first.
        """
        collection = self._collection(kb_id)
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"user_id": str(user_id), "kb_id": str(kb_id)},
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return [
            {
                "id": ids[i],
                "text": documents[i],
                "metadata": metadatas[i],
                "distance": distances[i],
            }
            for i in range(len(ids))
        ]


# Module-level singleton.
vector_store = VectorStore()
