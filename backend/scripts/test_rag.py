#!/usr/bin/env python
"""CLI to exercise the RAG pipeline without the API.

Examples:
    # Simple query against a KB
    python scripts/test_rag.py --kb-id <uuid> --user-id <uuid> --query "What is the refund policy?"

    # With a conversation history (triggers query rewriting)
    python scripts/test_rag.py --kb-id <uuid> --user-id <uuid> \
        --query "when does it apply?" \
        --history "user: What is the refund policy? assistant: It is 30 days."

    # Override the LLM provider and knobs
    python scripts/test_rag.py --provider ollama --model llama3.1 \
        --top-k-vector 10 --top-k-bm25 10 --rrf-k 60 --threshold 0.3

The command prints the rewritten query, retrieval stats, the answer, and the
parsed citations. It does NOT require the API server; it only needs a reachable
Chroma + embedding model (and the chosen LLM provider).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Allow running directly: `python scripts/test_rag.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.services.llm import LLMError, get_provider  # noqa: E402
from app.services.rag_pipeline import RAGPipeline  # noqa: E402
from app.services.retrieval import Retriever  # noqa: E402


def _parse_history(raw: str | None) -> list[dict[str, str]]:
    """Parse ``--history`` as alternating ``role: text`` lines."""
    if not raw:
        return []
    history: list[dict[str, str]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        role, _, content = line.partition(":")
        role = role.strip().lower()
        if role not in ("user", "assistant", "system"):
            continue
        history.append({"role": role, "content": content.strip()})
    return history


async def _run(args: argparse.Namespace) -> int:
    """Build the pipeline and run a single RAG query."""
    provider = get_provider(args.provider)
    retriever = Retriever()
    pipeline = RAGPipeline(provider, retriever)

    history = _parse_history(args.history)

    result = await pipeline.answer(
        args.query,
        args.kb_id,
        args.user_id,
        history=history,
        use_history=args.use_history,
    )

    payload = {
        "query": result.query,
        "model": result.model,
        "latency_ms": result.latency_ms,
        "skipped": result.skipped,
        "chunks": [
            {"source": c.source, "file_name": c.file_name, "page": c.page, "score": c.score}
            for c in result.chunks
        ],
        "answer": result.answer,
        "citations": [c.__dict__ for c in result.citations],
    }

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print("=" * 60)
    print("REWRITTEN QUERY:", result.query)
    print("MODEL:", result.model)
    print("LATENCY:", result.latency_ms, "ms")
    print("RETRIEVED CHUNKS:", len(result.chunks))
    for chunk in result.chunks:
        print(
            f"  - [{chunk.source}] {chunk.file_name} p.{chunk.page}"
            f" score={chunk.score:.4f}"
        )
    print("-" * 60)
    print("ANSWER:")
    print(result.answer)
    print("-" * 60)
    print("CITATIONS:")
    if not result.citations:
        print("  (none)")
    for citation in result.citations:
        print(
            f"  - [{citation.file_name}, p.{citation.page}]"
            f" chunk={citation.chunk_id or 'unresolved'}"
        )
    print("=" * 60)
    if result.skipped:
        print("(retrieval gate short-circuited: no LLM call was made)")
    return 0


def main() -> int:
    """Parse arguments and run the async entrypoint."""
    parser = argparse.ArgumentParser(description="Test the RAG pipeline (no API).")
    parser.add_argument("--query", required=True, help="Question to answer.")
    parser.add_argument("--kb-id", required=True, help="Knowledge base UUID.")
    parser.add_argument("--user-id", required=True, help="User UUID (isolation filter).")
    parser.add_argument("--history", default=None, help="'role: text' lines (optional).")
    parser.add_argument(
        "--use-history",
        type=lambda v: str(v).lower() in ("1", "true", "yes"),
        default=True,
        help="Rewrite the query using history (default: true).",
    )
    parser.add_argument("--provider", default=None, help="LLM provider (groq/gemini/ollama).")
    parser.add_argument("--model", default=None, help="Model name override.")
    parser.add_argument("--top-k-vector", type=int, default=settings.TOP_K_VECTOR)
    parser.add_argument("--top-k-bm25", type=int, default=settings.TOP_K_BM25)
    parser.add_argument("--rrf-k", type=int, default=settings.RRF_K)
    parser.add_argument("--threshold", type=float, default=settings.SIMILARITY_THRESHOLD)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = parser.parse_args()

    # Apply CLI overrides to settings so the retriever picks them up.
    settings.TOP_K_VECTOR = args.top_k_vector
    settings.TOP_K_BM25 = args.top_k_bm25
    settings.RRF_K = args.rrf_k
    settings.SIMILARITY_THRESHOLD = args.threshold

    try:
        result = asyncio.run(_run(args))
    except LLMError as exc:
        print(f"LLM error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - CLI surface
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return result


if __name__ == "__main__":
    raise SystemExit(main())
