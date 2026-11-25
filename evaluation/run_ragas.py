"""Run RAGAS evaluation (faithfulness, answer_relevancy, context_precision,
context_recall) against the RAG pipeline and print a results table.

For every triple in a dataset (see ``build_dataset.py``), the script:

1. Retrieves contexts for the question with the pipeline's :class:`Retriever`
   (vector + BM25 + RRF, optional rerank).
2. Generates an answer with the configured LLM via :class:`RAGPipeline`.
3. Runs RAGAS metrics (faithfulness, answer_relevancy, context_precision,
   context_recall) on the ``question / answer / contexts / ground_truths``
   rows and prints a per-question + summary table.

Optionally rebuilds a fresh evaluation knowledge base from a sample PDF
directory first (``--pdf-dir``). Requires a running Chroma instance and an
LLM provider with API keys; missing packages fail with a clear message.

Usage::

    # (optional) rebuild the eval KB from sample PDFs first
    python evaluation/run_ragas.py \
        --dataset evaluation/datasets/sample_qa.json \
        --pdf-dir evaluation/datasets/sample_pdfs \
        --kb-name eval-kb \
        --out evaluation/results_ragas.md \
        --json-out evaluation/results_ragas.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

# Make the backend importable when this script is run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import settings  # noqa: E402
from app.core.logging import get_logger  # noqa: E402
from app.services import embeddings, loaders, vector_store  # noqa: E402
from app.services.chunking import RecursiveCharacterTextSplitter  # noqa: E402
from app.services.llm import get_provider  # noqa: E402
from app.services.rag_pipeline import RAGPipeline  # noqa: E402
from app.services.retrieval import Retriever  # noqa: E402

logger = get_logger(__name__)

#: Metric names reported in the results table.
METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def _require_ragas() -> tuple[Any, Any, Any]:
    """Import and return ``(evaluate, Dataset, metrics)`` from ragas.

    Raises:
        RuntimeError: with installation instructions when ragas is missing.
    """
    try:
        from ragas import evaluate
        from ragas.datasets import Dataset
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        raise RuntimeError(
            "ragas is not installed; install it with:\n"
            "  pip install ragas==0.2.12\n"
            "(ragas pulls a local LLM for metric computation)"
        ) from exc
    metrics = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }
    return evaluate, Dataset, metrics


def load_dataset(dataset_path: Path) -> list[dict[str, Any]]:
    """Load and validate a dataset JSON produced by ``build_dataset.py``."""
    data = json.loads(dataset_path.read_text())
    triples = data["triples"] if isinstance(data, dict) else data
    if not triples:
        raise ValueError(f"dataset {dataset_path} contains no triples")
    for i, triple in enumerate(triples):
        for key in ("question", "answer", "contexts", "ground_truths"):
            if key not in triple:
                raise ValueError(f"triple {i} is missing required key '{key}'")
    return triples


async def rebuild_kb(
    kb_name: str,
    pdf_dir: Path,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> tuple[UUID, int]:
    """(Re)build a fresh evaluation knowledge base from sample PDFs.

    Deletes any existing collection for ``kb_name``, loads every supported
    file in ``pdf_dir``, chunks it, embeds the chunks and stores them in the
    configured Chroma instance, scoped to a fresh synthetic user id.

    Args:
        kb_name: Name of the evaluation knowledge base (also the collection).
        pdf_dir: Directory of sample files to ingest.
        chunk_size: Chunk size (defaults to ``settings.CHUNK_SIZE``).
        chunk_overlap: Chunk overlap (defaults to ``settings.CHUNK_OVERLAP``).

    Returns:
        ``(kb_id, chunk_count)`` for the rebuilt knowledge base.

    Raises:
        RuntimeError: if Chroma is unreachable or a heavy dependency is missing.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    client = vector_store.vector_store.client()
    from app.services.vector_store import _collection_name

    try:
        client.delete_collection(name=_collection_name(kb_name))
    except Exception:
        # Collection absent is fine; any other error will surface on add.
        pass

    kb_id = UUID(uuid4().hex)  # stable synthetic id for this run
    user_id = UUID(uuid4().hex)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    chunk_count = 0
    for path in sorted(pdf_dir.iterdir()):
        if path.suffix.lower() not in {".pdf", ".txt", ".md", ".docx"}:
            continue
        try:
            pages = loaders.load_file(path.suffix.lower().lstrip("."), path.read_bytes())
        except loaders.DocumentLoadError as exc:
            logger.warning("skipped %s: %s", path.name, exc)
            continue
        for page in pages:
            chunks = splitter.split_text(page.text)
            if not chunks:
                continue
            vectors = embeddings.embed_documents(chunks)
            from app.services.chunking import Chunk

            chroma_chunks = [
                Chunk(
                    text=text,
                    chunk_index=i,
                    document_id=path.name,
                    kb_id=str(kb_id),
                    user_id=str(user_id),
                    file_name=path.name,
                    page=page.page,
                    page_end=page.page_end,
                )
                for i, text in enumerate(chunks)
            ]
            await vector_store.vector_store.add_chunks(kb_id, chroma_chunks, vectors)
            chunk_count += len(chroma_chunks)
    logger.info("rebuilt KB %s with %d chunk(s)", kb_name, chunk_count)
    return kb_id, chunk_count


async def answer_triple(
    pipeline: RAGPipeline,
    retriever: Retriever,
    question: str,
    kb_id: UUID | str,
    user_id: UUID | str,
) -> tuple[str, list[str]]:
    """Retrieve contexts and generate an answer for one question.

    Returns:
        ``(answer, contexts)`` — the generated answer and the retrieved
        passage texts (empty list when the similarity gate rejects).
    """
    result = await retriever.retrieve(question, kb_id, user_id)
    contexts = [c.text for c in result.chunks if c.text]
    answer = await pipeline.answer(question, kb_id, user_id)
    return answer.answer, contexts


async def run_evaluation(
    triples: list[dict[str, Any]],
    kb_id: UUID,
    user_id: UUID,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Retrieve + answer every triple, then run RAGAS metrics over the rows.

    Returns:
        One result dict per triple: ``id``, ``question``, ``answer``,
        ``retrieved_contexts``, plus the four RAGAS metric values.
    """
    evaluate, Dataset, metrics = _require_ragas()

    provider = get_provider()
    pipeline = RAGPipeline(provider)
    retriever = Retriever()

    rows: list[dict[str, Any]] = []
    for triple in triples[:limit] if limit else triples:
        started = time.perf_counter()
        answer, contexts = await answer_triple(
            pipeline, retriever, triple["question"], kb_id, user_id
        )
        rows.append(
            {
                "id": triple["id"],
                "question": triple["question"],
                "answer": answer,
                "contexts": contexts,
                "ground_truths": list(triple["ground_truths"]),
                "retrieval_ms": int((time.perf_counter() - started) * 1000),
            }
        )

    dataset = Dataset.from_list(rows)
    scored = evaluate(dataset)

    results: list[dict[str, Any]] = []
    for row, scored_row in zip(rows, scored, strict=False):
        entry = {
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "retrieved_contexts": row["contexts"],
            "retrieval_ms": row["retrieval_ms"],
        }
        for name in METRIC_NAMES:
            value = scored_row.get(name)
            entry[name] = None if value is None else round(float(value), 4)
        results.append(entry)
    return results


def print_table(results: list[dict[str, Any]]) -> None:
    """Print a markdown results table (per-question + mean row) to stdout."""
    header = ["#", "Question", *METRIC_NAMES]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in results:
        cells = [
            row["id"],
            row["question"][:70] + ("..." if len(row["question"]) > 70 else ""),
        ]
        cells.extend("—" if row[name] is None else f"{row[name]:.3f}" for name in METRIC_NAMES)
        lines.append("| " + " | ".join(cells) + " |")

    means = [
        "**mean**",
        "",
    ]
    for name in METRIC_NAMES:
        values = [r[name] for r in results if r[name] is not None]
        means.append(f"**{sum(values) / len(values):.3f}**" if values else "—")
    lines.append("| " + " | ".join(means) + " |")
    print("\n".join(lines))


def save_markdown(path: Path, results: list[dict[str, Any]]) -> None:
    """Write the results table to a markdown file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# RAGAS Evaluation Results",
        "",
        f"Dataset: `{path.parent.name}/sample_qa.json` — "
        f"{len(results)} question(s) evaluated against the pipeline.",
        "",
    ]
    header = ["#", "Question", *METRIC_NAMES]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in results:
        cells = [
            row["id"],
            row["question"][:70] + ("..." if len(row["question"]) > 70 else ""),
        ]
        cells.extend("—" if row[name] is None else f"{row[name]:.3f}" for name in METRIC_NAMES)
        lines.append("| " + " | ".join(cells) + " |")

    means = ["**mean**", ""]
    for name in METRIC_NAMES:
        values = [r[name] for r in results if r[name] is not None]
        means.append(f"**{sum(values) / len(values):.3f}**" if values else "—")
    lines.append("| " + " | ".join(means) + " |")
    path.write_text("\n".join(lines) + "\n")


def save_json(path: Path, results: list[dict[str, Any]]) -> None:
    """Write the full machine-readable results to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/datasets/sample_qa.json"),
        help="Path to the Q/A/ground-truth dataset JSON.",
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=None,
        help="Optional: rebuild the eval KB from these sample files first.",
    )
    parser.add_argument("--kb-name", default="eval-kb", help="Evaluation KB name.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Evaluate only the first N triples."
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable cross-encoder reranking (default: off for reproducibility).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("evaluation/results_ragas.md"),
        help="Where to write the markdown results table.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("evaluation/results_ragas.json"),
        help="Where to write the machine-readable results JSON.",
    )
    return parser.parse_args(argv)


async def async_main(args: argparse.Namespace) -> int:
    """Async entry point: rebuild KB (optional), evaluate, print and save."""
    triples = load_dataset(args.dataset)

    if args.rerank:
        settings.RERANK_ENABLED = True
    else:
        settings.RERANK_ENABLED = False

    user_id = UUID(uuid4().hex)
    if args.pdf_dir:
        if not args.pdf_dir.is_dir():
            print(f"error: sample directory not found: {args.pdf_dir}", file=sys.stderr)
            return 1
        kb_id, chunk_count = await rebuild_kb(args.kb_name, args.pdf_dir)
        print(f"Rebuilt evaluation KB with {chunk_count} chunk(s).")
    else:
        # No rebuild: assume the KB already exists under this name.
        kb_id = args.kb_name

    print(f"Running RAGAS over {len(triples)} triple(s) ...")
    results = await run_evaluation(triples, kb_id, user_id, limit=args.limit)

    print()
    print_table(results)
    save_markdown(args.out, results)
    save_json(args.json_out, results)
    print(f"\nWrote results to {args.out} and {args.json_out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
