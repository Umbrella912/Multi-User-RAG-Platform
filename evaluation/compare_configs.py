"""Compare retrieval configurations and output a results table for the README.

Sweeps two axes and reports retrieval quality for each combination:

* **Retrieval mode**: ``vector-only`` (no BM25), ``hybrid`` (vector + BM25 +
  RRF, no rerank), ``hybrid+rerank`` (hybrid + cross-encoder reranking).
* **Chunk size**: ``400``, ``800`` (default), ``1200`` characters.

For every configuration the sample PDFs are (re)ingested with that chunk size
into a fresh Chroma collection, every dataset question is retrieved, and the
results are scored against the dataset ground truths:

* **Recall@k** — fraction of questions whose ground truth appears in the top-k
  retrieved chunks.
* **MRR** — mean reciprocal rank of the first relevant chunk.
* **nDCG@10** — normalized discounted cumulative gain over the top-10 chunks.
* **Avg latency** — mean retrieval time in milliseconds.

Output is a markdown table (and a JSON dump) ready to paste into the README.

Usage::

    python evaluation/compare_configs.py \
        --pdf-dir evaluation/datasets/sample_pdfs \
        --dataset evaluation/datasets/sample_qa.json \
        --out evaluation/results_compare.md \
        --json-out evaluation/results_compare.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
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
from app.services.chunking import Chunk, RecursiveCharacterTextSplitter  # noqa: E402
from app.services.retrieval import Retriever  # noqa: E402

logger = get_logger(__name__)

#: Chunk sizes evaluated (characters).
CHUNK_SIZES: tuple[int, ...] = (400, 800, 1200)

#: Retrieval modes evaluated.
MODES: tuple[tuple[str, bool, int], ...] = (
    ("vector-only", False, 0),  # (name, rerank, top_k_bm25)
    ("hybrid", False, 20),
    ("hybrid+rerank", True, 20),
)

#: Metrics columns in the output table.
METRIC_COLUMNS = ("Recall@5", "Recall@10", "MRR", "nDCG@10", "Latency (ms)")

_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Collapse whitespace and lowercase for ground-truth matching."""
    return _WS_RE.sub(" ", text).strip().lower()


def is_relevant(chunk_text: str, ground_truths: list[str]) -> bool:
    """Return ``True`` if any ground-truth sentence appears in ``chunk_text``."""
    chunk = _normalize(chunk_text)
    return any(_normalize(gt) in chunk for gt in ground_truths if gt)


def recall_at_k(scored: list[bool], k: int) -> float:
    """Fraction of questions with at least one relevant chunk in the top-k."""
    if not scored:
        return 0.0
    hits = sum(1 for s in scored[:k] if s)
    return hits / len(scored)


def mean_reciprocal_rank(ranks: list[int]) -> float:
    """Mean of ``1 / rank`` over questions (0 when nothing is relevant)."""
    if not ranks:
        return 0.0
    return sum(1.0 / r for r in ranks if r > 0) / len(ranks)


def ndcg_at_k(ranked_relevant: list[bool], k: int = 10) -> float:
    """Normalized DCG over the top-k binary relevance list."""
    if not ranked_relevant:
        return 0.0
    dcg = sum(
        (1.0 if rel else 0.0) / math.log2(index + 2)
        for index, rel in enumerate(ranked_relevant[:k])
    )
    ideal = sorted(ranked_relevant, reverse=True)[:k]
    idcg = sum(rel / math.log2(index + 2) for index, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def load_dataset(dataset_path: Path) -> list[dict[str, Any]]:
    """Load the Q/A/ground-truth dataset produced by ``build_dataset.py``."""
    data = json.loads(dataset_path.read_text())
    triples = data["triples"] if isinstance(data, dict) else data
    if not triples:
        raise ValueError(f"dataset {dataset_path} contains no triples")
    return triples


async def rebuild_kb(
    kb_name: str,
    pdf_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    user_id: UUID,
) -> tuple[UUID, int]:
    """Rebuild a fresh KB collection from ``pdf_dir`` with the given chunking.

    Deletes any existing collection for ``kb_name`` first so each config
    starts from a clean index.

    Returns:
        ``(kb_id, chunk_count)`` for the rebuilt knowledge base.
    """
    client = vector_store.vector_store.client()
    from app.services.vector_store import _collection_name

    try:
        client.delete_collection(name=_collection_name(kb_name))
    except Exception:
        # Absent collection is expected on a fresh run.
        pass

    kb_id = UUID(uuid4().hex)
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
            texts = splitter.split_text(page.text)
            if not texts:
                continue
            vectors = embeddings.embed_documents(texts)
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
                for i, text in enumerate(texts)
            ]
            await vector_store.vector_store.add_chunks(kb_id, chroma_chunks, vectors)
            chunk_count += len(chroma_chunks)
    return kb_id, chunk_count


async def evaluate_config(
    mode_name: str,
    rerank: bool,
    top_k_bm25: int,
    chunk_size: int,
    pdf_dir: Path,
    triples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run one configuration and return its metric row."""
    logger.info(
        "config %s | chunk_size=%d | rerank=%s | bm25=%d",
        mode_name,
        chunk_size,
        rerank,
        top_k_bm25,
    )
    user_id = UUID(uuid4().hex)
    kb_id, chunk_count = await rebuild_kb(
        f"compare-{mode_name}-{chunk_size}",
        pdf_dir,
        chunk_size,
        chunk_size // 5,
        user_id,
    )

    settings.RERANK_ENABLED = rerank
    retriever = Retriever()

    recall5: list[bool] = []
    recall10: list[bool] = []
    ranks: list[int] = []
    ndcgs: list[float] = []
    latencies: list[int] = []

    for triple in triples:
        ground_truths = list(triple.get("ground_truths") or [triple.get("answer", "")])
        started = time.perf_counter()
        result = await retriever.retrieve(
            triple["question"],
            kb_id,
            user_id,
            top_k_bm25=top_k_bm25,
            context_chunks=10,
        )
        latencies.append(int((time.perf_counter() - started) * 1000))

        ranked = [c.text for c in result.chunks if c.text]
        relevant = [is_relevant(text, ground_truths) for text in ranked]
        recall5.append(any(relevant[:5]))
        recall10.append(any(relevant[:10]))
        first = next((i + 1 for i, rel in enumerate(relevant) if rel), 0)
        ranks.append(first if first else 0)
        ndcgs.append(ndcg_at_k(relevant, k=10))

    latency_ms = sum(latencies) / len(latencies) if latencies else 0.0
    return {
        "mode": mode_name,
        "rerank": rerank,
        "top_k_bm25": top_k_bm25,
        "chunk_size": chunk_size,
        "chunks_indexed": chunk_count,
        "recall_at_5": round(recall_at_k(recall5, 5), 4),
        "recall_at_10": round(recall_at_k(recall10, 10), 4),
        "mrr": round(mean_reciprocal_rank(ranks), 4),
        "ndcg_at_10": round(sum(ndcgs) / len(ndcgs), 4) if ndcgs else 0.0,
        "avg_latency_ms": round(latency_ms, 1),
    }


def render_markdown(rows: list[dict[str, Any]]) -> str:
    """Render the results as a markdown table (README-ready)."""
    lines = [
        "# Retrieval Configuration Comparison",
        "",
        f"Dataset: `{Path('evaluation/datasets/sample_qa.json').name}` — "
        f"{len(rows) // 3} configuration(s) × 3 chunk sizes.",
        "",
        "| Config | Chunk Size | " + " | ".join(METRIC_COLUMNS) + " |",
        "|" + "|".join(["---"] * (2 + len(METRIC_COLUMNS))) + "|",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['chunk_size']} | "
            f"{row['recall_at_5']:.3f} | {row['recall_at_10']:.3f} | "
            f"{row['mrr']:.3f} | {row['ndcg_at_10']:.3f} | "
            f"{row['avg_latency_ms']:.0f} |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("evaluation/datasets/sample_pdfs"),
        help="Directory of sample files to build the KB from.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/datasets/sample_qa.json"),
        help="Path to the Q/A/ground-truth dataset JSON.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("evaluation/results_compare.md"),
        help="Where to write the markdown results table.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("evaluation/results_compare.json"),
        help="Where to write the machine-readable results JSON.",
    )
    return parser.parse_args(argv)


async def async_main(args: argparse.Namespace) -> int:
    """Async entry point: sweep the config matrix, score, print and save."""
    if not args.pdf_dir.is_dir():
        print(f"error: sample directory not found: {args.pdf_dir}", file=sys.stderr)
        return 1
    triples = load_dataset(args.dataset)
    print(f"Sweeping {len(MODES)} retrieval modes × {len(CHUNK_SIZES)} chunk sizes ...")

    rows: list[dict[str, Any]] = []
    for chunk_size in CHUNK_SIZES:
        for mode_name, rerank, top_k_bm25 in MODES:
            row = await evaluate_config(
                mode_name, rerank, top_k_bm25, chunk_size, args.pdf_dir, triples
            )
            rows.append(row)

    table = render_markdown(rows)
    print()
    print(table)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(table)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Wrote results to {args.out} and {args.json_out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
