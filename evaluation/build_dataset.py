"""Build an evaluation dataset of Q/A/ground-truth triples from sample PDFs.

Extracts text from every PDF (and txt/md/docx) in a sample directory, picks
the most answer-worthy sentences, and turns each into one evaluation triple::

    {
      "id": "q01",
      "question": "...",
      "answer": "...",              # ground-truth answer text
      "contexts": ["..."],          # ground-truth passages for RAGAS
      "ground_truths": ["..."],     # reference answer(s) for RAGAS
      "source": {"file_name": "...", "page": 3}
    }

Question generation uses the project's configured LLM provider when API keys
are available (``--method auto``); otherwise it falls back to a deterministic
keyword-template question so the dataset can still be built offline.

A fill-in-the-blank manual template (``template_qa.json``) is written next to
the dataset so you can hand-author additional triples in the same format.

Usage::

    python evaluation/build_dataset.py \
        --pdf-dir evaluation/datasets/sample_pdfs \
        --out evaluation/datasets/sample_qa.json \
        --template evaluation/datasets/template_qa.json \
        --count 25 --method auto
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# Make the backend importable when this script is run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.loaders import DocumentLoadError, load_file  # noqa: E402

#: Output schema version (bump when the triple format changes).
SCHEMA_VERSION = "1.0"

#: Sentences shorter than this are rarely answer-worthy; longer than this get
#: truncated into multiple candidates.
MIN_SENTENCE_CHARS = 30
MAX_SENTENCE_CHARS = 320

#: Common English stopwords (kept small; only used for keyword scoring).
_STOPWORDS = frozenset(
    """
    a an and are as at be by for from has have in is it its of on or that
    the to was were will with this these those they their them he she his
    her we our you your not no but if then so do does did can could should
    """.split()
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")

#: Default keyword count for template questions.
_TEMPLATE_KEYWORDS = 2


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentence candidates (blank lines act as boundaries)."""
    sentences: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        for raw in _SENTENCE_SPLIT.split(block):
            sentence = " ".join(raw.split())
            if MIN_SENTENCE_CHARS <= len(sentence) <= MAX_SENTENCE_CHARS:
                sentences.append(sentence)
    return sentences


def score_sentence(sentence: str) -> float:
    """Return a 0–1 answer-worthiness score for ``sentence``.

    Rewards information density (longer sentences with more content words)
    and penalizes sentences made mostly of stopwords.
    """
    words = _WORD_RE.findall(sentence.lower())
    if not words:
        return 0.0
    content = [w for w in words if w not in _STOPWORDS and len(w) > 2]
    if not content:
        return 0.0
    content_ratio = len(content) / len(words)
    length_bonus = min(len(sentence) / MAX_SENTENCE_CHARS, 1.0)
    return round(content_ratio * (0.4 + 0.6 * length_bonus), 4)


def extract_keywords(sentence: str, limit: int = _TEMPLATE_KEYWORDS) -> list[str]:
    """Return the top content-word keywords of ``sentence`` (by frequency)."""
    counts: dict[str, int] = {}
    for word in _WORD_RE.findall(sentence.lower()):
        if word in _STOPWORDS or len(word) <= 2:
            continue
        counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts, key=lambda w: (-counts[w], w))
    return ranked[:limit]


def template_question(file_name: str, sentence: str) -> str:
    """Build a deterministic question from a sentence (offline fallback)."""
    keywords = extract_keywords(sentence)
    topic = ", ".join(keywords) if keywords else "the main topic"
    return f"In {file_name}, what is stated about {topic}?"


def load_documents(pdf_dir: Path) -> list[dict[str, Any]]:
    """Load every supported file in ``pdf_dir`` into ``(file, page)`` segments.

    Raises:
        DocumentLoadError: if a file cannot be parsed (skipped with a warning
            instead of aborting the whole run).
    """
    documents: list[dict[str, Any]] = []
    files = sorted(p for p in pdf_dir.iterdir() if p.is_file())
    for path in files:
        if path.suffix.lower() not in {".pdf", ".txt", ".md", ".docx"}:
            continue
        try:
            pages = load_file(path.suffix.lower().lstrip("."), path.read_bytes())
        except DocumentLoadError as exc:
            print(f"  [skip] {path.name}: {exc}", file=sys.stderr)
            continue
        documents.append({"file_name": path.name, "pages": pages})
        print(f"  [ok]   {path.name}: {len(pages)} segment(s)")
    if not documents:
        raise DocumentLoadError(
            f"no loadable files found in {pdf_dir}; drop sample PDFs (or "
            "txt/md/docx) there and re-run"
        )
    return documents


def pick_candidates(
    documents: list[dict[str, Any]], count: int
) -> list[tuple[str, int, str]]:
    """Select ``count`` (file_name, page, sentence) candidates.

    Candidates are scored by answer-worthiness and spread across documents
    (round-robin) so the dataset is not dominated by a single file.
    """
    scored: list[tuple[float, str, int, str]] = []
    for doc in documents:
        for page in doc["pages"]:
            for sentence in split_sentences(page.text):
                scored.append((score_sentence(sentence), doc["file_name"], page.page, sentence))

    # De-duplicate identical sentences, then sort by score.
    seen: set[str] = set()
    unique: list[tuple[float, str, int, str]] = []
    for score, file_name, page, sentence in sorted(scored, key=lambda s: -s[0]):
        if sentence in seen:
            continue
        seen.add(sentence)
        unique.append((score, file_name, page, sentence))

    # Round-robin across files to spread the selection.
    by_file: dict[str, list[tuple[float, str, int, str]]] = {}
    for item in unique:
        by_file.setdefault(item[1], []).append(item)

    picked: list[tuple[str, int, str]] = []
    while len(picked) < count and any(by_file.values()):
        for file_name in list(by_file):
            bucket = by_file[file_name]
            if bucket:
                _, _, page, sentence = bucket.pop(0)
                picked.append((file_name, page, sentence))
            if len(picked) >= count:
                break
    return picked


def llm_question(file_name: str, sentence: str) -> str | None:
    """Ask the configured LLM to turn ``sentence`` into a natural question.

    Returns ``None`` when no LLM provider can be used (missing package or
    missing API key); the caller falls back to the template question.
    """
    try:
        from app.services.llm import get_provider
    except Exception:
        return None
    try:
        provider = get_provider()
        prompt = (
            "You are building an evaluation dataset. Given a sentence from a "
            "document, write one clear, natural question whose complete answer "
            "is exactly the sentence. Reply with the question only.\n\n"
            f"Sentence: {sentence}\nQuestion:"
        )
        result = provider.generate_with_usage([{"role": "user", "content": prompt}])
        question = result.text.strip().strip('"')
        return question or None
    except Exception as exc:  # pragma: no cover - depends on runtime config
        print(f"  [warn] LLM question generation unavailable: {exc}", file=sys.stderr)
        return None


def build_triples(
    documents: list[dict[str, Any]],
    count: int,
    method: str,
) -> list[dict[str, Any]]:
    """Build ``count`` Q/A/ground-truth triples from the loaded documents.

    Args:
        documents: Output of :func:`load_documents`.
        count: Number of triples to build.
        method: ``"llm"`` (LLM-generated questions), ``"template"`` (offline
            template questions), or ``"auto"`` (LLM when available, template
            otherwise).

    Returns:
        A list of triple dicts matching the output schema.
    """
    candidates = pick_candidates(documents, count)
    triples: list[dict[str, Any]] = []
    for index, (file_name, page, sentence) in enumerate(candidates, start=1):
        question: str | None = None
        if method in {"auto", "llm"}:
            question = llm_question(file_name, sentence)
        if question is None:
            if method == "llm":
                print(
                    "  [warn] LLM unavailable; falling back to a template "
                    "question for this triple.",
                    file=sys.stderr,
                )
            question = template_question(file_name, sentence)

        triples.append(
            {
                "id": f"q{index:02d}",
                "question": question,
                "answer": sentence,
                "contexts": [sentence],
                "ground_truths": [sentence],
                "source": {"file_name": file_name, "page": page},
            }
        )
    return triples


def write_template(template_path: Path, example: dict[str, Any], count: int) -> None:
    """Write the fill-in-the-blank manual template JSON file."""
    template = {
        "schema_version": SCHEMA_VERSION,
        "description": (
            "Manual Q/A/ground-truth template. Fill each triple in the "
            "'triples' list. 'contexts' are the passages that should be "
            "retrieved for the question (used by RAGAS context metrics); "
            "'ground_truths' are the reference answers (used by RAGAS "
            "context_recall and answer_relevancy)."
        ),
        "example": example,
        "triples": [
            {
                "id": f"q{i:02d}",
                "question": "",
                "answer": "",
                "contexts": [],
                "ground_truths": [],
                "source": {"file_name": "", "page": 0},
            }
            for i in range(1, count + 1)
        ],
    }
    template_path.write_text(json.dumps(template, indent=2) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("evaluation/datasets/sample_pdfs"),
        help="Directory of sample PDFs (or txt/md/docx) to build the dataset from.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("evaluation/datasets/sample_qa.json"),
        help="Where to write the generated dataset.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("evaluation/datasets/template_qa.json"),
        help="Where to write the manual template JSON.",
    )
    parser.add_argument("--count", type=int, default=25, help="Number of triples to build.")
    parser.add_argument(
        "--method",
        choices=["auto", "llm", "template"],
        default="auto",
        help="Question generation method (default: auto = LLM if available).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point: build and write the dataset + manual template."""
    args = parse_args(argv)
    if not args.pdf_dir.is_dir():
        print(f"error: sample directory not found: {args.pdf_dir}", file=sys.stderr)
        return 1

    print(f"Loading sample files from {args.pdf_dir} ...")
    documents = load_documents(args.pdf_dir)
    total_segments = sum(len(d["pages"]) for d in documents)
    print(f"Loaded {len(documents)} file(s), {total_segments} segment(s).")

    triples = build_triples(documents, args.count, args.method)
    dataset = {
        "schema_version": SCHEMA_VERSION,
        "source": str(args.pdf_dir),
        "method": args.method,
        "triples": triples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dataset, indent=2) + "\n")
    if triples:
        write_template(args.template, triples[0], args.count)

    print(f"Wrote {len(triples)} triples to {args.out}")
    print(f"Wrote manual template to {args.template}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
