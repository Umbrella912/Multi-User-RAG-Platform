# Phase 4 — Retrieval & Answer Generation

Retrieval pipeline: **rewrite → vector search → BM25 → RRF fusion → rerank →
similarity gate → prompt → generate → extract citations**.

## Chosen defaults and why

### `TOP_K_VECTOR = 20` / `TOP_K_BM25 = 20`
Each retriever returns its top 20 hits. 20 is enough to capture relevant
passages for most knowledge bases without pulling in too much noise. These are
the *per-list* depths fed into RRF.

**How to tune:**
- **Large / diverse KBs (hundreds of docs):** raise both toward **30–50** so
  the recall pool is broad enough to include the right passage. Cost: slightly
  slower retrieval and a wider rerank pool.
- **Small / focused KBs (a few docs):** lower to **10** to keep the rerank
  pool tight and reduce irrelevant chunks.
- If recall feels low (missing relevant answers), the lever is `TOP_K_VECTOR` /
  `TOP_K_BM25` **up**. If precision feels noisy (irrelevant context included),
  tune the **rerank + threshold** instead — don't just shrink these, since RRF
  needs enough candidates to merge well.

### `RRF_K = 60`
Reciprocal Rank Fusion scores each list as `Σ 1/(k + rank)`. The constant `k`
controls how much *rank position* matters versus the raw score. `k=60` is the
widely-used default (Cormack et al.): it makes fusion depend almost entirely on
**relative rank order**, not the absolute magnitudes of the two retrievers'
scores. This is the right choice here because vector cosine similarity and BM25
scores are on totally different scales — RRF only needs the rankings to be
correct.

**How to tune:**
- `k` **smaller** (e.g. 10–20) → top ranks dominate more; fusion becomes more
  sensitive to the #1 hit. Use when the top result is usually correct.
- `k` **larger** (e.g. 100+) → fusion is nearly a simple rank sum; positions
  matter less, results flatten. Use when top results are noisy and you want a
  smoother blend.
- In practice `60` works across the board; only tweak if you've measured that
  one retriever's top-1 is systematically wrong.

### `SIMILARITY_THRESHOLD = 0.35`
Gate: if the best score (cross-encoder score when reranking is on, else the
best vector cosine similarity) is below this, the pipeline returns
*"I couldn't find this in your documents."* and **skips the LLM call** — saving
cost and preventing hallucination on out-of-scope questions.

`0.35` is a deliberately moderate-low floor for the *normalized* signals used:
- With the **reranker on**, scores are sigmoid-normalized cross-encoder
  probabilities (0–1); 0.35 is just above "unrelated" while still admitting
  weakly-relevant passages.
- With the **reranker off**, scores are cosine similarity from normalized BGE
  embeddings (0–1); 0.35 admits genuinely related text while filtering
  tangential matches.

**How to tune:**
- **Too many "I couldn't find"** (false negatives / recall down): **lower** the
  threshold (e.g. **0.25–0.30**). The model gets more context and answers more,
  at the risk of slightly lower precision.
- **Too many irrelevant answers / hallucination** (false positives): **raise**
  the threshold (e.g. **0.40–0.50**). The gate rejects weak matches more
  aggressively.
- The threshold and the reranker interact: with the reranker **on**, the
  signal is sharper, so you can usually **raise** the threshold for the same
  recall. With it **off**, keep it lower (cosine similarity is a blunter
  signal).
- Tune by running the CLI against a sample of known-answer / known-irrelevant
  queries and sweeping the threshold; pick the value that maximizes the
  answer/no-answer split accuracy.

### Other defaults
- `CONTEXT_CHUNKS = 6` — how many chunks become the LLM context after reranking.
  Raise for multi-part questions, lower for focused Q&A (cost/latency tradeoff).
- `RERANK_TOP_N = 6` — kept as the post-rerank context size (aligned with
  `CONTEXT_CHUNKS`); the reranker scores a wider pool (~30) then keeps the top 6.
- `RERANK_ENABLED = True` — cross-encoder reranking is on by default for quality.
  Turn **off** (`RERANK_ENABLED=false`) for a latency-sensitive deployment or
  when CPU for the cross-encoder isn't available; then the gate falls back to
  vector similarity and the threshold should be set accordingly.
- `HISTORY_TURNS = 6` — messages folded into query rewriting.

## Tuning workflow
1. Build a small eval set of (query, expected-answer-or-"not-in-docs").
2. Sweep `SIMILARITY_THRESHOLD` first (biggest lever for the no-answer gate).
3. Then adjust `TOP_K_VECTOR`/`TOP_K_BM25` for recall.
4. Confirm `RRF_K=60` unless you have evidence the top-1 ranking is systematically
   off.
5. Use the CLI: `python scripts/test_rag.py --query ... --kb-id ... --user-id ...
   --threshold 0.3 --json` for quick, reproducible checks.
