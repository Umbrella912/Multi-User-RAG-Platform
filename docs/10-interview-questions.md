# 10 Interview Questions About This Project

Short model answers for the most likely interview questions about the
Multi-User RAG Platform. Each answer is a tight summary — expand with concrete
examples from the codebase when asked.

---

### 1. What is this project, and what problem does it solve?

**Answer:** It is a multi-user Retrieval-Augmented Generation platform.
Users authenticate, curate private knowledge bases, upload documents, and ask
questions that are answered **only from their own documents**, with citations
back to the exact source passages. It solves the "private-data RAG" problem:
grounding LLM answers in a user's documents while guaranteeing that user A
can never retrieve user B's data — enforced at the API, ORM, and vector-store
layers.

### 2. Walk me through the retrieval pipeline.

**Answer:** The pipeline is **rewrite → vector search → BM25 → RRF fusion →
rerank → similarity gate → prompt → generate → extract citations**. A query is
optionally rewritten with chat history, embedded with a BGE model and searched
in Chroma (vector); simultaneously BM25 ranks chunks by keyword. The two
ranked lists are merged with Reciprocal Rank Fusion, an optional cross-encoder
reranks the fused pool, and a similarity gate drops low-confidence queries so
the LLM is never called for out-of-scope questions. Retrieved context becomes
the prompt, and the model's `[file, p.X]` citations are parsed and resolved to
the retrieved chunks.

### 3. How do you guarantee data isolation between users?

**Answer:** Defense in depth, three layers. **API/ORM:** every route resolves
the current user and every query filters by `user_id`, returning 404 (not 403)
so the existence of another user's resources never leaks. **Vector store:**
every Chroma query and BM25 index build carries a mandatory `user_id` filter,
so a leaked query can never fetch another user's chunks. **Storage:** local
keys are built from `{user_id}/{document_id}/{safe_filename}` with
sanitization and path-traversal checks. There is a dedicated security test
suite (`tests/test_security_isolation.py`) covering cross-user access, JWT
hardening, storage traversal, and retrieval scoping.

### 4. Why hybrid retrieval instead of just vector search?

**Answer:** Vector search is great at semantic matching but weak on exact
terms — product names, IDs, acronyms, code, numbers — which BM25 handles
precisely. Running both and fusing with RRF gets the best of both without
needing to tune score normalization, because RRF only uses *rank positions*,
not raw score magnitudes. A cross-encoder rerank then sharpens the final
ordering. The config comparator (`evaluation/compare_configs.py`) measures
the trade-off empirically: vector-only vs hybrid vs hybrid+rerank across
chunk sizes.

### 5. How do you handle the similarity gate, and what happens on a rejected query?

**Answer:** After retrieval, the best score is compared against
`SIMILARITY_THRESHOLD` (0.35). With reranking on, the signal is the
sigmoid-normalized cross-encoder score; with it off, it is the best vector
cosine similarity. If the score is below the threshold, the pipeline returns a
fixed *"I couldn't find this in your documents."* message with
`skipped=True` and **never calls the LLM** — saving cost and preventing
hallucination on out-of-scope questions. Tuning guidance: lower the threshold
if recall is too low, raise it if there are irrelevant answers.

### 6. How is the RAG pipeline tested without expensive dependencies?

**Answer:** The pipeline is built against two interfaces — `LLMProvider` and
`Retriever`/`VectorStore` — so tests inject fakes. Unit tests cover chunking,
loaders, retrieval primitives (RRF, the gate), retry/backoff and prompts with
no heavy deps. Integration tests (`tests/test_integration_e2e.py`) run the
**full user journey** — register → KB → upload → ingest → ask → persisted
messages — against a real PostgreSQL database with an in-memory vector store,
fake embeddings and a fake LLM, so they're fast, offline and deterministic.
They auto-skip when no test database is reachable, and CI provides one.

### 7. How do you evaluate RAG quality?

**Answer:** With RAGAS metrics and a retrieval comparator, both in
`evaluation/`. `build_dataset.py` creates 25 Q/A/ground-truth triples from
sample PDFs (LLM-generated questions with a template fallback).
`run_ragas.py` runs the four RAGAS metrics — **faithfulness** (is the answer
grounded in context?), **answer relevancy** (does it address the question?),
**context precision** (are retrieved chunks relevant?), **context recall**
(do the retrieved chunks cover the ground truth?) — and prints a per-question
plus mean table. `compare_configs.py` sweeps retrieval modes × chunk sizes and
reports recall@k, MRR, nDCG@10 and latency for a README-ready table.

### 8. How does the system stream answers to the frontend?

**Answer:** Over **Server-Sent Events** from `POST
/api/chat/sessions/{id}/ask`. The pipeline yields events as they happen:
`rewrite` (the rewritten query), `citation` (one per relevant context chunk),
`token` (streaming answer text), and a final `done` carrying citations,
retrieved chunks, latency and token usage. The Streamlit frontend consumes
the stream with `httpx` streaming, re-authenticates once on a 401 (refresh
token rotation), and renders the tokens progressively with citation chips. A
non-streaming `ask-sync` endpoint exists for simple clients and testing.

### 9. How is the project deployed, and how would you scale it?

**Answer:** Everything is containerized: multi-stage, non-root Docker images;
Docker Compose for dev (with healthchecks) and a prod override (restart
policies, resource limits, JSON logs, S3 uploads, required `JWT_SECRET`); an
Nginx HTTPS reverse proxy routing `/` to Streamlit and `/api` to FastAPI;
GitHub Actions CI (ruff, mypy, pytest, docker build) and CD (build/push to
ECR, SSH-deploy to EC2). Scaling is staged: **RDS PostgreSQL** for managed
backups/HA, **ElastiCache Redis** for shared rate limiting (the
`RateLimiter` interface is already abstracted), **ECS/Fargate** for
multi-instance compute behind a load balancer, and a **managed vector store**
(e.g. Bedrock Knowledge Bases) when the Chroma index grows. The AWS guide
includes cost estimates and triggers for each step.

### 10. What are the trade-offs and limitations you'd call out?

**Answer:** Three main ones. **Single-instance data stores** — Postgres and
Chroma run as containers on one host for simplicity; HA needs RDS and a
managed vector store. **In-memory rate limiting** — process-local, so
multi-worker deployments must swap in a Redis-backed `RateLimiter` (interface
ready, no call-site changes). **In-process ingestion** — background tasks are
fine for this scale; very large or mission-critical workloads would move to a
task queue (Celery/Redis). Additionally the embedding model is English-tuned,
and scanned PDFs without a text layer are rejected with a clear error rather
than OCR'd — both are deliberate, documented limitations with planned paths.
