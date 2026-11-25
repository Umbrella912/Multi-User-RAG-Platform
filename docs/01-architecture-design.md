# Multi-User RAG Platform — Architecture & Design Spec (Phase 1)

> Status: **Design (no code yet).** This document is the contract for Phases 2–9.
> Branch: `arena/99e1c768-multi-user-rag-platform`

---

## 1. Tech decisions (summary)

| Decision | Choice | Rationale |
|---|---|---|
| Python | 3.11 | Required; `tomllib`, faster, stable typing |
| Web framework | **FastAPI** + Uvicorn | Native async, SSE, OpenAPI, `httpx` TestClient |
| Validation | **Pydantic v2** (strict) | Performance + schema-first |
| ORM | **SQLAlchemy 2.0 async** + `asyncpg` | Typed `AsyncSession`, 2.0 style, migrations |
| Migrations | **Alembic** | Standard, scripted |
| Config | **pydantic-settings** | Typed env vars, `.env.example` |
| Auth | **JWT access + refresh**, **bcrypt** (work factor 12) | Stateless API, revocable refresh |
| Relational DB | **PostgreSQL 15+** (UUIDv4 PKs) | Required; JSONB for citations |
| Vector DB | **ChromaDB** (1 collection per KB) | Required; metadata-filtered by `user_id` |
| Embeddings | **BAAI/bge-small-en-v1.5** (384-dim, local) | Free, strong English retrieval |
| LLM | **Groq** `llama-3.3-70b-versatile` behind `LLMProvider` ABC | Switch via `LLM_PROVIDER=groq\|openai\|gemini\|ollama` |
| Ingestion | **FastAPI BackgroundTasks** wrapped in a `TaskBackend` interface | Swappable to Celery+Redis later |
| Retrieval | **Hybrid** = BM25 ∪ vector, merged with **RRF** (k=60), optional **cross-encoder** rerank | Best-in-class, cheap |
| Streaming | **SSE** (`text/event-stream`) FastAPI → Streamlit | Simple, native to `st.write_stream` |
| Frontend | **Streamlit** multipage | Required |
| Runtime | **Docker + docker-compose** (postgres, backend, frontend, chromadb) | Required |
| CI/CD | **GitHub Actions** (lint → test → build) | Required |
| Eval | **RAGAS** (faithfulness, answer_relevancy, context_recall) | Required |

**Async note:** The API, DB, and HTTP call paths are async. Chroma and `sentence-transformers` are **sync/CPU-bound**, so all calls to them go through `fastapi.concurrency.run_in_threadpool` (or a dedicated `ThreadPoolExecutor(max_workers=N)`). This keeps a single async model end-to-end while isolating blocking work.

---

## 2. Architecture diagram

### 2.1 Components

```
                        ┌──────────────────────────────────────────────┐
                        │                 STREAMLIT (Frontend)          │
                        │  Login · KBs · Upload · Chat · Settings       │
                        └───────────────────────┬──────────────────────┘
                                                │ HTTPS + Bearer JWT
                                                ▼
                        ┌──────────────────────────────────────────────┐
                        │             FASTAPI BACKEND (Uvicorn)         │
                        │  CORS · RateLimit · ExceptionHandlers · Logs  │
                        │  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │
                        │  │  /auth   │  │ /kb /doc  │  │ /chat /SSE  │  │
                        │  └────┬─────┘  └────┬─────┘  └──────┬──────┘  │
                        │       │             │              │         │
                        │  ┌────▼─────────────▼──────────────▼───────┐  │
                        │  │  SERVICES                              │  │
                        │  │ auth · kb · doc · ingestion · retrieval│  │
                        │  │ · chat · usage · storage · llm(provider)│ │
                        │  └────┬─────────────┬──────────────┬───────┘  │
                        └───────┼─────────────┼──────────────┼─────────┘
                                │             │              │
              ┌─────────────────┼─────────────┼──────────────┼─────────┐
              │                 │             │              │         │
              ▼                 ▼             ▼              ▼         │
        ┌──────────┐      ┌──────────┐  ┌──────────┐  ┌────────────┐  │
        │PostgreSQL│      │ ChromaDB │  │ Local/S3 │  │  Groq / LLM │  │
        │ (users,  │      │ kb_<id>  │  │ raw files│  │  (provider) │  │
        │  KB, doc,│      │ cols     │  │          │  │             │  │
        │ sessions,│      │ (vect+BM25│ │          │  │             │  │
        │ messages,│      │ metadata)│  │          │  │             │  │
        │ usage)   │      └──────────┘  └──────────┘  └─────┬──────┘  │
        └──────────┘                                        │         │
              ▲                                              │         │
              │                              ┌───────────────▼───────┐ │
              │                              │ bge-small (embedder)   │ │
              │                              │ CPU-bound, threadpool  │ │
              │                              └───────────────────────┘ │
              │                                                        │
        ┌─────┴──────────────────────────────────────────────────────┐│
        │  TASK BACKEND (BackgroundTasks now → Celery+Redis later)     ││
        │  ingestion pipeline: extract → chunk → embed → upsert        ││
        └─────────────────────────────────────────────────────────────┘│
```

### 2.2 Ingestion data flow

```
User ──POST /kb/{kb_id}/documents(multipart)
  1. Validate (ext, magic bytes, size)  → 400/413 if bad
  2. Persist raw file to storage (local / S3: {user_id}/{kb_id}/{doc_id}.ext)
  3. INSERT documents row (status=pending)            [PostgreSQL]
  4. enqueue ingest(kb_id, doc_id)  → BackgroundTask
        └─ pipeline.extract()  (PDF/DOCX/TXT/MD → text + page map)
        └─ pipeline.chunk()   (recursive splitter, page kept in metadata)
        └─ embedder.embed_chunks()  (bge-small, threadpool)
        └─ vector_store.upsert(collection=kb_<id>,
              embeddings, metas={user_id,kb_id,doc_id,file_name,page,chunk_idx})
        └─ UPDATE documents (status=ready, chunk_count, page_count)  or =failed
        └─ usage_log.insert(action=ingestion, ...)
```

### 2.3 Query data flow

```
User ──POST /sessions/{id}/chat/stream  (SSE)
  1. Auth + rate limit (per user)
  2. Load session + last N=6 messages
  3. query_rewriter.rewrite(history, question) → standalone_query   (LLM call)
  4. retrieval:
       a. vector: Chroma query(collection=kb_<id>, where={user_id,kb_id},
                               n=20)  cosine sim
       b. bm25:   BM25 over KB chunk texts, top 20
       c. RRF merge (k=60) → top 30
       d. [optional] cross-encoder rerank → top 5–8
  5. gate: if best score < SIM_THRESHOLD(0.35) → stream "I don't know", stop
  6. build prompt (context + citation instructions)  → stream LLM tokens (SSE)
  7. on finish: INSERT messages(user, assistant w/ citations),
                 usage_log.insert(tokens, latency, retrieved_chunks)
```

---

## 3. Final folder structure

```
Multi-User-RAG-Platform/
├─ docker-compose.yml
├─ .env.example
├─ README.md
├─ Makefile
├─ .github/
│  └─ workflows/
│     ├─ ci.yml            # lint + test + build backend image
│     └─ release.yml       # (later) push to ECR
├─ docs/
│  └─ 01-architecture-design.md
├─ backend/
│  ├─ pyproject.toml
│  ├─ requirements.txt
│  ├─ Dockerfile
│  ├─ alembic.ini
│  ├─ app/
│  │  ├─ main.py                      # FastAPI app factory, routers, middleware
│  │  ├─ core/
│  │  │  ├─ config.py                 # pydantic-settings (Settings)
│  │  │  ├─ security.py               # bcrypt, JWT create/verify, deps
│  │  │  ├─ database.py               # async engine, sessionmaker, Base
│  │  │  ├─ logging_config.py         # structlog / JSON formatter
│  │  │  ├─ exceptions.py             # global handlers, AppError hierarchy
│  │  │  ├─ rate_limit.py             # token-bucket per user
│  │  │  └─ threadpool.py             # run_in_threadpool helpers
│  │  ├─ models/                      # SQLAlchemy 2.0 ORM (async)
│  │  │  ├─ user.py
│  │  │  ├─ knowledge_base.py
│  │  │  ├─ document.py
│  │  │  ├─ chat_session.py
│  │  │  ├─ message.py
│  │  │  └─ usage_log.py
│  │  ├─ schemas/                     # Pydantic v2 request/response
│  │  │  ├─ user.py
│  │  │  ├─ knowledge_base.py
│  │  │  ├─ document.py
│  │  │  ├─ chat.py
│  │  │  └─ common.py
│  │  ├─ api/
│  │  │  ├─ deps.py                   # get_db, get_current_user, get_kb
│  │  │  └─ routers/
│  │  │     ├─ health.py
│  │  │     ├─ auth.py
│  │  │     ├─ knowledge_bases.py
│  │  │     ├─ documents.py
│  │  │     ├─ chat.py
│  │  │     └─ settings.py
│  │  ├─ services/
│  │  │  ├─ auth_service.py
│  │  │  ├─ kb_service.py
│  │  │  ├─ document_service.py
│  │  │  ├─ chat_service.py
│  │  │  ├─ usage_service.py
│  │  │  ├─ storage.py                # local + S3 backends (StorageBackend ABC)
│  │  │  ├─ ingestion/
│  │  │  │  ├─ pipeline.py            # orchestrates extract→chunk→embed→upsert
│  │  │  │  ├─ extractors.py          # pdf/docx/txt/md → (text, page)
│  │  │  │  ├─ chunker.py             # recursive splitter + page merge
│  │  │  │  ├─ embedder.py            # bge-small wrapper (singleton, threadpool)
│  │  │  │  └─ vector_store.py        # Chroma wrapper (per-KB collection)
│  │  │  ├─ retrieval/
│  │  │  │  ├─ retriever.py           # BM25 + vector + RRF
│  │  │  │  ├─ bm25.py                # per-KB BM25 index (lazy/cached)
│  │  │  │  ├─ reranker.py            # cross-encoder (optional)
│  │  │  │  └─ query_rewriter.py      # standalone query (LLM)
│  │  │  └─ llm/
│  │  │     ├─ base.py                # LLMProvider ABC (complete/stream)
│  │  │     ├─ factory.py             # provider from env
│  │  │     ├─ groq_provider.py
│  │  │     ├─ openai_provider.py
│  │  │     ├─ gemini_provider.py
│  │  │     └─ ollama_provider.py
│  │  ├─ tasks/
│  │  │  ├─ backend.py                # TaskBackend ABC (submit/run)
│  │  │  ├─ background_tasks.py       # FastAPI BackgroundTasks impl
│  │  │  └─ celery_app.py             # (stub) Celery+Redis impl later
│  │  ├─ prompts/
│  │  │  ├─ rewrite.txt
│  │  │  └─ answer.txt
│  │  └─ alembic/
│  │     ├─ env.py
│  │     └─ versions/
│  └─ tests/
│     ├─ conftest.py
│     ├─ fixtures.py
│     ├─ test_auth.py
│     ├─ test_kb.py
│     ├─ test_documents.py
│     ├─ test_retrieval.py
│     ├─ test_chat.py
│     ├─ test_isolation.py            # critical: cross-user access denied
│     └─ test_eval_ragas.py
├─ frontend/
│  ├─ requirements.txt
│  ├─ Dockerfile
│  ├─ app.py                          # entrypoint + nav
│  ├─ pages/
│  │  ├─ 01_Login_Register.py
│  │  ├─ 02_Knowledge_Bases.py
│  │  ├─ 03_Upload.py
│  │  ├─ 04_Chat.py
│  │  └─ 05_Settings.py
│  ├─ components/
│  │  ├─ api_client.py                # owns the JWT, refresh logic
│  │  ├─ auth_state.py                # session_state tokens
│  │  └─ chat_widget.py               # SSE consumer
│  └─ utils/
│     └─ config.py                    # BACKEND_URL from env
└─ eval/
   ├─ datasets/                       # sample Q/A + contexts
   └─ run_ragas.py
```

---

## 4. PostgreSQL schema

Conventions: `UUID` PKs via `gen_random_uuid()`, `TIMESTAMPTZ` (UTC), soft conventions, all user-scoped rows carry `user_id` for defense-in-depth filtering. Enums are `VARCHAR` CHECK-backed (not PG `ENUM`) for migration flexibility.

### 4.1 `users`
| column | type | constraints | notes |
|---|---|---|---|
| id | UUID | PK DEFAULT gen_random_uuid() | |
| email | CITEXT | NOT NULL UNIQUE | lowercased |
| hashed_password | VARCHAR(255) | NOT NULL | bcrypt `$2b$12$…` |
| full_name | VARCHAR(255) | NULL | |
| is_active | BOOLEAN | NOT NULL DEFAULT true | |
| is_superuser | BOOLEAN | NOT NULL DEFAULT false | |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT now() | |
| updated_at | TIMESTAMPTZ | NOT NULL DEFAULT now() | |
| last_login_at | TIMESTAMPTZ | NULL | |

Indexes: `UNIQUE(email)`.

### 4.2 `knowledge_bases`
| column | type | constraints | notes |
|---|---|---|---|
| id | UUID | PK | collection name = `kb_{id}` |
| user_id | UUID | FK→users.id ON DELETE CASCADE | isolation key |
| name | VARCHAR(255) | NOT NULL | |
| description | TEXT | NULL | |
| embedding_model | VARCHAR(255) | NOT NULL DEFAULT 'BAAI/bge-small-en-v1.5' | |
| chunk_size | INT | NOT NULL DEFAULT 800 | chars |
| chunk_overlap | INT | NOT NULL DEFAULT 160 | chars |
| status | VARCHAR(20) | NOT NULL DEFAULT 'active' CHECK | active/archived |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT now() | |
| updated_at | TIMESTAMPTZ | NOT NULL DEFAULT now() | |

Indexes: `UNIQUE(user_id, name)`, `idx_kb_user(user_id)`.

### 4.3 `documents`
| column | type | constraints | notes |
|---|---|---|---|
| id | UUID | PK | |
| kb_id | UUID | FK→knowledge_bases.id ON DELETE CASCADE | |
| user_id | UUID | FK→users.id ON DELETE CASCADE | denormalized for fast filter |
| file_name | VARCHAR(512) | NOT NULL | original name |
| file_type | VARCHAR(8) | NOT NULL CHECK in (pdf,docx,txt,md) | |
| storage_path | TEXT | NOT NULL | local path or S3 key |
| status | VARCHAR(20) | NOT NULL DEFAULT 'pending' CHECK | pending/processing/ready/failed |
| error_message | TEXT | NULL | |
| chunk_count | INT | NOT NULL DEFAULT 0 | |
| char_count | INT | NULL | |
| page_count | INT | NULL | |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT now() | |
| updated_at | TIMESTAMPTZ | NOT NULL DEFAULT now() | |

Indexes: `idx_doc_kb(kb_id)`, `idx_doc_user(user_id)`, `idx_doc_status(status)`, `UNIQUE(kb_id, file_name)`.

### 4.4 `chat_sessions`
| column | type | constraints | notes |
|---|---|---|---|
| id | UUID | PK | |
| user_id | UUID | FK→users.id ON DELETE CASCADE | |
| kb_id | UUID | FK→knowledge_bases.id ON DELETE CASCADE | |
| title | VARCHAR(255) | NULL | |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT now() | |
| updated_at | TIMESTAMPTZ | NOT NULL DEFAULT now() | |

Indexes: `idx_sess_user_kb(user_id, kb_id)`.

### 4.5 `messages`
| column | type | constraints | notes |
|---|---|---|---|
| id | UUID | PK | |
| session_id | UUID | FK→chat_sessions.id ON DELETE CASCADE | |
| user_id | UUID | FK→users.id ON DELETE CASCADE | |
| role | VARCHAR(20) | NOT NULL CHECK in (user,assistant,system) | |
| content | TEXT | NOT NULL | |
| citations | JSONB | NULL | `[{file_name,page,chunk_id,snippet}]` |
| token_usage | JSONB | NULL | `{prompt,completion,total}` |
| latency_ms | INT | NULL | assistant only |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT now() | |

Indexes: `idx_msg_sess(session_id, created_at)`.

### 4.6 `usage_logs`
| column | type | constraints | notes |
|---|---|---|---|
| id | BIGSERIAL | PK | high-volume |
| user_id | UUID | FK→users.id ON DELETE CASCADE | |
| kb_id | UUID | FK NULL | |
| session_id | UUID | FK NULL | |
| action | VARCHAR(50) | NOT NULL | ingestion/chat_completion/rewrite |
| model | VARCHAR(100) | NULL | llm model id |
| prompt_tokens | INT | NOT NULL DEFAULT 0 | |
| completion_tokens | INT | NOT NULL DEFAULT 0 | |
| total_tokens | INT | NOT NULL DEFAULT 0 | |
| latency_ms | INT | NULL | |
| retrieved_chunks | INT | NULL | |
| cost_usd | NUMERIC(10,6) | NULL | |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT now() | |

Indexes: `idx_usage_user_time(user_id, created_at)`, `idx_usage_kb(kb_id)`.

---

## 5. REST API contract

All paths prefixed `/api`. Auth = `Authorization: Bearer <access_token>`. Errors: `{ "detail": str, "code": str }` (422 for validation, 429 for rate limit).

| # | Method | Path | Auth | Request body | Success response | Codes |
|---|---|---|---|---|---|---|
| 1 | POST | `/auth/register` | no | `{email, password, full_name?}` | `{access_token, refresh_token, token_type, user}` | 201, 400, 409 |
| 2 | POST | `/auth/login` | no | `{email, password}` | `{access_token, refresh_token, token_type, user}` | 200, 401 |
| 3 | POST | `/auth/refresh` | refresh | `{refresh_token}` | `{access_token}` | 200, 401 |
| 4 | POST | `/auth/logout` | refresh | `{refresh_token}` | `204` | 204, 401 |
| 5 | GET | `/auth/me` | yes | — | `UserOut` | 200, 401 |
| 6 | GET | `/knowledge-bases` | yes | `?status=` | `[KnowledgeBaseOut]` | 200, 401 |
| 7 | POST | `/knowledge-bases` | yes | `{name, description?, chunk_size?, chunk_overlap?}` | `KnowledgeBaseOut` | 201, 400, 409, 422 |
| 8 | GET | `/knowledge-bases/{kb_id}` | yes | — | `KnowledgeBaseOut` | 200, 401, 403, 404 |
| 9 | PATCH | `/knowledge-bases/{kb_id}` | yes | `{name?, description?, chunk_size?, chunk_overlap?, status?}` | `KnowledgeBaseOut` | 200, 401, 403, 404 |
| 10 | DELETE | `/knowledge-bases/{kb_id}` | yes | — | `204` (cascades docs+chunks) | 204, 401, 403, 404 |
| 11 | POST | `/knowledge-bases/{kb_id}/documents` | yes | `multipart: file` | `DocumentOut(status=pending)` | 201, 400, 413, 401, 403, 404, 429 |
| 12 | GET | `/knowledge-bases/{kb_id}/documents` | yes | `?status=&limit=&offset=` | `[DocumentOut]` | 200, 401, 403, 404 |
| 13 | GET | `/documents/{doc_id}` | yes | — | `DocumentOut` | 200, 401, 403, 404 |
| 14 | DELETE | `/documents/{doc_id}` | yes | — | `204` (deletes chunks) | 204, 401, 403, 404 |
| 15 | GET | `/knowledge-bases/{kb_id}/sessions` | yes | `?limit=&offset=` | `[ChatSessionOut]` | 200, 401, 403, 404 |
| 16 | POST | `/knowledge-bases/{kb_id}/sessions` | yes | `{title?}` | `ChatSessionOut` | 201, 401, 403, 404 |
| 17 | GET | `/sessions/{session_id}/messages` | yes | `?limit=&offset=` | `[MessageOut]` | 200, 401, 403, 404 |
| 18 | POST | `/sessions/{session_id}/chat` | yes | `{message, use_history?}` | `MessageOut` (assistant, non-stream) | 200, 401, 403, 404, 429 |
| 19 | POST | `/sessions/{session_id}/chat/stream` | yes | `{message, use_history?, top_k?}` | `text/event-stream` SSE (see §5.4) | 200, 401, 403, 404, 429 |
| 20 | GET | `/settings/me` | yes | — | `SettingsOut` | 200, 401 |
| 21 | PATCH | `/settings/me` | yes | `{current_password?, new_password?, full_name?}` | `SettingsOut` | 200, 400, 401 |
| 22 | GET | `/users/me/usage` | yes | `?from=&to=&kb_id=` | `{total_tokens, requests, by_action[]}` | 200, 401 |
| 23 | GET | `/health` | no | — | `{status, db, chroma, version}` | 200, 503 |
| 24 | GET | `/health/ready` / `/health/live` | no | — | `{status:"ok"}` | 200 |

**SSE event format (`/chat/stream`):**
```
event: token     data: {"text": "partial..."}
event: citation  data: {"file_name":"x.pdf","page":3,"chunk_id":"...","snippet":"..."}
event: done      data: {"session_id":"...","message_id":"...","latency_ms":1234,"tokens":{...}}
event: error     data: {"detail":"...","code":"retrieval_empty"}
```

---

## 6. RAG pipeline (exact parameters)

### 6.1 Ingestion pipeline
1. **Extract** by type:
   - PDF → `pypdf`/`pdfplumber` page-by-page; keep `page` index (1-based).
   - DOCX → `python-docx` by paragraph; map to page heuristically (or store page=0).
   - TXT/MD → whole text; page=1 (or split by `---` page breaks for MD).
   - Output: `list[(text:str, page:int)]`.
2. **Normalize** whitespace; drop empty chunks.
3. **Chunk** — `RecursiveCharacterTextSplitter`:
   - `separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]`
   - `chunk_size = kb.chunk_size` (**default 800 chars**)
   - `chunk_overlap = kb.chunk_overlap` (**default 160 chars**)
   - `length_function = len` (char-based; safe under bge 512-token limit)
   - Preserve `page` via page-span merge (a chunk inherits min→max page of its source spans).
4. **Embed** — `BAAI/bge-small-en-v1.5` (384-dim):
   - Documents: embed **as-is** (no prefix).
   - Normalize embeddings (`normalize_embeddings=True`).
   - Batch size 32, run in `run_in_threadpool`.
5. **Upsert** to Chroma collection `kb_{kb_id}`:
   - `ids = f"{doc_id}:{chunk_idx}"`
   - `metadatas = {user_id, kb_id, doc_id, file_name, page, chunk_idx}`
   - `where` filters always include `user_id` (isolation).
6. **Finalize** — `documents.status=ready`, `chunk_count`, `page_count`; emit `usage_log` (action=ingestion).

### 6.2 Query pipeline
1. **Rewrite** (when `use_history`): feed last **N=6** messages + current question to LLM → standalone query. (LLM call, cached per request.)
2. **Vector retrieval**: Chroma `query(where={"user_id":u,"kb_id":k}, n_results=20)`; **add bge query prefix** `"Represent this sentence for searching relevant passages: "` to the standalone query before embedding. Cosine similarity (0–1).
3. **BM25 retrieval**: per-KB BM25 index over chunk texts, top 20 (`rank_bm25`; index lazily built from Chroma metadata, cached in memory, invalidated on ingest).
4. **RRF merge**: `score = Σ 1/(k + rank)`, **k=60**. Take top 30.
5. **Rerank (optional, default ON for quality)**: cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` over top 30 → keep top **K=6** context chunks. Controlled by `RERANK_ENABLED`.
6. **Similarity gate**: if best normalized score (vector sim **or** reranker score, depending on config) `< SIMILARITY_THRESHOLD = 0.35` → stream **"I don't know / I couldn't find that in your documents."** and stop (no LLM call). Tunable per deployment.
7. **Prompt**: system + retrieved context (each chunk tagged `[file: x.pdf, page: 3]`), instruct to answer **only** from context and cite as `[file_name, page]`. `temperature=0.1`, `max_tokens=1024`, `top_p=0.9`.
8. **Stream** LLM tokens via SSE; emit `citation` events as context is bound; on `done` persist assistant message (`citations`, `token_usage`, `latency_ms`) and emit `usage_log`.

**Recommended knobs (env):** `TOP_K_VECTOR=20`, `TOP_K_BM25=20`, `RRF_K=60`, `RERANK_TOP_N=6`, `SIMILARITY_THRESHOLD=0.35`, `CONTEXT_CHUNKS=6`, `HISTORY_TURNS=6`, `LLM_TEMPERATURE=0.1`.

---

## 7. Security checklist

**Data isolation (critical):**
- Every ORM query and every Chroma call filters by `user_id` (and `kb_id`). Use `get_kb` dependency that re-checks ownership on every KB/doc/session access — never trust client-supplied ids.
- Chroma collections are namespaced `kb_{kb_id}`; `where={"user_id": ...}` is mandatory (defense in depth even though logically scoped).
- `test_isolation.py` asserts user B gets 403/404 on user A's KB/doc/session/chunks.

**Auth & secrets:**
- bcrypt work factor **12**; password min length 8, reject common passwords.
- Access token **15 min**, refresh **7 days**; refresh tokens are **rotated** (server issues new refresh, old invalidated) and stored hashed (jti) for revocation/logout.
- JWT secret from env only (`JWT_SECRET`, ≥32 bytes); HS256; never commit `.env`. In prod use **AWS Secrets Manager / KMS**; inject at runtime.
- CORS: explicit `ALLOWED_ORIGINS` list; never `"*"` with credentials; restrict methods/headers.

**File validation & limits:**
- Extension + **MIME/magic-byte** check (reject if mismatch). Max **25 MB/file**, **200 MB/KB**, configurable.
- Disallow path traversal in filenames (strip dirs, sanitize).
- Optional: ClamAV scan in ingestion before chunking.
- Storage backend (local/S3) enforces `user_id/kb_id/` prefix paths.

**Other:**
- Global exception handlers → structured logs, no stack traces to client.
- `pydantic-settings` strict; no secrets in logs; redact tokens in access logs.
- Rate limiting: token-bucket per `user_id` (e.g. 20 req/min chat, 10 uploads/min).
- Dependency pinning + `pip-audit` in CI; minimal base image (`python:3.11-slim`).
- Health endpoints public; everything else requires auth.
- All DB access via ORM (parameterized); no string-built SQL.

---

## 8. Phase plan

| Phase | Goal | Key files produced |
|---|---|---|
| **1. Scaffold & foundations** | App factory, config, DB, Alembic, logging, exceptions, health, CI skeleton | `backend/app/main.py`, `core/{config,security,database,logging_config,exceptions,rate_limit,threadpool}.py`, `models/*`, `alembic/`, `alembic.ini`, `pyproject.toml`, `requirements.txt`, `Dockerfile`, `docker-compose.yml`, `.env.example`, `.github/workflows/ci.yml` |
| **2. Auth** | Register/login/refresh/logout, bcrypt, JWT deps | `core/security.py`, `services/auth_service.py`, `schemas/user.py`, `api/routers/{auth,health}.py`, `api/deps.py`, `tests/test_auth.py` |
| **3. Knowledge bases** | KB CRUD + ownership `get_kb` | `models/knowledge_base.py`, `schemas/knowledge_base.py`, `services/kb_service.py`, `api/routers/knowledge_bases.py`, `tests/test_kb.py`, `tests/test_isolation.py` |
| **4. Ingestion** | Upload, storage, background pipeline (extract→chunk→embed→upsert), status tracking | `services/storage.py`, `services/document_service.py`, `services/ingestion/{pipeline,extractors,chunker,embedder,vector_store}.py`, `tasks/{backend,background_tasks}.py`, `models/document.py`, `api/routers/documents.py`, `tests/test_documents.py` |
| **5. Retrieval** | BM25 + vector + RRF, reranker, query rewrite | `services/retrieval/{retriever,bm25,reranker,query_rewriter}.py`, `prompts/*`, `tests/test_retrieval.py` |
| **6. Chat** | SSE streaming, citations, "I don't know" gate, follow-up | `services/chat_service.py`, `services/llm/{base,factory,groq_provider,...}.py`, `models/{chat_session,message}.py`, `api/routers/chat.py`, `tests/test_chat.py` |
| **7. Ops & hardening** | Usage logging, rate limiting wiring, settings, structured logs, exception polish | `services/usage_service.py`, `api/routers/settings.py`, `core/rate_limit.py` wiring, `models/usage_log.py`, `tests/test_eval_*` |
| **8. Frontend & deploy** | Streamlit pages (Login/Register, KBs, Upload, Chat, Settings), SSE client, Docker, CI build, RAGAS eval | `frontend/app.py`, `frontend/pages/*`, `frontend/components/*`, `frontend/Dockerfile`, `eval/run_ragas.py`, `tests/test_eval_ragas.py`, full `docker-compose.yml` |

> Next: tell me which phase to start coding (recommend **Phase 1**). I will implement only that phase's files, with tests.
