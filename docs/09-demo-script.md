# 2-Minute Demo Script

A step-by-step walkthrough for recording a demo video of the Multi-User RAG
Platform. Total runtime: ~2 minutes. Setup (one-time, not recorded): start the
stack with `docker compose up -d --build` and open
http://localhost:8501 in your browser.

> **Tip:** record at 1080p, keep the browser console visible for the API calls,
> and narrate the *why* behind each step, not just the clicks.

---

## 0:00 – 0:15 — Introduction (on-screen title)

**Say:** "This is the Multi-User RAG Platform — a multi-user, citation-based
retrieval-augmented generation system. Users get private knowledge bases,
upload their own documents, and every answer is grounded in *their* files —
with citations, or it refuses to answer."

**On screen:** the login page (`http://localhost:8501`).

---

## 0:15 – 0:30 — Registration & authentication

1. Click **Register**.
2. Enter an email and password; submit.
3. **Say:** "Auth is JWT-based — short-lived access tokens plus rotating
   refresh tokens, hashed with bcrypt. Each user only ever sees their own
   data."

**On screen:** you land on the Knowledge Bases page after registering.

---

## 0:30 – 0:45 — Create a knowledge base

1. Click **New Knowledge Base**, name it e.g. *Product Docs*, create it.
2. **Say:** "A knowledge base is a private container for documents — chunked,
   embedded, and indexed. Chunk size and overlap are per-KB."

**On screen:** the new KB appears in the list.

---

## 0:45 – 1:10 — Upload a document

1. Open the KB, click **Upload Document**.
2. Drop in a sample PDF (or .txt/.docx), wait for ingestion.
3. **Say:** "On upload, the file is parsed, split into overlapping chunks with
   full provenance — file name, page span — embedded, and indexed. Duplicate
   uploads are detected by content hash. The status transitions from
   processing to ready."

**On screen:** the document row flips to **Ready** (show the status change).

---

## 1:10 – 1:40 — Ask a question (the core demo)

1. Open **Chat**, create a session in the KB.
2. Ask a question the document answers, e.g. *"What is our refund policy?"*
3. Let the answer stream in; show the **citation chips** below the answer.
4. **Say:** "The query goes through hybrid retrieval — vector search plus BM25
   keyword search, fused with RRF and reranked. The answer is streamed over
   SSE, and citations like *refund_policy.pdf, page 2* are resolved to the
   exact retrieved chunk."
5. Ask a **second, unrelated question** (e.g. *"What is the stock price of
   Acme?"*).
6. **Say:** "When nothing relevant is found, the similarity gate short-circuits
   — no LLM call, no hallucination. It tells you it couldn't find it in your
   documents."

**On screen:** streaming answer → citation chips → the "couldn't find it"
response for the off-topic question.

---

## 1:40 – 1:55 — Multi-user isolation (the security pitch)

1. Log out, register a **second user** in a new browser tab (or private
   window).
2. Try to open the first user's knowledge base / document / chat session —
   show the **404**.
3. **Say:** "Isolation is enforced at every layer — the API checks ownership,
   every ORM query filters by user id, and every vector-store query carries a
   mandatory user-id filter. One user can never retrieve another's documents."

**On screen:** the second user sees an empty KB list; direct URL access to the
first user's resources returns 404.

---

## 1:55 – 2:00 — Wrap-up

**Say:** "That's the full loop — auth, private knowledge bases, hybrid
retrieval, citation-grounded streaming answers, and enforced data isolation.
The stack is containerized with Docker Compose, has CI/CD, RAGAS evaluation
tooling, and a full AWS deployment guide. Thanks for watching."

**On screen:** final shot of the platform, end card with the repo URL.

---

## Recording checklist

- [ ] Stack running and seeded with one sample document before recording.
- [ ] Two browser contexts prepared (user A / user B) for the isolation part.
- [ ] Demo account passwords noted for re-records.
- [ ] Volume/audio levels tested; cursor-follow recording enabled.
- [ ] Captions enabled or narration script printed beside the camera.
