#!/usr/bin/env bash
# Phase 5 manual API test: chat sessions + SSE streaming answer.
# Run against a booted server:  uvicorn app.main:app --host 0.0.0.0 --port 8000
# Requires:  alembic upgrade head  (and Chroma + an LLM provider reachable).
set -euo pipefail

BASE="${BASE:-http://127.0.0.1:8000}"
API="$BASE/api"

# 1) Register + create a KB; capture the auth header.
REG=$(curl -s -X POST "$API/auth/register" -H 'Content-Type: application/json' \
  -d '{"email":"chat@example.com","password":"secret123","full_name":"Chat"}')
TOKEN=$(echo "$REG" | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
H="Authorization: Bearer $TOKEN"

KB=$(curl -s -X POST "$API/knowledge-bases" -H "$H" -H 'Content-Type: application/json' \
  -d '{"name":"Chat KB"}')
KB_ID=$(echo "$KB" | python -c "import sys,json; print(json.load(sys.stdin)['id'])")

# 2) Create a chat session bound to the KB.
SESSION=$(curl -s -X POST "$API/chat/sessions" -H "$H" -H 'Content-Type: application/json' \
  -d "{\"kb_id\":\"$KB_ID\",\"title\":\"Refund Q&A\"}")
SESSION_ID=$(echo "$SESSION" | python -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "SESSION_ID=$SESSION_ID"

# 3) List sessions / messages (initially empty).
echo "--- sessions ---"; curl -s "$API/chat/sessions" -H "$H"; echo
echo "--- messages (empty) ---"; curl -s "$API/chat/sessions/$SESSION_ID/messages" -H "$H"; echo

# 4) Ask a question — STREAMED as Server-Sent Events.
#    Each `data:` line is one event: rewrite, citation, token, ..., done.
#    The final `done` event carries citations, retrieved chunks, latency,
#    and token usage. Run in a terminal to see tokens arrive live.
echo "--- ask (SSE stream) ---"
curl -N -X POST "$API/chat/sessions/$SESSION_ID/ask" -H "$H" \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the refund policy?"}'
echo

# 5) Ask again with history (the rewriter folds prior turns into the query).
echo "--- ask with history (SSE stream) ---"
curl -N -X POST "$API/chat/sessions/$SESSION_ID/ask" -H "$H" \
  -H 'Content-Type: application/json' \
  -d '{"question":"How long is it?","use_history":true}'
echo

# 6) Non-streaming variant (single JSON response) — handy for testing.
echo "--- ask-sync (JSON) ---"
curl -s -X POST "$API/chat/sessions/$SESSION_ID/ask-sync" -H "$H" \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the refund policy?"}'
echo

# 7) Messages are now persisted (user + assistant, citations as JSON).
echo "--- messages (persisted) ---"
curl -s "$API/chat/sessions/$SESSION_ID/messages" -H "$H" | python -m json.tool

# 8) Delete the session (cascades to messages).
echo "--- delete session ---"
curl -s -o /dev/null -w "delete -> HTTP %{http_code} (expect 204)\n" \
  -X DELETE "$API/chat/sessions/$SESSION_ID" -H "$H"
