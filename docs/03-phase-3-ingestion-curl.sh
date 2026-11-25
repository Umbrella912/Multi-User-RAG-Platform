#!/usr/bin/env bash
# Phase 3 manual API test: document ingestion pipeline.
# Run against a booted server:  uvicorn app.main:app --host 0.0.0.0 --port 8000
# Requires:  alembic upgrade head  (and chromadb reachable for real ingestion).
set -euo pipefail

BASE="${BASE:-http://127.0.0.1:8000}"
API="$BASE/api"

# 1) Register + create a KB, capture token and kb_id.
REG=$(curl -s -X POST "$API/auth/register" -H 'Content-Type: application/json' \
  -d '{"email":"ingest@example.com","password":"secret123","full_name":"Ingest"}')
TOKEN=$(echo "$REG" | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
H="Authorization: Bearer $TOKEN"

KB=$(curl -s -X POST "$API/knowledge-bases" -H "$H" -H 'Content-Type: application/json' \
  -d '{"name":"Ingestion KB"}')
KB_ID=$(echo "$KB" | python -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "KB_ID=$KB_ID"

# 2) Upload a text file -> returns DocumentOut with status=pending.
echo "--- upload txt ---"
curl -s -X POST "$API/knowledge-bases/$KB_ID/documents" -H "$H" \
  -F "file=@./README.md;type=text/markdown"
echo

# 3) Upload a PDF (real ingestion requires PyMuPDF + Chroma running).
echo "--- upload pdf ---"
curl -s -X POST "$API/knowledge-bases/$KB_ID/documents" -H "$H" \
  -F "file=@./some.pdf;type=application/pdf"
echo

# 4) Duplicate upload (same content) -> expect 409.
echo "--- duplicate upload ---"
curl -s -o /dev/null -w "HTTP %{http_code} (expect 409)\n" -X POST \
  "$API/knowledge-bases/$KB_ID/documents" -H "$H" \
  -F "file=@./README.md;type=text/markdown"

# 5) Bad extension -> expect 400.
echo "--- bad extension ---"
curl -s -o /dev/null -w "HTTP %{http_code} (expect 400)\n" -X POST \
  "$API/knowledge-bases/$KB_ID/documents" -H "$H" \
  -F "file=@./script.sh;type=application/octet-stream"

# 6) List documents (optional ?status=ready|processing|pending|failed).
curl -s "$API/knowledge-bases/$KB_ID/documents" -H "$H"; echo
curl -s "$API/knowledge-bases/$KB_ID/documents?status=ready" -H "$H"; echo

# 7) Poll a single document's status (id from the upload response above).
#    Status goes pending -> processing -> ready (or failed with error_message).
DOC_ID="..."  # <- set from step 2
curl -s "$API/documents/$DOC_ID" -H "$H"; echo

# 8) Delete a document (removes its file and its vectors).
curl -s -o /dev/null -w "delete -> HTTP %{http_code} (expect 204)\n" -X DELETE \
  "$API/documents/$DOC_ID" -H "$H"
