#!/usr/bin/env bash
# Phase 2 manual API test: auth + knowledge-base isolation.
# Run against a booted server:  uvicorn app.main:app --host 0.0.0.0 --port 8000
# Requires the DB migrated:      alembic upgrade head
set -euo pipefail

BASE="${BASE:-http://127.0.0.1:8000}"
API="$BASE/api"

# 1) Register a user (returns access + refresh tokens).
REG=$(curl -s -X POST "$API/auth/register" -H 'Content-Type: application/json' \
  -d '{"email":"alice@example.com","password":"secret123","full_name":"Alice"}')
echo "REGISTER: $REG"
ACCESS=$(echo "$REG" | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# 2) Login.
curl -s -X POST "$API/auth/login" -H 'Content-Type: application/json' \
  -d '{"email":"alice@example.com","password":"secret123"}'; echo

# 3) Current user (valid token).
curl -s "$API/auth/me" -H "Authorization: Bearer $ACCESS"; echo

# 4) Current user with an INVALID token -> expect 401.
curl -s -o /dev/null -w "invalid token -> HTTP %{http_code}\n" \
  "$API/auth/me" -H "Authorization: Bearer not.a.real.token"

# 5) Create a knowledge base.
KB=$(curl -s -X POST "$API/knowledge-bases" -H "Authorization: Bearer $ACCESS" \
  -H 'Content-Type: application/json' -d '{"name":"Alice KB"}')
echo "CREATE KB: $KB"
KB_ID=$(echo "$KB" | python -c "import sys,json; print(json.load(sys.stdin)['id'])")

# 6) List the user's knowledge bases.
curl -s "$API/knowledge-bases" -H "Authorization: Bearer $ACCESS"; echo

# 7) Get one.
curl -s "$API/knowledge-bases/$KB_ID" -H "Authorization: Bearer $ACCESS"; echo

# 8) A DIFFERENT user tries to read Alice's KB -> expect 404.
REG_B=$(curl -s -X POST "$API/auth/register" -H 'Content-Type: application/json' \
  -d '{"email":"bob@example.com","password":"secret123","full_name":"Bob"}')
ACCESS_B=$(echo "$REG_B" | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
curl -s -o /dev/null -w "Bob reads Alice's KB -> HTTP %{http_code} (expect 404)\n" \
  "$API/knowledge-bases/$KB_ID" -H "Authorization: Bearer $ACCESS_B"

# 9) Bob lists his KBs -> empty.
curl -s "$API/knowledge-bases" -H "Authorization: Bearer $ACCESS_B"; echo

# 10) Update / delete Alice's KB.
curl -s -X PATCH "$API/knowledge-bases/$KB_ID" -H "Authorization: Bearer $ACCESS" \
  -H 'Content-Type: application/json' -d '{"description":"updated"}'; echo
curl -s -o /dev/null -w "delete KB -> HTTP %{http_code}\n" \
  -X DELETE "$API/knowledge-bases/$KB_ID" -H "Authorization: Bearer $ACCESS"

# 11) Refresh the access token.
REF=$(echo "$REG" | python -c "import sys,json; print(json.load(sys.stdin)['refresh_token'])")
curl -s -X POST "$API/auth/refresh" -H 'Content-Type: application/json' \
  -d "{\"refresh_token\":\"$REF\"}"; echo
