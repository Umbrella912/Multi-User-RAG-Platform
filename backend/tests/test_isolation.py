"""Data-isolation tests: a user can never read another user's resources."""
from __future__ import annotations

from httpx import AsyncClient

API = "/api"


async def _register_and_token(client: AsyncClient, email: str) -> tuple[dict, str]:
    body = await client.post(
        f"{API}/auth/register",
        json={"email": email, "password": "secret123", "full_name": "U"},
    )
    assert body.status_code == 201, body.text
    token = body.json()["access_token"]
    return body.json(), token


async def test_user_b_cannot_access_user_a_knowledge_base(client: AsyncClient) -> None:
    _, token_a = await _register_and_token(client, "user-a@example.com")
    headers_a = {"Authorization": f"Bearer {token_a}"}

    created = await client.post(
        f"{API}/knowledge-bases", json={"name": "Private KB"}, headers=headers_a
    )
    assert created.status_code == 201
    kb_id = created.json()["id"]

    _, token_b = await _register_and_token(client, "user-b@example.com")
    headers_b = {"Authorization": f"Bearer {token_b}"}

    # B's listing is empty.
    listing = await client.get(f"{API}/knowledge-bases", headers=headers_b)
    assert listing.status_code == 200
    assert listing.json() == []

    # B cannot fetch A's KB by id -> 404 (existence must not leak).
    fetch = await client.get(f"{API}/knowledge-bases/{kb_id}", headers=headers_b)
    assert fetch.status_code == 404

    # A can fetch it.
    own = await client.get(f"{API}/knowledge-bases/{kb_id}", headers=headers_a)
    assert own.status_code == 200
    assert own.json()["id"] == kb_id


async def test_unauthenticated_kb_access_rejected(client: AsyncClient) -> None:
    resp = await client.get(f"{API}/knowledge-bases")
    assert resp.status_code == 401
