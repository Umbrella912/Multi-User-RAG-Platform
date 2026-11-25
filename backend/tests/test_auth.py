"""Auth flow tests: register, login, me, refresh, and failure cases."""
from __future__ import annotations

from httpx import AsyncClient

API = "/api"


async def _register(client: AsyncClient, email: str, password: str = "secret123") -> dict:
    resp = await client.post(
        f"{API}/auth/register",
        json={"email": email, "password": password, "full_name": "Tester"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


async def test_register_login_me_flow(client: AsyncClient) -> None:
    body = await _register(client, "alice@example.com")
    assert body["token_type"] == "bearer"
    assert body["access_token"] and body["refresh_token"]
    assert body["user"]["email"] == "alice@example.com"

    # Login with correct credentials.
    login = await client.post(
        f"{API}/auth/login",
        json={"email": "alice@example.com", "password": "secret123"},
    )
    assert login.status_code == 200
    access = login.json()["access_token"]

    # /auth/me with a valid token.
    me = await client.get(f"{API}/auth/me", headers={"Authorization": f"Bearer {access}"})
    assert me.status_code == 200
    assert me.json()["email"] == "alice@example.com"


async def test_duplicate_registration_conflicts(client: AsyncClient) -> None:
    await _register(client, "dup@example.com")
    again = await client.post(
        f"{API}/auth/register",
        json={"email": "dup@example.com", "password": "secret123"},
    )
    assert again.status_code == 409


async def test_login_wrong_password_rejected(client: AsyncClient) -> None:
    await _register(client, "bob@example.com")
    resp = await client.post(
        f"{API}/auth/login",
        json={"email": "bob@example.com", "password": "nope-wrong"},
    )
    assert resp.status_code == 401


async def test_me_requires_token(client: AsyncClient) -> None:
    resp = await client.get(f"{API}/auth/me")
    assert resp.status_code == 401


async def test_invalid_token_rejected(client: AsyncClient) -> None:
    resp = await client.get(
        f"{API}/auth/me", headers={"Authorization": "Bearer not.a.valid.jwt"}
    )
    assert resp.status_code == 401


async def test_refresh_rotates_tokens(client: AsyncClient) -> None:
    body = await _register(client, "carol@example.com")
    refresh = body["refresh_token"]
    resp = await client.post(f"{API}/auth/refresh", json={"refresh_token": refresh})
    assert resp.status_code == 200
    assert resp.json()["access_token"]
