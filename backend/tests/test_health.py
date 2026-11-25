"""Health endpoint and application wiring tests (no live DB required)."""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_live(client: TestClient) -> None:
    """The liveness probe must always respond 200 without a database."""
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_openapi_schema_generates(client: TestClient) -> None:
    """Route registration must produce a valid OpenAPI document."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "paths" in schema
    # Health routes registered without the /api prefix.
    assert "/health" in schema["paths"]
    assert "/health/live" in schema["paths"]
    assert "/health/ready" in schema["paths"]


def test_unknown_route_404(client: TestClient) -> None:
    """Unknown routes return the standard JSON error shape."""
    response = client.get("/does-not-exist")
    assert response.status_code == 404
    body = response.json()
    assert body["code"] == "http_error"
    assert "detail" in body
