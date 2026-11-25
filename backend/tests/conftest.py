"""Shared pytest fixtures.

DB-backed tests use a real PostgreSQL instance (per ``TEST_DATABASE_URL``,
defaulting to ``settings.DATABASE_URL``). When no database is reachable the
fixtures skip, so the suite runs cleanly in environments without Postgres.
"""
from __future__ import annotations

import os
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.db.base import Base
from app.db.session import get_session
from app.main import create_app

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", settings.DATABASE_URL)


@pytest_asyncio.fixture
async def app() -> AsyncGenerator:
    """Provide an app bound to a throwaway test database (recreated per test)."""
    engine = create_async_engine(TEST_DATABASE_URL, future=True)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception:
        pytest.skip(f"no reachable test database at {TEST_DATABASE_URL}")

    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS citext"))
        await conn.run_sync(Base.metadata.create_all)

    test_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def _override() -> AsyncGenerator[AsyncSession, None]:
        async with test_factory() as session:
            yield session

    application = create_app()
    application.dependency_overrides[get_session] = _override
    # Expose the test session factory for helpers that open their own sessions.
    application.state.test_factory = test_factory
    yield application

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
def test_session_factory(app):
    """Return the test session factory (for helpers opening their own sessions)."""
    return app.state.test_factory


@pytest_asyncio.fixture
async def client(app) -> AsyncGenerator[AsyncClient, None]:
    """HTTP client bound to the test app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
