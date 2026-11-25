"""FastAPI application factory and top-level wiring."""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import auth, chat, documents, health, knowledge_bases
from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import get_logger, setup_logging

logger = get_logger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: start-up and shutdown hooks."""
    logger.info("application_startup", extra={"environment": settings.ENVIRONMENT})
    yield
    logger.info("application_shutdown")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application.

    Returns:
        A fully configured :class:`FastAPI` instance.
    """
    setup_logging(settings.LOG_LEVEL)

    app = FastAPI(
        title=settings.PROJECT_NAME,
        version="0.1.0",
        debug=settings.DEBUG,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    # Phase 1: health (no /api prefix). Phase 2: auth + KBs (/api).
    # Phase 3: documents upload/CRUD (/api).
    app.include_router(health.router)
    app.include_router(auth.router, prefix=settings.API_V1_PREFIX)
    app.include_router(knowledge_bases.router, prefix=settings.API_V1_PREFIX)
    app.include_router(documents.router, prefix=settings.API_V1_PREFIX)
    app.include_router(chat.router, prefix=settings.API_V1_PREFIX)

    return app


# Uvicorn entrypoint: `uvicorn app.main:app`.
app = create_app()
