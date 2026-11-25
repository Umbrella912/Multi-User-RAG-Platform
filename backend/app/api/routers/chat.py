"""Chat endpoints: sessions, messages, and streaming/sync RAG answers.

Sessions are bound to a knowledge base and owned by a user. Asking a question
runs the RAG pipeline (rewrite -> retrieve -> rerank -> generate) and streams
the answer as Server-Sent Events, persisting both the user and assistant
messages (with citations as JSON) and a :class:`UsageLog` row per request. A
non-streaming ``ask-sync`` variant is provided for testing.

All queries are scoped to the authenticated user (data isolation).
"""
from __future__ import annotations

import json
from typing import Annotated, Any
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
)
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user, get_rag_pipeline, get_session
from app.core.config import settings
from app.core.logging import get_logger
from app.core.rate_limit import rate_limit
from app.db.session import async_session_factory as _default_session_factory
from app.models.chat_session import ChatSession
from app.models.knowledge_base import KnowledgeBase
from app.models.message import Message
from app.models.usage_log import UsageLog
from app.models.user import User
from app.schemas.chat import (
    AskRequest,
    AskSyncResponse,
    ChatSessionCreate,
    ChatSessionOut,
    MessageOut,
)
from app.services.rag_pipeline import RAGPipeline

#: Session factory used for background persistence; patchable in tests.
_session_factory = _default_session_factory

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


def _format_sse(event: dict[str, Any]) -> str:
    """Format an event dict as a Server-Sent Events ``data:`` frame."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


async def _get_owned_session(
    session: AsyncSession, session_id: UUID, user_id: UUID
) -> ChatSession:
    """Fetch a chat session owned by ``user_id``; 404 if missing or not owned."""
    chat_session = await session.get(ChatSession, session_id)
    if chat_session is None or chat_session.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="chat session not found",
        )
    return chat_session


async def _get_owned_kb(
    session: AsyncSession, kb_id: UUID, user_id: UUID
) -> KnowledgeBase:
    """Fetch a KB owned by ``user_id``; 404 if missing or not owned."""
    kb = await session.get(KnowledgeBase, kb_id)
    if kb is None or kb.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="knowledge base not found",
        )
    return kb


async def _load_history(session: AsyncSession, chat_session: ChatSession) -> list[dict[str, str]]:
    """Return the last ``HISTORY_TURNS`` messages as ``{role, content}`` dicts."""
    result = await session.scalars(
        select(Message)
        .where(Message.session_id == chat_session.id)
        .order_by(Message.created_at.desc())
        .limit(settings.HISTORY_TURNS)
    )
    messages = list(result.all())
    messages.reverse()  # back to chronological order
    return [{"role": m.role, "content": m.content} for m in messages]


async def _persist_user_message(
    session: AsyncSession, chat_session: ChatSession, question: str
) -> Message:
    """Insert and commit the user's question message."""
    message = Message(
        session_id=chat_session.id,
        user_id=chat_session.user_id,
        role="user",
        content=question,
    )
    session.add(message)
    await session.commit()
    await session.refresh(message)
    return message


async def _persist_assistant_and_log(
    chat_session: ChatSession,
    answer: str,
    citations: list[dict[str, Any]],
    token_usage: dict[str, Any] | None,
    retrieved_chunks: list[dict[str, Any]],
    latency_ms: int | None,
    model: str,
) -> None:
    """Persist the assistant message and a UsageLog row in a fresh session."""
    prompt_tokens = int((token_usage or {}).get("prompt_tokens", 0) or 0)
    completion_tokens = int((token_usage or {}).get("completion_tokens", 0) or 0)
    total_tokens = int((token_usage or {}).get("total_tokens", 0) or 0)

    async with _session_factory() as db:
        assistant = Message(
            session_id=chat_session.id,
            user_id=chat_session.user_id,
            role="assistant",
            content=answer,
            citations=citations,
            token_usage=token_usage,
            latency_ms=latency_ms,
        )
        db.add(assistant)

        usage = UsageLog(
            user_id=chat_session.user_id,
            kb_id=chat_session.kb_id,
            session_id=chat_session.id,
            action="chat_completion",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            retrieved_chunks=len(retrieved_chunks),
        )
        db.add(usage)
        await db.commit()


@router.post(
    "/sessions",
    response_model=ChatSessionOut,
    status_code=status.HTTP_201_CREATED,
)
async def create_session(
    body: ChatSessionCreate,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ChatSession:
    """Create a chat session bound to an owned knowledge base."""
    await _get_owned_kb(session, body.kb_id, user.id)
    chat_session = ChatSession(
        user_id=user.id,
        kb_id=body.kb_id,
        title=body.title,
    )
    session.add(chat_session)
    await session.commit()
    await session.refresh(chat_session)
    return chat_session


@router.get("/sessions", response_model=list[ChatSessionOut])
async def list_sessions(
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
    kb_id: Annotated[UUID | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[ChatSession]:
    """List the current user's chat sessions (optionally filtered by KB)."""
    query = (
        select(ChatSession)
        .where(ChatSession.user_id == user.id)
        .order_by(ChatSession.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    if kb_id is not None:
        query = query.where(ChatSession.kb_id == kb_id)
    result = await session.scalars(query)
    return list(result.all())


@router.get("/sessions/{session_id}/messages", response_model=list[MessageOut])
async def list_messages(
    session_id: UUID,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[Message]:
    """List messages in an owned session (oldest first)."""
    chat_session = await _get_owned_session(session, session_id, user.id)
    result = await session.scalars(
        select(Message)
        .where(Message.session_id == chat_session.id)
        .order_by(Message.created_at.asc())
        .limit(limit)
        .offset(offset)
    )
    return list(result.all())


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
)
async def delete_session(
    session_id: UUID,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> None:
    """Delete an owned session (cascades to its messages)."""
    chat_session = await _get_owned_session(session, session_id, user.id)
    await session.delete(chat_session)
    await session.commit()


@router.post(
    "/sessions/{session_id}/ask",
    dependencies=[Depends(rate_limit(settings.RATE_LIMIT_CHAT_PER_MIN, 60, key_prefix="chat"))],
)
async def ask(
    session_id: UUID,
    body: AskRequest,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
    pipeline: Annotated[RAGPipeline, Depends(get_rag_pipeline)],
) -> StreamingResponse:
    """Ask a question; stream the RAG answer as Server-Sent Events.

    Emits ``rewrite``, ``citation``, ``token`` events, then a final ``done``
    event carrying citations, retrieved chunks, latency, and token usage. The
    user and assistant messages plus a UsageLog row are persisted.
    """
    chat_session = await _get_owned_session(session, session_id, user.id)
    await _get_owned_kb(session, chat_session.kb_id, user.id)
    history = await _load_history(session, chat_session)
    await _persist_user_message(session, chat_session, body.question)

    async def event_stream():
        answer = ""
        citations: list[dict[str, Any]] = []
        retrieved_chunks: list[dict[str, Any]] = []
        token_usage: dict[str, Any] | None = None
        latency_ms: int | None = None
        completed = False
        try:
            async for event in pipeline.answer_stream(
                body.question,
                chat_session.kb_id,
                user.id,
                history=history,
                use_history=body.use_history,
            ):
                yield _format_sse(event)
                if event.get("type") == "done":
                    completed = True
                    answer = event.get("answer", "")
                    citations = event.get("citations", [])
                    retrieved_chunks = event.get("retrieved_chunks", [])
                    token_usage = event.get("token_usage")
                    latency_ms = event.get("latency_ms")
        except Exception as exc:  # noqa: BLE001 - surfaced as an SSE error frame
            logger.exception("chat stream failed", extra={"session_id": str(session_id)})
            yield _format_sse({"type": "error", "detail": str(exc)})
            return

        if completed:
            try:
                await _persist_assistant_and_log(
                    chat_session,
                    answer,
                    citations,
                    token_usage,
                    retrieved_chunks,
                    latency_ms,
                    pipeline.model,
                )
            except Exception as exc:  # noqa: BLE001 - persistence failure
                logger.error("failed to persist assistant message", extra={"error": str(exc)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post(
    "/sessions/{session_id}/ask-sync",
    response_model=AskSyncResponse,
    dependencies=[Depends(rate_limit(settings.RATE_LIMIT_CHAT_PER_MIN, 60, key_prefix="chat"))],
)
async def ask_sync(
    session_id: UUID,
    body: AskRequest,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
    pipeline: Annotated[RAGPipeline, Depends(get_rag_pipeline)],
) -> AskSyncResponse:
    """Non-streaming RAG answer (for testing and simple clients)."""
    chat_session = await _get_owned_session(session, session_id, user.id)
    await _get_owned_kb(session, chat_session.kb_id, user.id)
    history = await _load_history(session, chat_session)
    await _persist_user_message(session, chat_session, body.question)

    result = await pipeline.answer(
        body.question,
        chat_session.kb_id,
        user.id,
        history=history,
        use_history=body.use_history,
    )

    await _persist_assistant_and_log(
        chat_session,
        result.answer,
        [c.__dict__ for c in result.citations],
        result.token_usage,
        [
            {"id": c.id, "file_name": c.file_name, "page": c.page, "score": c.score}
            for c in result.chunks
        ],
        result.latency_ms,
        result.model,
    )

    return AskSyncResponse(
        session_id=chat_session.id,
        answer=result.answer,
        citations=[c.__dict__ for c in result.citations],
        retrieved_chunks=[
            {"id": c.id, "file_name": c.file_name, "page": c.page, "score": c.score}
            for c in result.chunks
        ],
        latency_ms=result.latency_ms or 0,
        token_usage=result.token_usage,
        skipped=result.skipped,
        model=result.model,
    )
