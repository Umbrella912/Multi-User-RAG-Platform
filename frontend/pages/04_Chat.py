"""Chat page: session list in the sidebar, streaming RAG answers, citations.

Renders prior messages with ``st.chat_message``, streams the new answer token-
by-token via SSE (updating a placeholder per token), and shows retrieved
citations as expandable blocks with the source chunk text.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

import common
from api_client import ApiError


def main() -> None:
    """Render the chat page."""
    st.set_page_config(page_title="Chat · RAG Platform", page_icon="📚", layout="wide")
    common.apply_styles()
    common.require_auth()
    client = common.get_client()

    _render_sidebar(client)

    st.title("💬 Chat")
    kb = st.session_state.get("current_kb")
    if not kb:
        st.warning("Select a knowledge base first (Knowledge Bases page).")
        return
    st.caption(f"Chatting with **{kb['name']}**.")

    session = _select_session(client, kb)
    if not session:
        st.info("Create a new chat to start asking questions.")
        return

    # Render conversation history.
    messages = _load_messages(client, session["id"])
    for msg in messages:
        _render_message(msg)

    # Stream a new answer.
    question = st.chat_input("Ask a question about your documents…")
    if question:
        _stream_answer(client, session, question)


def _render_sidebar(client) -> None:
    """Render the branded sidebar with the session list and new-chat button."""
    st.sidebar.title("📚 RAG Platform")
    user = st.session_state.get("user", {})
    st.sidebar.caption(f"Signed in as\n**{user.get('full_name') or user.get('email', '')}**")
    st.sidebar.divider()

    if st.sidebar.button("➕ New chat", use_container_width=True):
        kb = st.session_state.get("current_kb")
        if not kb:
            st.sidebar.warning("Select a KB first.")
        else:
            session = client.create_session(kb["id"], title="New chat")
            st.session_state.active_session = session
            st.rerun()

    st.sidebar.header("Sessions")
    try:
        sessions = client.list_sessions()
    except ApiError as exc:
        st.sidebar.error(exc.message)
        sessions = []

    if not sessions:
        st.sidebar.caption("No sessions yet.")
    active_id = st.session_state.get("active_session", {}).get("id")
    for session in sessions:
        label = session.get("title") or "New chat"
        is_active = session["id"] == active_id
        label = f"{'🟢 ' if is_active else ''}{label}"
        if st.sidebar.button(label, use_container_width=True, key=f"sess_{session['id']}"):
            st.session_state.active_session = session
            st.rerun()
        if st.sidebar.button("🗑️", key=f"delsess_{session['id']}"):
            try:
                client.delete_session(session["id"])
                if st.session_state.get("active_session", {}).get("id") == session["id"]:
                    st.session_state.pop("active_session", None)
                st.rerun()
            except ApiError as exc:
                st.sidebar.error(exc.message)

    if st.sidebar.button("🔄 Refresh", use_container_width=True):
        st.rerun()


def _select_session(client, kb: dict):
    """Return the active session, ensuring it belongs to the current KB."""
    session = st.session_state.get("active_session")
    if session and session.get("kb_id") == kb["id"]:
        return session
    sessions = client.list_sessions(kb["id"])
    if sessions:
        st.session_state.active_session = sessions[0]
        return sessions[0]
    return None


def _load_messages(client, session_id: str) -> list[dict]:
    """Load and cache the session's messages."""
    if f"messages_{session_id}" not in st.session_state:
        try:
            st.session_state[f"messages_{session_id}"] = client.list_messages(session_id)
        except ApiError as exc:
            st.error(exc.message)
            st.session_state[f"messages_{session_id}"] = []
    return st.session_state[f"messages_{session_id}"]


def _render_message(msg: dict) -> None:
    """Render a persisted message, expanding citations for assistant turns."""
    role = msg["role"]
    with st.chat_message(role):
        st.markdown(msg["content"])
        if role == "assistant" and msg.get("citations"):
            _render_citations(msg["citations"], with_chunk_text=True)
        if role == "assistant" and msg.get("token_usage"):
            usage = msg["token_usage"]
            st.caption(
                f"⏱ {msg.get('latency_ms') or 0} ms · "
                f"{usage.get('total_tokens', 0)} tokens"
            )


def _stream_answer(client, session: dict, question: str) -> None:
    """Render the user turn, stream the assistant answer, and show citations."""
    st.chat_message("user").markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        citation_events: list[dict] = []
        done_event: dict | None = None

        answer_parts: list[str] = []
        try:
            for event in client.ask_stream(session["id"], question):
                etype = event.get("type")
                if etype == "token":
                    answer_parts.append(event.get("text", ""))
                    partial = "".join(answer_parts)
                    placeholder.markdown(
                        f'<div class="stream-caret">{partial}</div>',
                        unsafe_allow_html=True,
                    )
                elif etype == "citation":
                    citation_events.append(event)
                elif etype == "done":
                    done_event = event
                elif etype == "error":
                    placeholder.empty()
                    st.error(event.get("detail", "An error occurred during generation."))
        except ApiError as exc:
            placeholder.empty()
            st.error(exc.message)
            return

        # Replace the streaming caret with the final answer + citations.
        final_answer = (done_event or {}).get("answer") or "".join(answer_parts)
        placeholder.markdown(final_answer or "(no response)")

        citations = (done_event or {}).get("citations") or citation_events
        if citations:
            _render_citations(citations, with_chunk_text=True)

        done = done_event or {}
        if done.get("token_usage"):
            usage = done["token_usage"]
            st.caption(
                f"⏱ {done.get('latency_ms') or 0} ms · "
                f"{usage.get('total_tokens', 0)} tokens · "
                f"{len(done.get('retrieved_chunks', []))} chunks"
            )

    # Refresh the cached message list so the next turn includes this exchange.
    try:
        st.session_state[f"messages_{session['id']}"] = client.list_messages(session["id"])
    except ApiError:
        pass


def _render_citations(citations: list[dict], with_chunk_text: bool = False) -> None:
    """Render citations as expandable blocks, optionally with chunk text."""
    if not citations:
        return
    with st.expander(f"📚 {len(citations)} cited source(s)"):
        for cit in citations:
            file_name = cit.get("file_name", "unknown")
            page = cit.get("page", "?")
            chunk_id = cit.get("chunk_id", "")
            st.markdown(f"**`{file_name}`** — page {page}")
            if with_chunk_text and cit.get("snippet"):
                st.caption(cit["snippet"])
            if chunk_id:
                st.caption(f"chunk: `{chunk_id}`")


if __name__ == "__main__":
    main()
