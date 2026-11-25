"""Streamlit entry point for the Multi-User RAG Platform.

Sets up the page config and global styling, initializes the :class:`ApiClient`
in session state, enforces the auth guard, and renders a home dashboard. The
multi-page navigation is provided by Streamlit's built-in ``pages/`` discovery
(Login, Knowledge Bases, Documents, Chat, Settings).
"""
from __future__ import annotations

import streamlit as st

import common


def main() -> None:
    """Application entry: page config, styles, auth guard, home dashboard."""
    st.set_page_config(
        page_title="Multi-User RAG Platform",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    common.apply_styles()

    # Ensure the API client exists for every run.
    client = common.get_client()

    # Auth guard: unauthenticated users are redirected to the login page.
    common.require_auth()
    common.render_sidebar()

    # --- Home dashboard ------------------------------------------------------
    st.title("📚 Multi-User RAG Platform")
    st.caption("Upload documents, build knowledge bases, and chat with your data.")

    user = st.session_state.get("user", {})
    name = user.get("full_name") or user.get("email", "")
    st.success(f"Welcome back, **{name}**.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Knowledge bases", _count_kbs(client))
    with col2:
        st.metric("Chat sessions", _count_sessions(client))
    with col3:
        st.metric("Backend", _backend_health(client))

    st.divider()
    st.markdown(
        "Use the **pages** in the sidebar to manage your knowledge bases, "
        "upload documents, start a chat, or adjust retrieval settings."
    )

    # Quick-action buttons.
    left, right = st.columns(2)
    if left.button("🗂️ Manage Knowledge Bases", use_container_width=True):
        st.switch_page("pages/02_Knowledge_Bases.py")
    if right.button("💬 Open Chat", use_container_width=True):
        st.switch_page("pages/04_Chat.py")

    # Logout.
    if st.button("⎋ Log out", use_container_width=True):
        common.clear_auth()
        st.rerun()


def _count_kbs(client) -> int:
    """Return the number of knowledge bases, or 0 on error."""
    try:
        return len(client.list_kbs())
    except Exception:
        return 0


def _count_sessions(client) -> int:
    """Return the number of chat sessions, or 0 on error."""
    try:
        return len(client.list_sessions())
    except Exception:
        return 0


def _backend_health(client) -> str:
    """Return a health indicator string for the backend."""
    try:
        response = httpx_get(f"{client.base_url}/health/live")
        return "Online" if response.status_code == 200 else "Degraded"
    except Exception:
        return "Offline"


def httpx_get(url: str):
    """Lightweight health check without going through the auth client."""
    import httpx

    return httpx.get(url, timeout=5.0)


if __name__ == "__main__":
    main()
