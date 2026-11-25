"""Shared helpers for the Streamlit frontend.

Holds backend URL resolution, professional styling, and the auth-guard /
session-state helpers used by every page. All Streamlit calls live inside
functions so importing this module never renders anything.
"""
from __future__ import annotations

import os

import streamlit as st

#: Backend base URL. In Docker this is http://backend:8000; locally http://localhost:8000.
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

#: Professional dark-accent styling applied on every page.
CSS = """
<style>
    /* Global typography and spacing. */
    .stApp { font-family: "Inter", -apple-system, "Segoe UI", Roboto, sans-serif; }

    /* Primary heading color. */
    h1, h2, h3 { color: #1f2937; }

    /* Sidebar styling. */
    [data-testid="stSidebar"] {
        background: #0f172a;
        color: #e2e8f0;
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 { color: #f8fafc; }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stCheckbox label { color: #cbd5e1; }

    /* Cards via containers. */
    .stCard {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        background: #ffffff;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
    }
    .stCard h3 { margin: 0 0 0.5rem 0; }

    /* Chat message bubbles. */
    .stChatMessage {
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        margin: 0.4rem 0;
    }
    .stChatMessage.user {
        background: #2563eb;
        color: #ffffff;
    }
    .stChatMessage.assistant {
        background: #f1f5f9;
        color: #0f172a;
        border: 1px solid #e2e8f0;
    }

    /* Streaming caret. */
    .stream-caret::after {
        content: "▍";
        animation: blink 1s step-end infinite;
        color: #2563eb;
    }
    @keyframes blink { 50% { opacity: 0; } }

    /* Hide the default Streamlit footer spacing a touch. */
    .stMainBlock { padding: 1.5rem 2rem; }

    /* Success / error containers. */
    .stSuccess > div, .stError > div { border-radius: 8px; }
</style>
"""


def apply_styles() -> None:
    """Inject the global stylesheet into the page."""
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def get_client():
    """Return the :class:`ApiClient` from session state, creating it if needed."""
    if "client" not in st.session_state:
        from api_client import ApiClient

        st.session_state.client = ApiClient(BACKEND_URL)
    return st.session_state.client


def set_auth(access_token: str, refresh_token: str, user: dict) -> None:
    """Persist auth tokens and the user object in session state."""
    st.session_state.access_token = access_token
    st.session_state.refresh_token = refresh_token
    st.session_state.user = user
    client = get_client()
    client.set_tokens(access_token, refresh_token)


def clear_auth() -> None:
    """Clear all auth state (logout)."""
    for key in ("access_token", "refresh_token", "user"):
        st.session_state.pop(key, None)
    client = st.session_state.get("client")
    if client is not None:
        client.clear_tokens()


def is_authenticated() -> bool:
    """Return ``True`` if an access token is present in session state."""
    return bool(st.session_state.get("access_token"))


def require_auth() -> bool:
    """Guard every page: redirect to Login if not authenticated.

    Returns ``True`` when authenticated so pages can proceed.
    """
    if is_authenticated():
        return True
    st.switch_page("pages/01_Login_Register.py")
    return False  # unreachable, but keeps the return contract explicit


def render_sidebar(active: str = "") -> None:
    """Render a consistent branded sidebar with navigation and the current user."""
    st.sidebar.title("📚 RAG Platform")
    if is_authenticated():
        user = st.session_state.get("user", {})
        name = user.get("full_name") or user.get("email", "User")
        st.sidebar.caption(f"Signed in as\n**{name}**")
        st.sidebar.divider()
    if st.sidebar.button("🔄 Refresh", use_container_width=True):
        st.rerun()
    st.sidebar.divider()
    st.sidebar.caption("Tip: use the pages in the sidebar to navigate.")


def card(title: str) -> None:
    """Open a styled card container around content."""
    st.markdown(f'<div class="stCard"><h3>{title}</h3>', unsafe_allow_html=True)


def end_card() -> None:
    """Close a card opened by :func:`card`."""
    st.markdown("</div>", unsafe_allow_html=True)
