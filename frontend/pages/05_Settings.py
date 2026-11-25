"""Settings page: retrieval parameters and model preferences.

Captures the user's retrieval preferences (top_k, reranker toggle, model
choice) and stores them in session state. These reflect the deployment's
retrieval configuration and are surfaced across the app.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

import common
from api_client import ApiError

#: Model choices offered in the UI (mapped to backend LLM_MODEL / provider).
MODEL_CHOICES = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma-2-9b",
]


def main() -> None:
    """Render the settings page."""
    st.set_page_config(page_title="Settings · RAG Platform", page_icon="📚", layout="wide")
    common.apply_styles()
    common.require_auth()
    common.render_sidebar()

    st.title("⚙️ Settings")
    st.caption("Configure retrieval and generation preferences for your chats.")

    # --- Retrieval ----------------------------------------------------------
    common.card("Retrieval")
    top_k = st.slider(
        "Top-K results per retriever",
        min_value=5,
        max_value=50,
        value=int(st.session_state.get("setting_top_k", 20)),
        step=1,
        help="Number of results fetched from each of the vector and BM25 retrievers.",
    )
    st.session_state.setting_top_k = top_k

    reranker = st.checkbox(
        "Enable cross-encoder reranker",
        value=bool(st.session_state.get("setting_reranker", True)),
        help="Reranks retrieved chunks with a cross-encoder for higher answer quality.",
    )
    st.session_state.setting_reranker = reranker

    threshold = st.slider(
        "Similarity threshold",
        min_value=0.1,
        max_value=0.8,
        value=float(st.session_state.get("setting_threshold", 0.35)),
        step=0.05,
        help="Below this score the model replies that it couldn't find the answer.",
    )
    st.session_state.setting_threshold = threshold

    common.end_card()

    # --- Model --------------------------------------------------------------
    common.card("Generation model")
    model = st.selectbox(
        "LLM model",
        MODEL_CHOICES,
        index=MODEL_CHOICES.index(st.session_state.get("setting_model", MODEL_CHOICES[0]))
        if st.session_state.get("setting_model", MODEL_CHOICES[0]) in MODEL_CHOICES
        else 0,
        help="The model used for answer generation.",
    )
    st.session_state.setting_model = model

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.get("setting_temperature", 0.1)),
        step=0.1,
    )
    st.session_state.setting_temperature = temperature

    common.end_card()

    # --- Account ------------------------------------------------------------
    common.card("Account")
    try:
        user = st.session_state.client.me()
        st.markdown(f"**Email:** {user['email']}  ")
        st.markdown(f"**Full name:** {user.get('full_name') or '—'}  ")
        st.markdown(f"**Active:** {'Yes' if user.get('is_active') else 'No'}")
    except ApiError as exc:
        st.error(exc.message)

    if st.button("⎋ Log out", use_container_width=True):
        common.clear_auth()
        st.rerun()

    st.caption(
        "Note: retrieval parameters are applied by the backend configuration; "
        "the values above are your preferred settings for this session."
    )


if __name__ == "__main__":
    main()
