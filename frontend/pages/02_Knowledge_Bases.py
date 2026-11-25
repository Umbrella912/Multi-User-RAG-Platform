"""Knowledge Bases page: list, create, and delete knowledge bases.

Selecting a KB stores it as the current KB in session state so the Documents
and Chat pages can use it directly.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

import common
from api_client import ApiError


def main() -> None:
    """Render the knowledge bases management page."""
    st.set_page_config(page_title="Knowledge Bases · RAG Platform", page_icon="📚", layout="wide")
    common.apply_styles()
    common.require_auth()
    common.render_sidebar()
    client = common.get_client()

    st.title("🗂️ Knowledge Bases")
    st.caption("Create knowledge bases and load documents into them.")

    # --- Create form --------------------------------------------------------
    with st.expander("➕ Create a knowledge base", expanded=False):
        with st.form("create_kb"):
            name = st.text_input("Name", placeholder="e.g. Product Docs")
            description = st.text_area("Description (optional)")
            c1, c2 = st.columns(2)
            chunk_size = c1.number_input("Chunk size", min_value=64, max_value=4000, value=800, step=64)
            chunk_overlap = c2.number_input("Chunk overlap", min_value=0, max_value=1000, value=160, step=16)
            submitted = st.form_submit_button("Create", type="primary")
        if submitted:
            if not name:
                st.error("A name is required.")
            else:
                try:
                    kb = client.create_kb(name, description or None, chunk_size, chunk_overlap)
                    st.session_state.current_kb = kb
                    st.success(f"Created **{kb['name']}**.")
                    st.rerun()
                except ApiError as exc:
                    st.error(exc.message)

    st.divider()

    # --- List ---------------------------------------------------------------
    try:
        kbs = client.list_kbs()
    except ApiError as exc:
        st.error(exc.message)
        return

    if not kbs:
        st.info("No knowledge bases yet. Create one above to get started.")
        return

    st.subheader("Your knowledge bases")
    for kb in kbs:
        _render_kb_card(client, kb)


def _render_kb_card(client, kb: dict) -> None:
    """Render a single KB as a card with actions."""
    is_current = st.session_state.get("current_kb", {}).get("id") == kb["id"]
    header = f"{kb['name']}"
    if is_current:
        header += "  ✅ current"
    common.card(header)
    st.markdown(f"**Description:** {kb.get('description') or '—'}  ")
    st.markdown(
        f"**Chunking:** {kb['chunk_size']} chars · overlap {kb['chunk_overlap']} · "
        f"model `{kb['embedding_model']}`  "
    )
    st.caption(f"Created {kb['created_at'][:10]}")

    col1, col2 = st.columns([1, 1])
    if col1.button("Use", use_container_width=True, key=f"use_{kb['id']}"):
        st.session_state.current_kb = kb
        st.rerun()
    if col2.button("Delete", use_container_width=True, key=f"del_{kb['id']}"):
        if st.checkbox("Confirm delete (cascades documents)", key=f"confirm_{kb['id']}"):
            try:
                client.delete_kb(kb["id"])
                if st.session_state.get("current_kb", {}).get("id") == kb["id"]:
                    st.session_state.pop("current_kb", None)
                st.success(f"Deleted **{kb['name']}**.")
                st.rerun()
            except ApiError as exc:
                st.error(exc.message)
    common.end_card()


if __name__ == "__main__":
    main()
