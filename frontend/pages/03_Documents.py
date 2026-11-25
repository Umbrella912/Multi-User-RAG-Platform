"""Documents page: upload, monitor status (auto-refresh), and delete documents.

Uploads are streamed to the backend, then ingestion runs in the background.
Status is polled on an interval so the user sees ``pending -> processing ->
ready`` (or ``failed``) progress without manually refreshing.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

import common
from api_client import ApiError

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "md"}


def main() -> None:
    """Render the documents management page."""
    st.set_page_config(page_title="Documents · RAG Platform", page_icon="📚", layout="wide")
    common.apply_styles()
    common.require_auth()
    common.render_sidebar()
    client = common.get_client()

    st.title("📄 Documents")
    st.caption("Upload documents into a knowledge base and track ingestion.")

    kb = _select_kb(client)
    if not kb:
        st.warning("Select or create a knowledge base first (Knowledge Bases page).")
        return

    # --- Upload -------------------------------------------------------------
    st.subheader("Upload")
    uploaded = st.file_uploader(
        "Choose a file (PDF, DOCX, TXT, MD)",
        type=[ext for ext in ALLOWED_EXTENSIONS],
        key="doc_uploader",
    )
    if uploaded is not None:
        file_type = _extension(uploaded.name)
        if st.button("Upload", type="primary", key="upload_btn"):
            try:
                data = client.upload_document(
                    kb["id"], uploaded.getvalue(), uploaded.name, file_type
                )
                st.success(
                    f"Uploaded **{data['file_name']}** — status: {data['status']}."
                )
                st.rerun()
            except ApiError as exc:
                st.error(exc.message)

    st.divider()

    # --- Status with auto-refresh -------------------------------------------
    st.subheader(f"Documents in {kb['name']}")
    auto_refresh = st.checkbox("Auto-refresh status", value=True, key="auto_refresh")
    if auto_refresh:
        from streamlit_autorefresh import st_autorefresh

        st_autorefresh(interval=3000, key="doc_refresh")

    try:
        documents = client.list_documents(kb["id"])
    except ApiError as exc:
        st.error(exc.message)
        return

    if not documents:
        st.info("No documents yet. Upload one above.")
        return

    pending = any(d["status"] in ("pending", "processing") for d in documents)
    if pending:
        st.info("Documents are being ingested. Status will update automatically.")

    for doc in documents:
        _render_document(client, doc)


def _select_kb(client):
    """Pick the current KB (from session state) or via a dropdown."""
    current = st.session_state.get("current_kb")
    try:
        kbs = client.list_kbs()
    except ApiError as exc:
        st.error(exc.message)
        return None
    if not kbs:
        return None
    if current is None or current.get("id") not in {k["id"] for k in kbs}:
        options = {f"{k['name']}": k for k in kbs}
        choice = st.selectbox("Knowledge base", list(options.keys()), key="kb_select")
        return options[choice]
    return current


def _render_document(client, doc: dict) -> None:
    """Render a document row with status and a delete action."""
    status = doc["status"]
    icon = {"ready": "✅", "pending": "⏳", "processing": "⚙️", "failed": "❌"}.get(status, "•")
    common.card(f"{icon} {doc['file_name']}")
    st.markdown(f"**Type:** {doc['file_type']} · **Status:** `{status}`")
    if status == "ready":
        st.markdown(
            f"**Chunks:** {doc['chunk_count']} · "
            f"**Chars:** {doc.get('char_count') or '—'} · "
            f"**Pages:** {doc.get('page_count') or '—'}"
        )
    if status == "failed" and doc.get("error_message"):
        st.error(f"Ingestion failed: {doc['error_message']}")

    if st.button("Delete", use_container_width=True, key=f"del_doc_{doc['id']}"):
        try:
            client.delete_document(doc["id"])
            st.success(f"Deleted **{doc['file_name']}** and its vectors.")
            st.rerun()
        except ApiError as exc:
            st.error(exc.message)
    common.end_card()


def _extension(filename: str) -> str:
    """Return the lowercase extension (without dot) for ``filename``."""
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[1].lower()


if __name__ == "__main__":
    main()
