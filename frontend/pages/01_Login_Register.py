"""Login / Register page.

Tabs for signing in and creating an account. On success, tokens are stored in
session state and the user is sent to the Knowledge Bases page. Clean error
messages are shown for 401/409/429/500 and other failures.
"""
from __future__ import annotations

import os
import sys

# Make the frontend root importable when run as a Streamlit page.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

import common
from api_client import ApiError


def main() -> None:
    """Render the login/register page."""
    st.set_page_config(page_title="Login · RAG Platform", page_icon="📚", layout="wide")
    common.apply_styles()
    client = common.get_client()

    # Already authenticated? Offer a quick continue.
    if common.is_authenticated():
        st.success("You're already signed in.")
        if st.button("Go to Knowledge Bases", use_container_width=True):
            st.switch_page("pages/02_Knowledge_Bases.py")
        if st.button("⎋ Log out", use_container_width=True):
            common.clear_auth()
            st.rerun()
        return

    st.title("🔐 Sign in")
    st.caption("Log in to your RAG Platform account or create a new one.")

    tab_login, tab_register = st.tabs(["Log in", "Register"])

    with tab_login:
        _login_form(client)

    with tab_register:
        _register_form(client)


def _login_form(client) -> None:
    """Render and handle the login form."""
    with st.form("login_form"):
        email = st.text_input("Email", type="password", placeholder="you@example.com")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in", type="primary")
    if not submitted:
        return
    if not email or not password:
        st.error("Please enter your email and password.")
        return
    try:
        data = client.login(email, password)
        common.set_auth(data["access_token"], data["refresh_token"], data["user"])
        st.success("Logged in successfully.")
        st.switch_page("pages/02_Knowledge_Bases.py")
    except ApiError as exc:
        st.error(exc.message)


def _register_form(client) -> None:
    """Render and handle the registration form."""
    with st.form("register_form"):
        email = st.text_input("Email", placeholder="you@example.com")
        password = st.text_input("Password", type="password")
        full_name = st.text_input("Full name (optional)")
        submitted = st.form_submit_button("Create account", type="primary")
    if not submitted:
        return
    if not email or not password:
        st.error("Email and password are required.")
        return
    if len(password) < 8:
        st.error("Password must be at least 8 characters long.")
        return
    try:
        data = client.register(email, password, full_name or None)
        common.set_auth(data["access_token"], data["refresh_token"], data["user"])
        st.success("Account created. Welcome!")
        st.switch_page("pages/02_Knowledge_Bases.py")
    except ApiError as exc:
        st.error(exc.message)


if __name__ == "__main__":
    main()
