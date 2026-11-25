"""Typed HTTP client for the FastAPI backend.

Wraps ``httpx`` with JWT token handling and **automatic refresh**: on a 401 the
client refreshes the access token once and retries the original request. Error
responses are normalized into :class:`ApiError` carrying a friendly message
keyed by status code (401 / 429 / 500 etc.). The streaming ``ask_stream``
generator parses the SSE frames emitted by the backend.

The client talks to the backend server-side (from the Streamlit process), so no
CORS is involved — requests go directly over HTTP.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx


class ApiError(Exception):
    """An error returned by the backend, with a friendly message and status code."""

    def __init__(self, status_code: int, detail: str, code: str = "error") -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.code = code

    @property
    def message(self) -> str:
        """A clean, user-facing message appropriate to the status code."""
        return _friendly_message(self.status_code, self.detail)


def _friendly_message(status: int, detail: str) -> str:
    """Map an HTTP status + backend detail to a clean, human message."""
    if status == 401:
        return "Your session has expired. Please log in again."
    if status == 403:
        return "You don't have permission to do that."
    if status == 404:
        return detail or "The requested resource was not found."
    if status == 409:
        return detail or "That already exists."
    if status == 413:
        return "The file is too large. Please choose a smaller file."
    if status == 422:
        return detail or "Invalid input. Please check your fields."
    if status == 429:
        return "Rate limit reached. Please wait a moment and try again."
    if status in (502, 503, 504):
        return "The service is temporarily unavailable. Please try again later."
    if status >= 500:
        return "A server error occurred. Please try again later."
    return detail or f"Request failed (status {status})."


class ApiClient:
    """A typed wrapper around httpx with token storage and auto-refresh."""

    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.access_token: str | None = None
        self.refresh_token: str | None = None

    # ------------------------------------------------------------------ tokens
    def set_tokens(self, access_token: str, refresh_token: str) -> None:
        """Store an access/refresh token pair."""
        self.access_token = access_token
        self.refresh_token = refresh_token

    def clear_tokens(self) -> None:
        """Forget the stored tokens."""
        self.access_token = None
        self.refresh_token = None

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        if extra:
            headers.update(extra)
        return headers

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    # ---------------------------------------------------------------- internals
    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        try:
            body = response.json()
            detail = body.get("detail", response.text)
            code = body.get("code", "error")
        except Exception:
            detail = response.text
            code = "error"
        raise ApiError(response.status_code, str(detail), code)

    def _try_refresh(self) -> bool:
        """Attempt to refresh the access token; return ``True`` on success."""
        if not self.refresh_token:
            return False
        try:
            data = self._request_raw(
                "POST", "/api/auth/refresh",
                json={"refresh_token": self.refresh_token}, auth=False,
            )
            self.set_tokens(data["access_token"], data["refresh_token"])
            return True
        except ApiError:
            self.clear_tokens()
            return False

    def _request_raw(
        self,
        method: str,
        path: str,
        *,
        auth: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Perform a request, refreshing once on 401 when ``auth`` is enabled."""
        url = self._url(path)
        for attempt in range(2):
            headers = self._headers(kwargs.pop("headers", None)) if auth else {
                "Content-Type": "application/json"
            }
            response = httpx.request(
                method, url, headers=headers, timeout=self.timeout, **kwargs
            )
            if response.status_code == 401 and auth and attempt == 0:
                if self._try_refresh():
                    continue  # retry once with the fresh token
                self._raise_for_status(response)
            self._raise_for_status(response)
            return response.json()
        raise ApiError(401, "Authentication failed.")  # pragma: no cover

    # ------------------------------------------------------------------- auth
    def register(self, email: str, password: str, full_name: str | None = None) -> dict[str, Any]:
        """Register a new account and return the token response."""
        payload: dict[str, Any] = {"email": email, "password": password}
        if full_name:
            payload["full_name"] = full_name
        return self._request_raw("POST", "/api/auth/register", json=payload, auth=False)

    def login(self, email: str, password: str) -> dict[str, Any]:
        """Log in and return the token response."""
        return self._request_raw(
            "POST", "/api/auth/login", json={"email": email, "password": password},
            auth=False,
        )

    def me(self) -> dict[str, Any]:
        """Return the currently authenticated user."""
        return self._request_raw("GET", "/api/auth/me")

    # ----------------------------------------------------------- knowledge bases
    def list_kbs(self) -> list[dict[str, Any]]:
        """List the current user's knowledge bases."""
        return self._request_raw("GET", "/api/knowledge-bases")

    def create_kb(
        self,
        name: str,
        description: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> dict[str, Any]:
        """Create a knowledge base and return it."""
        payload: dict[str, Any] = {"name": name}
        if description:
            payload["description"] = description
        if chunk_size:
            payload["chunk_size"] = chunk_size
        if chunk_overlap:
            payload["chunk_overlap"] = chunk_overlap
        return self._request_raw("POST", "/api/knowledge-bases", json=payload)

    def delete_kb(self, kb_id: str) -> None:
        """Delete a knowledge base."""
        self._request_raw("DELETE", f"/api/knowledge-bases/{kb_id}")

    # ---------------------------------------------------------------- documents
    def list_documents(self, kb_id: str, status_filter: str | None = None) -> list[dict[str, Any]]:
        """List documents in a KB, optionally filtered by status."""
        params = {}
        if status_filter:
            params["status"] = status_filter
        return self._request_raw(
            "GET", f"/api/knowledge-bases/{kb_id}/documents", params=params
        )

    def upload_document(
        self, kb_id: str, file_bytes: bytes, file_name: str, file_type: str
    ) -> dict[str, Any]:
        """Upload a document (multipart) and return the pending document row."""
        files = {"file": (file_name, file_bytes, f"application/{file_type}" if file_type != "pdf" else "application/pdf")}
        return self._request_raw(
            "POST", f"/api/knowledge-bases/{kb_id}/documents", files=files
        )

    def get_document(self, doc_id: str) -> dict[str, Any]:
        """Return a single document's status/metadata."""
        return self._request_raw("GET", f"/api/documents/{doc_id}")

    def delete_document(self, doc_id: str) -> None:
        """Delete a document (and its vectors)."""
        self._request_raw("DELETE", f"/api/documents/{doc_id}")

    # -------------------------------------------------------------------- chat
    def list_sessions(self, kb_id: str | None = None) -> list[dict[str, Any]]:
        """List chat sessions, optionally for a single KB."""
        params = {"kb_id": kb_id} if kb_id else None
        return self._request_raw("GET", "/api/chat/sessions", params=params)

    def create_session(self, kb_id: str, title: str | None = None) -> dict[str, Any]:
        """Create a chat session bound to a KB."""
        payload: dict[str, Any] = {"kb_id": kb_id}
        if title:
            payload["title"] = title
        return self._request_raw("POST", "/api/chat/sessions", json=payload)

    def delete_session(self, session_id: str) -> None:
        """Delete a chat session."""
        self._request_raw("DELETE", f"/api/chat/sessions/{session_id}")

    def list_messages(self, session_id: str) -> list[dict[str, Any]]:
        """List messages in a chat session (chronological)."""
        return self._request_raw("GET", f"/api/chat/sessions/{session_id}/messages")

    def ask_sync(
        self, session_id: str, question: str, use_history: bool = True
    ) -> dict[str, Any]:
        """Ask a question non-streaming and return the full response."""
        return self._request_raw(
            "POST",
            f"/api/chat/sessions/{session_id}/ask-sync",
            json={"question": question, "use_history": use_history},
        )

    def ask_stream(
        self, session_id: str, question: str, use_history: bool = True
    ) -> Iterator[dict[str, Any]]:
        """Stream a RAG answer as SSE event dicts.

        Yields ``{"type": "rewrite"|"citation"|"token"|"done"|"error", ...}``
        frames parsed from the ``data:`` lines. Refreshes the token once on 401.
        Raises :class:`ApiError` for non-SSE error responses (429/500/...).
        """
        url = self._url(f"/api/chat/sessions/{session_id}/ask")
        payload = {"question": question, "use_history": use_history}
        stream_timeout = httpx.Timeout(600.0, connect=10.0)

        for attempt in range(2):
            headers = self._headers()
            try:
                with httpx.Client(timeout=stream_timeout) as client:
                    with client.stream(
                        "POST", url, json=payload, headers=headers
                    ) as response:
                        if response.status_code == 401 and attempt == 0:
                            if self._try_refresh():
                                continue  # retry the stream with a fresh token
                            self._raise_for_status(response)
                        self._raise_for_status(response)

                        for line in response.iter_lines():
                            if not line:
                                continue
                            if line.startswith("data: "):
                                yield json.loads(line[len("data: "):])
                        return
            except ApiError as exc:
                # A 401 that survived the refresh attempt is fatal; re-raise.
                if exc.status_code == 401 and attempt == 0 and self.refresh_token:
                    if self._try_refresh():
                        continue
                raise
            except httpx.HTTPError as exc:
                raise ApiError(500, f"Network error: {exc}", "network_error") from exc

        raise ApiError(401, "Authentication failed.")  # pragma: no cover
