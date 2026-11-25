"""Query rewriting for follow-up questions.

Combines the current question with the last N chat messages and asks the LLM to
produce a single standalone search query, resolving pronouns and references so
retrieval works without conversation context. Falls back to the original
question when there is no history or the rewrite fails.
"""
from __future__ import annotations

from collections.abc import Sequence

from app.core.config import settings
from app.core.logging import get_logger
from app.services.llm import LLMProvider
from app.services.prompts import REWRITE_SYSTEM_PROMPT

logger = get_logger(__name__)


class QueryRewriter:
    """Rewrite a follow-up question plus history into a standalone query."""

    def __init__(
        self, llm: LLMProvider, history_turns: int | None = None
    ) -> None:
        self._llm = llm
        self._history_turns = history_turns or settings.HISTORY_TURNS

    def build_history_text(self, history: Sequence[dict[str, str]]) -> str:
        """Render the last N messages as a compact transcript."""
        if not history:
            return ""
        recent = list(history)[-self._history_turns :]
        lines: list[str] = []
        for msg in recent:
            role = msg.get("role", "user")
            content = msg.get("content", "").strip().replace("\n", " ")
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def needs_rewrite(self, history: Sequence[dict[str, str]]) -> bool:
        """Return ``True`` if there is history worth folding into the query."""
        return len(history) > 0

    async def rewrite(
        self, question: str, history: Sequence[dict[str, str]]
    ) -> str:
        """Return a standalone query for ``question`` given ``history``.

        If there is no history or the LLM call fails, the original ``question``
        is returned unchanged.
        """
        if not self.needs_rewrite(history):
            return question

        history_text = self.build_history_text(history)
        user_prompt = (
            f"Prior conversation:\n{history_text}\n\n"
            f"Latest question: {question}\n\n"
            "Standalone search query:"
        )
        messages = [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        try:
            rewritten = await self._llm.generate(messages)
        except Exception as exc:
            logger.warning("query rewrite failed; using original", extra={"error": str(exc)})
            return question

        rewritten = rewritten.strip().strip('"').strip("'")
        return rewritten or question
