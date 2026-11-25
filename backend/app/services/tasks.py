"""Background task backend abstraction.

The default :class:`BackgroundTaskBackend` uses FastAPI's in-process
``BackgroundTasks`` (fine for a single worker / development). Implement
:class:`TaskBackend` against Celery+Redis for multi-worker deployments without
changing the router's call sites.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from typing import Any


class TaskBackend(ABC):
    """Interface for enqueueing background work."""

    @abstractmethod
    def enqueue(
        self, background: Any, func: Callable[..., Coroutine[Any, Any, Any]], *args: Any
    ) -> None:
        """Schedule ``func(*args)`` to run in the background via ``background``."""
        raise NotImplementedError


class BackgroundTaskBackend(TaskBackend):
    """Run tasks via FastAPI ``BackgroundTasks`` (in-process, single worker)."""

    def enqueue(
        self, background: Any, func: Callable[..., Coroutine[Any, Any, Any]], *args: Any
    ) -> None:
        """Delegate to the FastAPI ``BackgroundTasks`` object passed in."""
        background.add_task(func, *args)


# Module-level singleton. Swap for a Celery/Redis backend in production.
task_backend: TaskBackend = BackgroundTaskBackend()
