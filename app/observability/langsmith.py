"""LangSmith observability helpers with safe no-op behavior."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator

from app.config import settings

logger = logging.getLogger(__name__)

_configured = False
_available = False


def configure_langsmith() -> bool:
    """Configure LangSmith environment flags once per process."""
    global _configured, _available
    if _configured:
        return _available
    _configured = True

    if not settings.langsmith_enabled:
        _available = False
        logger.info("LangSmith tracing disabled (LANGSMITH_ENABLED=false)")
        return False
    if not settings.langsmith_api_key.strip():
        _available = False
        logger.warning("LangSmith enabled but LANGSMITH_API_KEY is empty; tracing disabled.")
        return False

    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    os.environ["LANGSMITH_TRACING_V2"] = "true" if settings.langsmith_tracing_v2 else "false"

    try:
        import langsmith  # noqa: F401
    except Exception as exc:
        _available = False
        logger.warning("LangSmith package unavailable; tracing disabled. err=%s", exc)
        return False

    _available = True
    logger.info(
        "LangSmith tracing enabled project=%s endpoint=%s",
        settings.langsmith_project,
        settings.langsmith_endpoint,
    )
    return True


@contextmanager
def trace_span(
    name: str,
    *,
    run_type: str = "chain",
    inputs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> Iterator[Any]:
    """Create a LangSmith span when available; otherwise no-op."""
    if not configure_langsmith():
        yield None
        return

    try:
        from langsmith.run_helpers import trace

        manager = trace(
            name=name,
            run_type=run_type,
            inputs=inputs or {},
            metadata=metadata or {},
            tags=tags or [],
        )
    except Exception:
        logger.debug("LangSmith trace_span fell back to no-op for %s", name, exc_info=True)
        yield None
        return

    # Do not wrap the yielded body in try/except that yields again — that
    # produces "generator didn't stop after throw()" when the body raises.
    with manager as run_tree:
        yield run_tree

