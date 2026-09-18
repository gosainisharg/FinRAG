"""Tracing setup.

LangChain and LangGraph trace automatically once the right environment variables are
present — there is no code to add at each call site. The catch is that they read
``os.environ``, while our configuration lives in a pydantic-settings object loaded from
``.env``. Nothing bridges those two automatically, so a key sitting in ``.env`` would
otherwise do nothing and tracing would stay silently off.

:func:`configure_tracing` is that bridge. Call it once, early, before any model is built.

Phoenix / OpenTelemetry export is Phase 9 and not wired here yet.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from finrag.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TracingStatus:
    """What tracing actually ended up doing, as opposed to what was requested."""

    langsmith_enabled: bool
    project: str | None
    reason: str

    def __str__(self) -> str:
        state = "on" if self.langsmith_enabled else "off"
        target = f" → project {self.project!r}" if self.langsmith_enabled else ""
        return f"LangSmith tracing {state}{target} ({self.reason})"


def _clear_langsmith_env_cache() -> None:
    """Drop langsmith's memoised view of the environment.

    ``langsmith.utils.get_env_var`` is ``lru_cache``d, so a variable set after the library
    has read it once is ignored for the life of the process. Harmless in a fresh script,
    but in tests or a notebook the first read wins and later configuration is silently
    dropped. Clearing the cache makes the call order not matter.
    """
    try:
        from langsmith.utils import get_env_var

        get_env_var.cache_clear()
    except (ImportError, AttributeError):  # pragma: no cover - defensive
        logger.debug("langsmith env cache not clearable; continuing")


def configure_tracing(settings: Settings | None = None) -> TracingStatus:
    """Export tracing configuration into ``os.environ`` and report what took effect.

    Enabling is deliberately conservative: tracing only turns on when it is both requested
    *and* credentialed. Asking for it without a key logs a warning and leaves it off,
    rather than letting LangChain fail on every call or, worse, appear to work.
    """
    settings = settings or get_settings()

    if not settings.langsmith_tracing:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        _clear_langsmith_env_cache()
        return TracingStatus(False, None, "not requested")

    if settings.langsmith_api_key is None:
        logger.warning(
            "LANGSMITH_TRACING is true but LANGSMITH_API_KEY is not set. "
            "Tracing stays off. Add the key to .env, or set LANGSMITH_TRACING=false."
        )
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        _clear_langsmith_env_cache()
        return TracingStatus(False, None, "requested but LANGSMITH_API_KEY is missing")

    # Both spellings: langsmith resolves LANGSMITH_* first and falls back to LANGCHAIN_*,
    # but parts of the package still read LANGCHAIN_TRACING_V2 directly.
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key.get_secret_value()
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    _clear_langsmith_env_cache()

    return TracingStatus(True, settings.langsmith_project, "enabled")
