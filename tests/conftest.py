"""Test isolation from the ambient environment.

Installing DeepEval added a pytest plugin that calls `load_dotenv()` at session start,
which exports the developer's entire `.env` into `os.environ`. Settings then pick it up
even when a test explicitly passes `_env_file=None`, so a suite that passed on a clean
machine failed on one with a configured `.env` — and would fail differently depending on
what that file happened to contain.

Tests assert on defaults and on values they set themselves. Anything the ambient
environment supplies is noise, so it is removed for every test.
"""

from __future__ import annotations

import os

import pytest

_AMBIENT_PREFIXES = ("FINRAG_",)
_AMBIENT_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGSMITH_TRACING",
    "LANGCHAIN_TRACING_V2",
)


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch):
    for key in list(os.environ):
        if key.startswith(_AMBIENT_PREFIXES) or key in _AMBIENT_KEYS:
            monkeypatch.delenv(key, raising=False)

    # Settings are cached process-wide; a cached instance built from the ambient
    # environment would survive the deletions above.
    from finrag.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
