"""Tracing configuration. No network — asserts on os.environ only."""

from __future__ import annotations

import os

import pytest

from finrag.config import Settings
from finrag.observability.tracing import configure_tracing


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "LANGSMITH_TRACING",
        "LANGCHAIN_TRACING_V2",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
    ):
        monkeypatch.delenv(var, raising=False)


class TestConfigureTracing:
    def test_off_by_default(self):
        status = configure_tracing(Settings(_env_file=None))
        assert status.langsmith_enabled is False
        assert os.environ["LANGSMITH_TRACING"] == "false"

    def test_requested_with_key_exports_both_spellings(self):
        # langsmith resolves LANGSMITH_* first but parts of it read LANGCHAIN_TRACING_V2.
        settings = Settings(_env_file=None, LANGSMITH_TRACING=True, LANGSMITH_API_KEY="ls-test-key")
        status = configure_tracing(settings)
        assert status.langsmith_enabled is True
        assert os.environ["LANGSMITH_TRACING"] == "true"
        assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
        assert os.environ["LANGSMITH_API_KEY"] == "ls-test-key"

    def test_requested_without_key_stays_off(self, caplog):
        # The dangerous case: tracing looks configured but silently records nothing.
        settings = Settings(_env_file=None, LANGSMITH_TRACING=True)
        status = configure_tracing(settings)
        assert status.langsmith_enabled is False
        assert "LANGSMITH_API_KEY is missing" in status.reason
        assert os.environ["LANGSMITH_TRACING"] == "false"
        assert "LANGSMITH_API_KEY is not set" in caplog.text

    def test_project_name_is_exported(self):
        settings = Settings(
            _env_file=None,
            LANGSMITH_TRACING=True,
            LANGSMITH_API_KEY="ls-test-key",
            langsmith_project="finrag-dev",
        )
        assert configure_tracing(settings).project == "finrag-dev"
        assert os.environ["LANGSMITH_PROJECT"] == "finrag-dev"

    def test_is_idempotent(self):
        settings = Settings(_env_file=None, LANGSMITH_TRACING=True, LANGSMITH_API_KEY="ls-test-key")
        assert configure_tracing(settings) == configure_tracing(settings)

    def test_langsmith_env_cache_is_cleared(self):
        """langsmith memoises env reads; a later write must still be visible."""
        from langsmith.utils import get_env_var

        os.environ["LANGSMITH_PROJECT"] = "stale"
        assert get_env_var("PROJECT") == "stale"  # prime the lru_cache

        configure_tracing(
            Settings(
                _env_file=None,
                LANGSMITH_TRACING=True,
                LANGSMITH_API_KEY="ls-test-key",
                langsmith_project="fresh",
            )
        )
        assert get_env_var("PROJECT") == "fresh"
