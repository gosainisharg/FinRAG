"""Unit tests for the provider factory. No network, no credentials, no cost."""

from __future__ import annotations

import pytest

from finrag.config import ModelRole, Provider, Settings
from finrag.llm import (
    ModelConfigError,
    _credential_for,
    build_chat_model,
    parse_model_spec,
    supports_sampling_params,
)


class TestParseModelSpec:
    def test_splits_provider_and_model(self):
        assert parse_model_spec("openai:gpt-4o-mini") == (Provider.OPENAI, "gpt-4o-mini")

    def test_llama_is_a_first_class_provider(self):
        # LLaMA is the model; Ollama is only the runtime. The config surface says "llama".
        assert parse_model_spec("llama:llama3.2") == (Provider.LLAMA, "llama3.2")

    def test_model_names_may_contain_colons(self):
        # Ollama tags look like "llama3.2:3b" — only the first colon separates the provider.
        assert parse_model_spec("llama:llama3.2:3b") == (Provider.LLAMA, "llama3.2:3b")

    def test_is_case_and_whitespace_insensitive_on_provider(self):
        assert parse_model_spec(" OpenAI : gpt-4o ") == (Provider.OPENAI, "gpt-4o")

    @pytest.mark.parametrize(
        "spec, expected_message",
        [
            ("gpt-4o-mini", "must be 'provider:model'"),
            ("openai:", "missing a model name"),
            ("ollama:llama3.2", "Unknown provider 'ollama'"),
            ("gemini:flash", "Unknown provider 'gemini'"),
        ],
    )
    def test_rejects_malformed_specs(self, spec, expected_message):
        with pytest.raises(ModelConfigError, match=expected_message):
            parse_model_spec(spec)

    def test_ollama_is_rejected_as_a_provider_name(self):
        # Guards the naming decision: the runtime must not leak into configuration.
        with pytest.raises(ModelConfigError):
            parse_model_spec("ollama:llama3.2")


class TestCredentialResolution:
    def test_llama_needs_no_credential(self):
        settings = Settings(_env_file=None)
        assert _credential_for(Provider.LLAMA, settings) is None

    def test_missing_key_names_the_env_var(self):
        settings = Settings(_env_file=None, OPENAI_API_KEY=None)
        with pytest.raises(ModelConfigError, match="OPENAI_API_KEY is not set"):
            _credential_for(Provider.OPENAI, settings)

    def test_returns_the_unwrapped_secret(self):
        settings = Settings(_env_file=None, ANTHROPIC_API_KEY="sk-test-value")
        assert _credential_for(Provider.ANTHROPIC, settings) == "sk-test-value"

    def test_secret_is_not_exposed_by_repr(self):
        settings = Settings(_env_file=None, OPENAI_API_KEY="sk-leak-me")
        assert "sk-leak-me" not in repr(settings)


class TestSettings:
    def test_every_role_resolves_to_a_valid_spec(self):
        settings = Settings(_env_file=None)
        for role in ModelRole:
            provider, model = parse_model_spec(settings.model_for(role))
            assert isinstance(provider, Provider)
            assert model

    def test_judge_defaults_to_a_different_family_than_generator(self):
        # Self-preference bias: a model scoring its own output inflates the score.
        settings = Settings(_env_file=None)
        generator, _ = parse_model_spec(settings.model_for(ModelRole.GENERATOR))
        judge, _ = parse_model_spec(settings.model_for(ModelRole.JUDGE))
        assert generator is not judge

    @pytest.mark.parametrize("field", ["hybrid_dense_weight", "cache_similarity_threshold"])
    def test_unit_interval_fields_reject_out_of_range(self, field):
        with pytest.raises(ValueError):
            Settings(_env_file=None, **{field: 1.5})

    def test_env_prefix_overrides_model_defaults(self, monkeypatch):
        monkeypatch.setenv("FINRAG_GENERATOR_MODEL", "llama:llama3.2")
        settings = Settings(_env_file=None)
        assert settings.model_for(ModelRole.GENERATOR) == "llama:llama3.2"


class TestLocalOnly:
    def test_disabled_by_default(self):
        settings = Settings(_env_file=None)
        assert settings.local_only is False
        assert settings.model_for(ModelRole.GENERATOR) != settings.local_model

    def test_collapses_every_role_onto_the_local_model(self):
        settings = Settings(_env_file=None, local_only=True)
        for role in ModelRole:
            assert settings.model_for(role) == settings.local_model

    def test_overrides_per_role_environment_variables(self, monkeypatch):
        # The switch must win, or a stray variable could still bill a hosted provider.
        monkeypatch.setenv("FINRAG_GENERATOR_MODEL", "anthropic:claude-sonnet-5")
        monkeypatch.setenv("FINRAG_LOCAL_ONLY", "true")
        settings = Settings(_env_file=None)
        assert settings.model_for(ModelRole.GENERATOR) == settings.local_model

    def test_local_only_needs_no_credentials(self):
        settings = Settings(_env_file=None, local_only=True)
        for role in ModelRole:
            provider, _ = parse_model_spec(settings.model_for(role))
            assert _credential_for(provider, settings) is None


class TestSamplingParameterSupport:
    """Anthropic removed temperature/top_p/top_k on its current frontier models."""

    @pytest.mark.parametrize(
        "model", ["claude-sonnet-5", "claude-opus-5", "claude-opus-4-8", "claude-fable-5-1"]
    )
    def test_current_anthropic_models_reject_sampling(self, model):
        assert supports_sampling_params(Provider.ANTHROPIC, model) is False

    @pytest.mark.parametrize("model", ["claude-sonnet-4-6", "claude-haiku-4-5"])
    def test_older_anthropic_models_still_accept_sampling(self, model):
        assert supports_sampling_params(Provider.ANTHROPIC, model) is True

    @pytest.mark.parametrize(
        "provider, model",
        [(Provider.OPENAI, "gpt-5.4"), (Provider.LLAMA, "llama3.1:8b")],
    )
    def test_other_providers_are_unaffected(self, provider, model):
        assert supports_sampling_params(provider, model) is True

    def test_temperature_is_omitted_for_models_that_reject_it(self):
        settings = Settings(_env_file=None, ANTHROPIC_API_KEY="sk-test")
        model = build_chat_model("anthropic:claude-sonnet-5", settings)
        assert getattr(model, "temperature", None) is None

    def test_temperature_is_sent_where_supported(self):
        settings = Settings(_env_file=None, OPENAI_API_KEY="sk-test", temperature=0.0)
        model = build_chat_model("openai:gpt-4o-mini", settings)
        assert model.temperature == 0.0

    def test_langchain_openai_strips_temperature_for_gpt5_itself(self):
        """Documents why no guard is needed on the OpenAI side.

        langchain-openai drops `temperature` for the gpt-5 family internally, so our
        factory can pass it unconditionally there. langchain-anthropic does not do the
        equivalent, which is why `supports_sampling_params` exists. If this test ever
        fails, langchain-openai changed and the guard may need to cover OpenAI too.
        """
        settings = Settings(_env_file=None, OPENAI_API_KEY="sk-test", temperature=0.7)
        assert build_chat_model("openai:gpt-5.4", settings).temperature is None
