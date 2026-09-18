"""Chat-model factory for the three swappable backends.

Everything in FinRAG that needs an LLM asks for one *by role*, never by provider::

    model = get_chat_model(ModelRole.GRADER)

which means swapping OpenAI for Anthropic for a locally-served LLaMA is an environment
variable, not a code change. No node ever imports a provider SDK directly.
"""

from __future__ import annotations

from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from finrag.config import ModelRole, Provider, Settings, get_settings

# LLaMA is the model; Ollama is the runtime that serves it. LangChain identifies backends by
# runtime, so this is where the distinction is reconciled — and the only place it appears.
_RUNTIME_FOR_PROVIDER: dict[Provider, str] = {
    Provider.OPENAI: "openai",
    Provider.ANTHROPIC: "anthropic",
    Provider.LLAMA: "ollama",
}


class ModelConfigError(RuntimeError):
    """Raised for a malformed model spec or a missing credential."""


def parse_model_spec(spec: str) -> tuple[Provider, str]:
    """Split a ``"provider:model"`` string into its parts.

    >>> parse_model_spec("anthropic:claude-sonnet-5")
    (<Provider.ANTHROPIC: 'anthropic'>, 'claude-sonnet-5')
    """
    if ":" not in spec:
        raise ModelConfigError(
            f"Model spec {spec!r} must be 'provider:model', "
            f"e.g. 'openai:gpt-4o-mini'. Valid providers: "
            f"{', '.join(p.value for p in Provider)}."
        )

    raw_provider, _, model_name = spec.partition(":")
    raw_provider, model_name = raw_provider.strip().lower(), model_name.strip()

    if not model_name:
        raise ModelConfigError(f"Model spec {spec!r} is missing a model name after the colon.")

    try:
        provider = Provider(raw_provider)
    except ValueError:
        raise ModelConfigError(
            f"Unknown provider {raw_provider!r} in spec {spec!r}. "
            f"Valid providers: {', '.join(p.value for p in Provider)}."
        ) from None

    return provider, model_name


def _credential_for(provider: Provider, settings: Settings) -> str | None:
    """Return the API key for a provider, or None if the provider needs no key.

    Fails loudly rather than letting the provider SDK raise a vaguer error deeper in the
    stack, where the message would not say which *role* was misconfigured.
    """
    if provider is Provider.LLAMA:
        # Served locally; no credential. Reachability is checked at call time.
        return None

    secret = {
        Provider.OPENAI: settings.openai_api_key,
        Provider.ANTHROPIC: settings.anthropic_api_key,
    }[provider]

    if secret is None:
        env_var = f"{provider.value.upper()}_API_KEY"
        raise ModelConfigError(
            f"{provider.value} is configured but {env_var} is not set. "
            f"Add it to your .env (see .env.example), or point this role at "
            f"a different provider."
        )
    return secret.get_secret_value()


def build_chat_model(spec: str, settings: Settings | None = None, **overrides) -> BaseChatModel:
    """Instantiate a chat model from a ``provider:model`` spec.

    ``overrides`` are forwarded to the underlying LangChain class, so a caller that needs a
    one-off change (say ``temperature=0.7`` for query rewriting) can ask for it without
    touching global configuration.
    """
    settings = settings or get_settings()
    provider, model_name = parse_model_spec(spec)

    kwargs: dict = {
        "model": model_name,
        "model_provider": _RUNTIME_FOR_PROVIDER[provider],
        "temperature": settings.temperature,
        "timeout": settings.request_timeout_s,
    }

    if provider is Provider.LLAMA:
        kwargs["base_url"] = settings.ollama_base_url
    else:
        kwargs["api_key"] = _credential_for(provider, settings)
        kwargs["max_tokens"] = settings.max_tokens

    kwargs.update(overrides)
    return init_chat_model(**kwargs)


@lru_cache(maxsize=len(ModelRole))
def get_chat_model(role: ModelRole) -> BaseChatModel:
    """Return the configured model for a role.

    Cached per role: constructing a client sets up an HTTP connection pool, and rebuilding
    one on every graph node would add latency for no benefit. Use :func:`build_chat_model`
    directly when you need per-call overrides.
    """
    settings = get_settings()
    return build_chat_model(settings.model_for(role), settings)


def describe_configuration() -> dict[str, str]:
    """Role → spec mapping, for logging at startup and for the ``/health`` endpoint.

    Contains no secrets, so it is safe to expose.
    """
    settings = get_settings()
    return {role.value: settings.model_for(role) for role in ModelRole}
