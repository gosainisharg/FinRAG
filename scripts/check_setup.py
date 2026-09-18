#!/usr/bin/env python
"""Verify the Phase 0 setup: configuration loads, and each backend actually answers.

    python scripts/check_setup.py            # config summary + reachability, no API calls
    python scripts/check_setup.py --live     # also send one tiny prompt per provider

``--live`` costs a fraction of a cent per hosted provider and nothing for LLaMA.
"""

from __future__ import annotations

import argparse
import sys

from finrag.config import ModelRole, Provider, get_settings
from finrag.llm import ModelConfigError, build_chat_model, describe_configuration, parse_model_spec

OK, FAIL, SKIP = "\033[32m✓\033[0m", "\033[31m✗\033[0m", "\033[33m-\033[0m"
PROBE = "Reply with exactly one word: ok"


def show_configuration() -> set[Provider]:
    """Print the role → model mapping and return the distinct providers in use."""
    settings = get_settings()
    print("\nRole → model")
    print("─" * 52)

    providers: set[Provider] = set()
    for role, spec in describe_configuration().items():
        try:
            provider, model = parse_model_spec(spec)
        except ModelConfigError as exc:
            print(f"  {FAIL} {role:<10} {spec}\n      {exc}")
            continue
        providers.add(provider)
        print(f"  {OK} {role:<10} {provider.value:<10} {model}")

    print("\nCredentials")
    print("─" * 52)
    for provider, secret in (
        (Provider.OPENAI, settings.openai_api_key),
        (Provider.ANTHROPIC, settings.anthropic_api_key),
    ):
        in_use = " (in use)" if provider in providers else ""
        print(
            f"  {OK if secret else SKIP} {provider.value:<10} "
            f"{'set' if secret else 'not set'}{in_use}"
        )
    print(f"  {SKIP} {'llama':<10} no credential needed — served locally by Ollama")

    print(f"\n  tracing: langsmith={settings.langsmith_tracing} phoenix={settings.phoenix_enabled}")
    print(f"  chroma:  {settings.chroma_dir}")
    return providers


def probe(providers: set[Provider]) -> bool:
    """Send one minimal prompt per provider. Returns True if all attempted probes passed."""
    print("\nLive probe")
    print("─" * 52)

    settings = get_settings()
    specs = {
        Provider.OPENAI: settings.model_for(ModelRole.ROUTER),
        Provider.ANTHROPIC: settings.model_for(ModelRole.GENERATOR),
        Provider.LLAMA: "llama:llama3.2",
    }

    all_passed = True
    for provider in Provider:
        spec = specs[provider]
        # Only probe a provider that some role is actually configured to use, unless it is
        # LLaMA — that one is free, so a reachability check is always worth reporting.
        if provider not in providers and provider is not Provider.LLAMA:
            print(f"  {SKIP} {provider.value:<10} not used by any role")
            continue
        try:
            reply = build_chat_model(spec).invoke(PROBE).content
            print(f"  {OK} {provider.value:<10} {spec} → {str(reply).strip()[:40]!r}")
        except ModelConfigError as exc:
            print(f"  {SKIP} {provider.value:<10} {exc}")
        except Exception as exc:  # noqa: BLE001 - surface any backend failure verbatim
            hint = ""
            if provider is Provider.LLAMA:
                hint = "\n      is Ollama running? `ollama serve` then `ollama pull llama3.2`"
            print(
                f"  {FAIL} {provider.value:<10} {spec}\n      "
                f"{type(exc).__name__}: {str(exc)[:160]}{hint}"
            )
            if provider in providers:
                all_passed = False
    return all_passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="send one real prompt per provider")
    args = parser.parse_args()

    providers = show_configuration()
    ok = probe(providers) if args.live else True

    if not args.live:
        print("\nConfiguration only. Re-run with --live to send a real prompt.\n")
    else:
        print(f"\n{'All configured providers responded.' if ok else 'Some providers failed.'}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
