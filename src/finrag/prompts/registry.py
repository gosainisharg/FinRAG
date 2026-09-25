"""Versioned prompt artifacts.

Prompts are files on disk, loaded by name and version::

    prompt = get_prompt("grade")            # the configured default version
    prompt = get_prompt("grade", "v2")      # a specific one
    text = prompt.render(question=q, context=c)

Two reasons this is not just string constants in the node modules.

**Prompts are the dominant variable.** Measured in Phase 0 on this corpus: an engineered
grader prompt took `mistral:7b` from 6/15 to 15/15 on the same task, and the naive prompt
made the *larger* model score worse than the smaller one. A change of that size deserves a
version, a diff and an eval run — the same treatment as a code change.

**Git is the source of truth, not a hosted registry.** A prompt pulled from a remote
service at runtime makes CI non-deterministic and the build dependent on someone else's
uptime. LangSmith's prompt hub is a useful mirror for browsing and comparison; it is not
where the bytes come from.

Every rendered prompt reports a checksum, so an eval report can record exactly which text
produced a number rather than just a version label someone may have edited in place.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent
# Version directories may hold documentation alongside the templates.
_NOT_A_PROMPT = {"README.md"}
_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)
_PLACEHOLDER = re.compile(r"(?<!\{)\{([a-z_][a-z0-9_]*)\}(?!\})")


class PromptError(RuntimeError):
    """Raised for a missing prompt, a malformed file, or a bad render."""


@dataclass(frozen=True)
class Prompt:
    """One prompt template plus the metadata needed to cite it."""

    name: str
    version: str
    template: str
    description: str = ""
    variables: tuple[str, ...] = field(default_factory=tuple)

    @property
    def checksum(self) -> str:
        """Short digest of the template text.

        A version label can be edited in place; a checksum cannot. Eval reports record
        this so a number is traceable to the exact bytes that produced it.
        """
        return hashlib.sha256(self.template.encode()).hexdigest()[:12]

    @property
    def reference(self) -> str:
        return f"{self.name}@{self.version}#{self.checksum}"

    def render(self, **values: object) -> str:
        """Fill the template, failing loudly on a missing or unexpected variable.

        A silently unfilled placeholder reaches the model as a literal `{context}` and
        produces a plausible-looking answer about nothing, which is far harder to notice
        than an exception.
        """
        missing = set(self.variables) - set(values)
        if missing:
            raise PromptError(f"{self.reference} needs {sorted(missing)}")

        unexpected = set(values) - set(self.variables)
        if unexpected:
            raise PromptError(f"{self.reference} does not take {sorted(unexpected)}")

        try:
            return self.template.format(**values)
        except (KeyError, IndexError) as exc:
            raise PromptError(f"{self.reference} failed to render: {exc}") from exc


def _parse(path: Path) -> Prompt:
    raw = path.read_text()
    match = _FRONTMATTER.match(raw)
    if not match:
        raise PromptError(f"{path} is missing its --- frontmatter block")

    header, body = match.groups()
    meta: dict[str, str] = {}
    for line in header.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()

    body = body.strip()
    declared = tuple(v.strip() for v in meta.get("variables", "").split(",") if v.strip())
    found = tuple(sorted(set(_PLACEHOLDER.findall(body))))

    # Drift between the declared contract and the actual placeholders is how a node ends
    # up passing an argument nothing consumes, or omitting one the template needs.
    if set(declared) != set(found):
        raise PromptError(
            f"{path}: declared variables {sorted(declared)} do not match "
            f"the placeholders in the body {sorted(found)}"
        )

    return Prompt(
        name=meta.get("name", path.parent.name),
        version=meta.get("version", path.parent.name),
        template=body,
        description=meta.get("description", ""),
        variables=declared,
    )


@lru_cache(maxsize=64)
def get_prompt(name: str, version: str | None = None) -> Prompt:
    """Load a prompt by name, defaulting to the configured version."""
    from finrag.config import get_settings

    settings = get_settings()
    resolved = version or settings.prompt_versions.get(name) or settings.prompt_version

    path = PROMPTS_DIR / resolved / f"{name}.md"
    if not path.exists():
        available = ", ".join(sorted(p.name for p in PROMPTS_DIR.glob("v*"))) or "none"
        raise PromptError(
            f"no prompt {name!r} at version {resolved!r} ({path}). Versions: {available}"
        )
    return _parse(path)


def list_versions(name: str) -> list[str]:
    """Every version available for a prompt, oldest label first."""
    return sorted(
        directory.name
        for directory in PROMPTS_DIR.iterdir()
        if directory.is_dir() and (directory / f"{name}.md").exists()
    )


def describe_prompts() -> dict[str, str]:
    """Name → reference for every prompt in use. For startup logs and eval reports."""
    names = {path.stem for path in PROMPTS_DIR.glob("v*/*.md") if path.name not in _NOT_A_PROMPT}
    return {name: get_prompt(name).reference for name in sorted(names)}
