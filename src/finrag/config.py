"""Application configuration.

Every tunable in FinRAG lives here and is overridable by environment variable, so the same
container image runs in dev, CI and production without a code change. Secrets are never
defaulted and never logged.

Settings are read from (in ascending precedence): field defaults, the `.env` file at the
project root, then real environment variables.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Provider(StrEnum):
    """The three swappable chat backends.

    ``LLAMA`` names the open-weight *model*. Ollama is merely the runtime that serves it
    locally, and is an implementation detail handled in ``finrag.llm`` — it is deliberately
    not a provider name here.
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LLAMA = "llama"


class ModelRole(StrEnum):
    """Distinct jobs a model does in the pipeline.

    Each role is configured independently because the requirements genuinely differ: routing
    and grading are high-volume, latency-sensitive classification calls that a small model
    handles well, while generation wants the strongest model available. ``JUDGE`` should be
    set to a *different model family* than ``GENERATOR`` — an LLM scoring its own output
    rates it more highly than a neutral judge does.
    """

    ROUTER = "router"
    GRADER = "grader"
    REWRITER = "rewriter"
    GENERATOR = "generator"
    JUDGE = "judge"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="FINRAG_",
        extra="ignore",
    )

    # ------------------------------------------------------------------ secrets
    # Read from the conventional unprefixed names, because the provider SDKs look for
    # exactly these and we want a single source of truth.
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    langsmith_api_key: SecretStr | None = Field(default=None, validation_alias="LANGSMITH_API_KEY")

    # ------------------------------------------------------------------ models
    # Format is "provider:model". Overridable per role, e.g. FINRAG_GENERATOR_MODEL=...
    router_model: str = "openai:gpt-5.4-nano"
    grader_model: str = "openai:gpt-5.4-nano"
    rewriter_model: str = "openai:gpt-5.4-nano"
    generator_model: str = "anthropic:claude-sonnet-5"
    judge_model: str = "openai:gpt-5.4"

    temperature: float = 0.0
    max_tokens: int = 2048
    request_timeout_s: float = 60.0

    # Where the Ollama runtime is listening, for the LLaMA backend.
    ollama_base_url: str = "http://localhost:11434"
    # Default locally-served model. Used by scripts/check_setup.py and as the zero-cost
    # target when developing against LLaMA instead of a hosted provider.
    local_model: str = "llama:llama3.1:8b"
    # Point every role at `local_model`, ignoring the per-role settings. One switch for
    # running the entire pipeline free of charge and offline.
    local_only: bool = False

    # ------------------------------------------------------------------ retrieval
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    # Both rerankers are sanctioned by the spec. Measured on three questions over this
    # corpus, ms-marco returned an identical top-1 to bge-reranker-v2-m3 at ~395ms mean
    # versus ~5,100ms — 13x faster for no visible loss, and 5s/query is not viable behind
    # a streaming endpoint. Phase 3 settles it on the golden set; swap via this setting.
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    alternate_reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # Candidates pulled from each retriever before fusion.
    retrieval_k: int = 20
    # Documents surviving the cross-encoder, i.e. what reaches the prompt.
    rerank_top_n: int = 5
    # Weight of the dense arm in reciprocal rank fusion; 0.5 is an even blend.
    hybrid_dense_weight: float = 0.5
    # Chunks fetched per company for a comparison question. Measured over four
    # cross-company questions: 2 each found both figures 1/4 of the time, 4 each 2/4,
    # 8 each 3/4. A comparison needs more context than a single-company question, not
    # the same budget split between them.
    comparison_chunks_per_company: int = 8

    # ------------------------------------------------------------------ prompts
    # Prompts are versioned artifacts on disk. Bumping this swaps every prompt at once;
    # prompt_versions overrides individual ones, which is how the eval compares a single
    # prompt change without moving anything else.
    prompt_version: str = "v4"
    prompt_versions: dict[str, str] = Field(default_factory=dict)

    # ------------------------------------------------------------------ graph
    # Hard ceiling on retrieve→grade→rewrite cycles. Bounds both latency and spend, and
    # guarantees termination if the grader is never satisfied.
    max_retrieval_attempts: int = 2

    # ------------------------------------------------------------------ cache
    cache_enabled: bool = True
    # Cosine similarity above which an incoming query is treated as a cache hit. High by
    # design: a false hit returns a confidently wrong answer, which is worse than a miss.
    cache_similarity_threshold: float = 0.95
    cache_max_entries: int = 1000

    # ------------------------------------------------------------------ observability
    langsmith_tracing: bool = Field(default=False, validation_alias="LANGSMITH_TRACING")
    langsmith_project: str = "finrag"
    phoenix_enabled: bool = False
    phoenix_endpoint: str = "http://localhost:6006/v1/traces"

    # ------------------------------------------------------------------ paths
    data_dir: Path = PROJECT_ROOT / "data"
    chroma_dir: Path = PROJECT_ROOT / "chroma"
    bm25_path: Path = PROJECT_ROOT / "chroma" / "bm25.pkl"
    collection_name: str = "finrag_filings"

    # ------------------------------------------------------------------ serving
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @field_validator("hybrid_dense_weight")
    @classmethod
    def _weight_in_unit_interval(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("hybrid_dense_weight must be between 0.0 and 1.0")
        return v

    @field_validator("cache_similarity_threshold")
    @classmethod
    def _threshold_in_unit_interval(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("cache_similarity_threshold must be between 0.0 and 1.0")
        return v

    @model_validator(mode="after")
    def _apply_local_only(self) -> Settings:
        """Collapse every role onto the local model when ``local_only`` is set.

        Applied after field validation so it overrides anything the environment supplied,
        which is the point: one switch, no chance a stray per-role variable still bills a
        hosted provider.
        """
        if self.local_only:
            for role in ModelRole:
                setattr(self, f"{role.value}_model", self.local_model)
        return self

    def model_for(self, role: ModelRole) -> str:
        """Return the configured ``provider:model`` spec for a role."""
        return getattr(self, f"{role.value}_model")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton.

    Cached so that importing config in a hot path (or in every FastAPI request) does not
    re-read and re-validate the environment. Call ``get_settings.cache_clear()`` in tests
    that need to vary the environment.
    """
    return Settings()
