"""Shared retrieval vocabulary.

One result type across every strategy, so callers — and the eval harness — never branch on
which retriever produced a hit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Strategy(StrEnum):
    """How a query is turned into candidates.

    ``HYBRID_RERANK`` is the intended default. The others exist because the eval harness
    needs to show what each stage contributes — a reranker that does not beat plain hybrid
    on the golden set is latency you are paying for nothing.
    """

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    DENSE_RERANK = "dense_rerank"
    HYBRID_RERANK = "hybrid_rerank"


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved chunk with the scores that produced it.

    Every stage's score is kept rather than overwritten. When an answer is wrong, the
    question is usually *which* stage ranked the bad chunk highly, and collapsing to a
    single number throws that away.
    """

    chunk_id: str
    text: str
    metadata: dict
    score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None
    ranks: dict[str, int] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        """Short human-readable provenance, for prompts and for debugging."""
        ticker = self.metadata.get("ticker", "?")
        year = self.metadata.get("fiscal_year", "?")
        section = self.metadata.get("section", "?")
        page = self.metadata.get("page", "?")
        return f"{ticker} FY{year} {section} p{page}"


@dataclass(frozen=True)
class Filters:
    """Metadata restrictions applied before scoring.

    Pre-filtering rather than filtering results: a question about Tesla should search
    Tesla's chunks, not search everything and discard. Post-filtering silently returns
    fewer than ``k`` and lets an irrelevant filing crowd out the right one.
    """

    tickers: tuple[str, ...] = ()
    fiscal_years: tuple[int, ...] = ()
    sections: tuple[str, ...] = ()
    form_types: tuple[str, ...] = ()
    exclude_exhibits: bool = True

    def is_empty(self) -> bool:
        return not (self.tickers or self.fiscal_years or self.sections or self.form_types)

    def matches(self, metadata: dict) -> bool:
        """Whether a chunk satisfies these filters. Used by the sparse index."""
        if self.exclude_exhibits and metadata.get("section") == "Exhibits":
            return False
        if self.tickers and metadata.get("ticker") not in self.tickers:
            return False
        if self.fiscal_years and metadata.get("fiscal_year") not in self.fiscal_years:
            return False
        if self.sections and metadata.get("section") not in self.sections:
            return False
        return not (self.form_types and metadata.get("form_type") not in self.form_types)

    def to_chroma_where(self) -> dict | None:
        """Translate to Chroma's `where` syntax.

        Chroma requires an explicit ``$and`` for more than one condition, and ``$in`` for
        multi-valued ones; a bare dict of several keys is silently wrong.
        """
        clauses: list[dict] = []
        if self.exclude_exhibits:
            clauses.append({"section": {"$ne": "Exhibits"}})
        if self.tickers:
            clauses.append({"ticker": {"$in": list(self.tickers)}})
        if self.fiscal_years:
            clauses.append({"fiscal_year": {"$in": list(self.fiscal_years)}})
        if self.sections:
            clauses.append({"section": {"$in": list(self.sections)}})
        if self.form_types:
            clauses.append({"form_type": {"$in": list(self.form_types)}})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}
