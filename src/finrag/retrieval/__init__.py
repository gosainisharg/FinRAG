"""Retrieval, behind one function.

    from finrag.retrieval import retrieve, Strategy, Filters

    chunks = retrieve("What were Tesla's 2023 revenues?", Strategy.HYBRID_RERANK)

Everything downstream — the graph nodes, the API, the eval harness — calls this and
nothing else. Strategy selection is an argument rather than a branch at the call site, so
the router can change strategy at runtime and the eval can sweep all of them without
touching any other code.

No LLM calls happen here. That keeps retrieval independently testable and independently
measurable, which is what lets Phase 3 report a retrieval baseline before any agentic
machinery exists to muddy it.
"""

from __future__ import annotations

import logging

from finrag.config import get_settings
from finrag.retrieval import dense, sparse
from finrag.retrieval import rerank as rerank_module
from finrag.retrieval.hybrid import reciprocal_rank_fusion
from finrag.retrieval.types import Filters, RetrievedChunk, Strategy

logger = logging.getLogger(__name__)

__all__ = ["Filters", "RetrievedChunk", "Strategy", "retrieve"]

# Strategies that rerank fetch a wider candidate pool first: the cross-encoder can only
# reorder what it is given, so handing it exactly `k` candidates wastes it entirely.
RERANK_CANDIDATE_MULTIPLIER = 4


def retrieve(
    query: str,
    strategy: Strategy = Strategy.HYBRID_RERANK,
    filters: Filters | None = None,
    k: int | None = None,
    *,
    reranker_model: str | None = None,
) -> list[RetrievedChunk]:
    """Retrieve the `k` most relevant chunks for a query.

    Args:
        query: natural-language question.
        strategy: which retrieval path to use.
        filters: metadata restrictions, applied before scoring.
        k: how many chunks to return; defaults to the configured rerank_top_n.
        reranker_model: override the cross-encoder, for comparing rerankers in eval.
    """
    settings = get_settings()
    filters = filters if filters is not None else Filters()
    k = k or settings.rerank_top_n

    reranking = strategy in (Strategy.DENSE_RERANK, Strategy.HYBRID_RERANK)
    # Fetch wider when a reranker will narrow it again.
    candidate_k = max(k * RERANK_CANDIDATE_MULTIPLIER, settings.retrieval_k) if reranking else k

    if strategy is Strategy.DENSE:
        results = dense.search(query, candidate_k, filters)
    elif strategy is Strategy.SPARSE:
        results = sparse.get_bm25_index().search(query, candidate_k, filters)
    elif strategy is Strategy.DENSE_RERANK:
        results = dense.search(query, candidate_k, filters)
    else:
        results = reciprocal_rank_fusion(
            dense.search(query, candidate_k, filters),
            sparse.get_bm25_index().search(query, candidate_k, filters),
            k=candidate_k,
            dense_weight=settings.hybrid_dense_weight,
        )

    if reranking:
        results = rerank_module.rerank(query, results, top_n=k, model_name=reranker_model)

    return results[:k]
