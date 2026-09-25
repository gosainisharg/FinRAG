"""Cross-encoder reranking.

Bi-encoders embed query and document separately, so a chunk's vector is fixed before the
query exists. A cross-encoder reads both together and can weigh their interaction, which
is why it reorders well but cannot be used to search — scoring the whole corpus per query
is infeasible. It runs last, over the handful of candidates the earlier stages produced.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from sentence_transformers import CrossEncoder

from finrag.config import get_settings
from finrag.ingestion.embeddings import resolve_device
from finrag.retrieval.types import RetrievedChunk

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def get_reranker(model_name: str | None = None) -> CrossEncoder:
    """Process-wide cross-encoder.

    Cached per model name so the eval harness can compare two rerankers in one run without
    reloading either. The original constructed a CrossEncoder inside the ranking function,
    reloading the model on every query.
    """
    name = model_name or get_settings().reranker_model
    logger.info("loading reranker %s", name)
    return CrossEncoder(name, device=resolve_device())


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_n: int,
    model_name: str | None = None,
) -> list[RetrievedChunk]:
    """Reorder candidates by cross-encoder relevance and keep the best `top_n`."""
    if not chunks:
        return []

    model = get_reranker(model_name)
    scores = model.predict([(query, chunk.text) for chunk in chunks])

    scored = sorted(zip(chunks, scores, strict=True), key=lambda pair: -float(pair[1]))
    return [
        RetrievedChunk(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            metadata=chunk.metadata,
            score=float(score),
            dense_score=chunk.dense_score,
            sparse_score=chunk.sparse_score,
            rerank_score=float(score),
            ranks={**chunk.ranks, "rerank": rank},
        )
        for rank, (chunk, score) in enumerate(scored[:top_n], start=1)
    ]
