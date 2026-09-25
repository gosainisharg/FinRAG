"""Vector retrieval over the Chroma collection."""

from __future__ import annotations

import logging

from finrag.retrieval.types import Filters, RetrievedChunk

logger = logging.getLogger(__name__)


def _similarity_from_distance(distance: float) -> float:
    """Convert Chroma's cosine distance to a similarity in [0, 1].

    The collection is built with `hnsw:space: cosine` over normalised vectors, so distance
    is `1 - cosine_similarity`. Reporting similarity keeps dense scores oriented the same
    way as sparse ones — higher is better — which matters when both are shown side by side.
    """
    return 1.0 - distance


def search(query: str, k: int, filters: Filters | None = None) -> list[RetrievedChunk]:
    """Top-k chunks by embedding similarity, pre-filtered by metadata."""
    from finrag.ingestion.pipeline import open_vector_store

    store = open_vector_store()
    where = filters.to_chroma_where() if filters is not None else None

    hits = store.similarity_search_with_score(query, k=k, filter=where)
    return [
        RetrievedChunk(
            chunk_id=document.metadata.get("chunk_id", document.id or ""),
            text=document.page_content,
            metadata=dict(document.metadata),
            score=_similarity_from_distance(distance),
            dense_score=_similarity_from_distance(distance),
            ranks={"dense": rank},
        )
        for rank, (document, distance) in enumerate(hits, start=1)
    ]
