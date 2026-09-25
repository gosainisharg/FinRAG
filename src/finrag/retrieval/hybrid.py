"""Fusing dense and sparse results.

Reciprocal rank fusion rather than score blending. Cosine similarity and BM25 live on
different, unbounded, corpus-dependent scales; normalising them against each other
requires a calibration that would have to be re-derived whenever the corpus changes. RRF
uses only rank position, so it needs no calibration and cannot be skewed by one retriever
returning unusually large scores.
"""

from __future__ import annotations

from finrag.retrieval.types import RetrievedChunk

# Standard RRF damping. Large enough that the top few ranks are not overwhelmingly
# dominant, so a chunk ranked highly by one retriever and moderately by the other can beat
# one ranked first by a single retriever — which is the point of fusing at all.
RRF_K = 60


def reciprocal_rank_fusion(
    dense: list[RetrievedChunk],
    sparse: list[RetrievedChunk],
    k: int,
    dense_weight: float = 0.5,
) -> list[RetrievedChunk]:
    """Merge two ranked lists into one.

    `dense_weight` splits the contribution between the two arms; 0.5 is an even blend,
    1.0 ignores sparse entirely.
    """
    sparse_weight = 1.0 - dense_weight
    merged: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}

    for results, weight in ((dense, dense_weight), (sparse, sparse_weight)):
        for rank, chunk in enumerate(results, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + weight / (RRF_K + rank)
            existing = merged.get(chunk.chunk_id)
            if existing is None:
                merged[chunk.chunk_id] = chunk
            else:
                # Seen by both arms: keep every stage's score and rank, so a result can be
                # explained after the fact rather than reduced to one opaque number.
                merged[chunk.chunk_id] = RetrievedChunk(
                    chunk_id=existing.chunk_id,
                    text=existing.text,
                    metadata=existing.metadata,
                    score=existing.score,
                    dense_score=existing.dense_score or chunk.dense_score,
                    sparse_score=existing.sparse_score or chunk.sparse_score,
                    rerank_score=existing.rerank_score or chunk.rerank_score,
                    ranks={**existing.ranks, **chunk.ranks},
                )

    ordered = sorted(merged.values(), key=lambda c: -scores[c.chunk_id])[:k]
    return [
        RetrievedChunk(
            chunk_id=c.chunk_id,
            text=c.text,
            metadata=c.metadata,
            score=scores[c.chunk_id],
            dense_score=c.dense_score,
            sparse_score=c.sparse_score,
            rerank_score=c.rerank_score,
            ranks={**c.ranks, "hybrid": rank},
        )
        for rank, c in enumerate(ordered, start=1)
    ]
