"""Keyword retrieval over the chunk corpus.

This is the half of "hybrid search" the original project claimed but never had — its
``hybrid_search`` ran two *dense* searches over the same embedding space. Sparse retrieval
earns its place on exact tokens that embeddings blur: a specific figure (``96,773``), a
form name (``10-K``), a defined term (``non-GAAP``), a ticker.

Chroma has no sparse index, so this is a separate ``rank_bm25`` index built from the same
chunks and persisted alongside the vector store. That is fine for a corpus of this size and
does not scale; the production answer is Elasticsearch, or a store with native hybrid
support such as Qdrant. Keeping the limitation explicit is the point.
"""

from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

from finrag.config import get_settings
from finrag.retrieval.types import Filters, RetrievedChunk

logger = logging.getLogger(__name__)

# Tokens are alphanumeric runs that may contain internal separators, so financial notation
# survives: "96,773" stays one token rather than becoming "96" and "773", and "10-k",
# "non-gaap" and "3.14" stay intact. Splitting these is the difference between a query for
# an exact figure hitting and missing.
_TOKEN = re.compile(r"[a-z0-9]+(?:[.,'’-][a-z0-9]+)*")
_POSSESSIVE = re.compile(r"['’]s$")
_GROUPED_NUMBER = re.compile(r"^\d{1,3}(?:,\d{3})+$")


def tokenize(text: str) -> list[str]:
    """Lowercase and split, preserving financial notation.

    Grouped numbers also emit an unpunctuated variant, so a query for ``96773`` matches a
    document containing ``96,773``. Possessives are normalised so "Company's" is findable
    by "company".
    """
    tokens: list[str] = []
    for raw in _TOKEN.findall(text.lower()):
        token = _POSSESSIVE.sub("", raw)
        if not token:
            continue
        tokens.append(token)
        if _GROUPED_NUMBER.match(token):
            tokens.append(token.replace(",", ""))
    return tokens


@dataclass
class BM25Index:
    """An in-memory BM25 index over the chunk corpus."""

    chunk_ids: list[str]
    texts: list[str]
    metadatas: list[dict]
    bm25: BM25Okapi

    def __len__(self) -> int:
        return len(self.chunk_ids)

    @classmethod
    def build(cls, chunk_ids: list[str], texts: list[str], metadatas: list[dict]) -> BM25Index:
        logger.info("building BM25 index over %d chunks", len(texts))
        corpus = [tokenize(text) for text in texts]
        return cls(chunk_ids, texts, metadatas, BM25Okapi(corpus))

    def search(self, query: str, k: int, filters: Filters | None = None) -> list[RetrievedChunk]:
        """Top-k by BM25 score, restricted to chunks satisfying the filters."""
        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        allowed = range(len(self.chunk_ids))
        if filters is not None:
            allowed = [i for i in allowed if filters.matches(self.metadatas[i])]

        # A zero score means no query term appeared; returning those pads the result with
        # arbitrary chunks that fusion would then treat as evidence.
        ranked = sorted((i for i in allowed if scores[i] > 0), key=lambda i: -scores[i])[:k]

        return [
            RetrievedChunk(
                chunk_id=self.chunk_ids[i],
                text=self.texts[i],
                metadata=self.metadatas[i],
                score=float(scores[i]),
                sparse_score=float(scores[i]),
                ranks={"sparse": rank},
            )
            for rank, i in enumerate(ranked, start=1)
        ]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("wrote BM25 index to %s", path)

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        with path.open("rb") as handle:
            return pickle.load(handle)


def build_from_vector_store(persist: bool = True) -> BM25Index:
    """Build the sparse index from the contents of the vector store.

    Sourcing both indexes from the same place is what keeps them consistent: a chunk can
    never be dense-retrievable but sparse-invisible, which would make hybrid results depend
    on which index happened to be rebuilt last.
    """
    from finrag.ingestion.pipeline import open_vector_store

    store = open_vector_store()
    data = store.get(include=["documents", "metadatas"])
    index = BM25Index.build(data["ids"], data["documents"], list(data["metadatas"]))

    if persist:
        index.save(get_settings().bm25_path)
    return index


@lru_cache(maxsize=1)
def get_bm25_index() -> BM25Index:
    """Process-wide sparse index, loaded from disk or built on first use."""
    path = get_settings().bm25_path
    if path.exists():
        logger.info("loading BM25 index from %s", path)
        return BM25Index.load(path)
    return build_from_vector_store()
