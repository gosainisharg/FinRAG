"""Embedding model wrapper.

Implements LangChain's ``Embeddings`` interface properly rather than duck-typing it, so
the model works with ``langchain-chroma`` and anything else expecting the protocol.

Two details matter for retrieval quality, and both were missing from the original project:

- **The model is loaded once.** Constructing a SentenceTransformer reads ~1.3 GB from disk.
  The original rebuilt it on every call, which on a Streamlit rerun meant reloading per
  keystroke.
- **Queries and documents are embedded asymmetrically.** BGE models are trained with an
  instruction prefix on the query side only. Applying it to documents, or omitting it from
  queries, measurably degrades retrieval.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import torch
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from finrag.config import get_settings

logger = logging.getLogger(__name__)

# BGE's documented query prefix. Applied to queries only — the asymmetry is deliberate and
# is what the model was trained with.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def resolve_device() -> str:
    """Pick the fastest available backend.

    Apple Silicon's MPS backend is roughly an order of magnitude faster than CPU for this
    model, which is the difference between a minute and ten for a full re-index.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BGEEmbeddings(Embeddings):
    """bge-large-en-v1.5 behind LangChain's embedding protocol."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        batch_size: int = 32,
        query_instruction: str = BGE_QUERY_INSTRUCTION,
    ) -> None:
        self.model_name = model_name
        self.device = device or resolve_device()
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        logger.info("loading %s on %s", model_name, self.device)
        self._model = SentenceTransformer(model_name, device=self.device)

    @property
    def dimension(self) -> int:
        # Renamed in sentence-transformers 5.x; keep the old name as a fallback so the
        # pinned range stays installable either side of the change.
        getter = (
            getattr(self._model, "get_embedding_dimension", None)
            or self._model.get_sentence_embedding_dimension
        )
        return int(getter())

    def _encode(self, texts: list[str], show_progress: bool = False) -> list[list[float]]:
        # Normalised vectors make Chroma's cosine distance a plain dot product, and keep
        # dense scores on a comparable scale to the reranker's later.
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages. No instruction prefix — that is the query side only."""
        if not texts:
            return []
        return self._encode(texts, show_progress=len(texts) > 200)

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query, with the instruction prefix the model expects."""
        return self._encode([f"{self.query_instruction}{text}"])[0]


@lru_cache(maxsize=1)
def get_embeddings() -> BGEEmbeddings:
    """Process-wide embedding model.

    Cached because loading is expensive and the model is stateless once built.
    """
    settings = get_settings()
    return BGEEmbeddings(settings.embedding_model)
