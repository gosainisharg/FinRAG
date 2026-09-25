"""Embedding and index-building. Local models only; no network, no LLM."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from finrag.config import get_settings
from finrag.ingestion.embeddings import BGE_QUERY_INSTRUCTION, get_embeddings, resolve_device
from finrag.ingestion.pipeline import build_index, ingest_path

DATA = Path(__file__).resolve().parents[1] / "data"
# The smallest filing, so the write-path tests stay quick.
SMALL = DATA / "Meta-Reports-First-Quarter-2024-Results-2024.pdf"


class TestEmbeddings:
    @pytest.fixture(scope="class")
    def embeddings(self):
        return get_embeddings()

    def test_model_is_loaded_once(self):
        # The original rebuilt the model per call, reloading 1.3GB on every Streamlit rerun.
        assert get_embeddings() is get_embeddings()

    def test_dimension_matches_bge_large(self, embeddings):
        assert embeddings.dimension == 1024

    def test_vectors_are_normalised(self, embeddings):
        # Unit vectors make cosine distance a dot product and keep scores comparable.
        vectors = np.array(embeddings.embed_documents(["revenue grew", "risk factors"]))
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    def test_queries_and_documents_are_embedded_asymmetrically(self, embeddings):
        """BGE is trained with an instruction on the query side only."""
        text = "What were total revenues?"
        as_query = np.array(embeddings.embed_query(text))
        as_document = np.array(embeddings.embed_documents([text])[0])
        assert not np.allclose(as_query, as_document)

    def test_query_instruction_is_the_documented_prefix(self, embeddings):
        assert embeddings.query_instruction == BGE_QUERY_INSTRUCTION

    def test_ranks_the_relevant_passage_higher(self, embeddings):
        docs = np.array(
            embeddings.embed_documents(
                [
                    "Total revenues were 96,773 million in 2023.",
                    "The board approved a new equity incentive plan.",
                ]
            )
        )
        query = np.array(embeddings.embed_query("What were total revenues in 2023?"))
        scores = docs @ query
        assert scores[0] > scores[1]

    def test_empty_input_returns_empty(self, embeddings):
        assert embeddings.embed_documents([]) == []

    def test_device_is_resolvable(self):
        assert resolve_device() in {"cuda", "mps", "cpu"}


class TestBuildIndex:
    def test_dry_run_writes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(get_settings(), "chroma_dir", tmp_path / "chroma")
        report = build_index(DATA, dry_run=True)
        assert report.total_chunks > 1000
        assert not (tmp_path / "chroma").exists()

    def test_reports_every_source(self):
        report = build_index(DATA, dry_run=True)
        assert len(report.sources) == 6
        assert {s.ticker for s in report.sources} == {"AAPL", "TSLA", "COST", "META"}

    def test_tables_are_counted_separately(self):
        report = build_index(DATA, dry_run=True)
        tesla = next(s for s in report.sources if s.ticker == "TSLA")
        assert tesla.tables > 40

    def test_reingest_replaces_rather_than_accumulates(self, tmp_path, monkeypatch):
        """An edited document changes chunk ids, so upserting would leave the superseded
        chunks behind and still retrievable. This is the bug the original had."""
        settings = get_settings()
        monkeypatch.setattr(settings, "chroma_dir", tmp_path / "chroma")
        monkeypatch.setattr(settings, "collection_name", "test_reingest")

        from finrag.ingestion.pipeline import open_vector_store

        store = open_vector_store()
        first, chunks = ingest_path(store, SMALL)
        assert first.deleted == 0
        count_after_first = store._collection.count()
        assert count_after_first == len(chunks)

        second, _ = ingest_path(store, SMALL)
        assert second.deleted == count_after_first
        assert store._collection.count() == count_after_first

    def test_written_chunks_carry_filter_metadata(self, tmp_path, monkeypatch):
        settings = get_settings()
        monkeypatch.setattr(settings, "chroma_dir", tmp_path / "chroma")
        monkeypatch.setattr(settings, "collection_name", "test_metadata")

        from finrag.ingestion.pipeline import open_vector_store

        store = open_vector_store()
        ingest_path(store, SMALL)

        stored = store.get(where={"ticker": "META"}, include=["metadatas"])
        assert stored["metadatas"]
        for metadata in stored["metadatas"]:
            assert metadata["form_type"] == "earnings-release"
            assert metadata["fiscal_year"] == 2024
            assert "page" in metadata

    def test_missing_data_dir_fails_loudly(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_index(tmp_path, dry_run=True)
