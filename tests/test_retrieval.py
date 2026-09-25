"""Retrieval layer. Pure-function tests run anywhere; integration tests need a built index."""

from __future__ import annotations

import pytest

from finrag.config import get_settings
from finrag.retrieval import Filters, Strategy, retrieve
from finrag.retrieval.hybrid import RRF_K, reciprocal_rank_fusion
from finrag.retrieval.sparse import BM25Index, tokenize
from finrag.retrieval.types import RetrievedChunk


def chunk(chunk_id: str, **kwargs) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=kwargs.pop("text", f"text of {chunk_id}"),
        metadata=kwargs.pop("metadata", {}),
        score=kwargs.pop("score", 0.0),
        **kwargs,
    )


class TestTokenize:
    def test_keeps_grouped_numbers_intact(self):
        """Splitting 96,773 into 96 and 773 is the difference between hitting and missing."""
        assert "96,773" in tokenize("Total revenues were $96,773 million")

    def test_also_emits_an_unpunctuated_number_variant(self):
        # So a query for 96773 matches a document containing 96,773.
        assert "96773" in tokenize("revenues of 96,773")

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Form 10-K", "10-k"),
            ("non-GAAP measures", "non-gaap"),
            ("EPS of 3.12", "3.12"),
        ],
    )
    def test_preserves_financial_notation(self, text, expected):
        assert expected in tokenize(text)

    def test_normalises_possessives(self):
        # "Company's" must be findable by "company".
        assert "company" in tokenize("The Company's results")
        assert "company's" not in tokenize("The Company's results")

    def test_strips_currency_and_percent_symbols(self):
        tokens = tokenize("$1,200 up 19%")
        assert "1,200" in tokens
        assert "19" in tokens
        assert "$" not in tokens

    def test_empty_text(self):
        assert tokenize("") == []


class TestFilters:
    def test_empty_filters_still_exclude_exhibits_by_default(self):
        where = Filters().to_chroma_where()
        assert where == {"section": {"$ne": "Exhibits"}}

    def test_multiple_conditions_use_explicit_and(self):
        """Chroma silently misreads a bare multi-key dict; it requires $and."""
        where = Filters(tickers=("TSLA",), fiscal_years=(2023,)).to_chroma_where()
        assert "$and" in where
        assert {"ticker": {"$in": ["TSLA"]}} in where["$and"]

    def test_single_condition_is_not_wrapped(self):
        where = Filters(tickers=("TSLA",), exclude_exhibits=False).to_chroma_where()
        assert where == {"ticker": {"$in": ["TSLA"]}}

    def test_no_filters_at_all_is_none(self):
        assert Filters(exclude_exhibits=False).to_chroma_where() is None

    @pytest.mark.parametrize(
        "metadata, expected",
        [
            ({"ticker": "TSLA", "section": "Item 8"}, True),
            ({"ticker": "AAPL", "section": "Item 8"}, False),
            ({"ticker": "TSLA", "section": "Exhibits"}, False),
        ],
    )
    def test_matches_mirrors_the_chroma_filter(self, metadata, expected):
        # The sparse index filters in Python, so the two paths must agree.
        assert Filters(tickers=("TSLA",)).matches(metadata) is expected


class TestReciprocalRankFusion:
    def test_agreement_beats_a_single_first_place(self):
        """The point of fusing: consistent across both arms outranks first in one."""
        dense = [chunk("solo"), chunk("both")]
        sparse = [chunk("other"), chunk("both")]
        fused = reciprocal_rank_fusion(dense, sparse, k=3, dense_weight=0.5)
        assert fused[0].chunk_id == "both"

    def test_scores_follow_the_rrf_formula(self):
        fused = reciprocal_rank_fusion([chunk("a")], [], k=1, dense_weight=1.0)
        assert fused[0].score == pytest.approx(1.0 / (RRF_K + 1))

    def test_dense_weight_of_one_ignores_sparse(self):
        fused = reciprocal_rank_fusion([chunk("d")], [chunk("s")], k=2, dense_weight=1.0)
        assert fused[0].chunk_id == "d"
        assert fused[1].score == 0.0

    def test_preserves_every_stage_score(self):
        """Collapsing to one number loses which stage ranked a bad chunk highly."""
        dense = [chunk("x", dense_score=0.9)]
        sparse = [chunk("x", sparse_score=12.0)]
        fused = reciprocal_rank_fusion(dense, sparse, k=1)
        assert fused[0].dense_score == 0.9
        assert fused[0].sparse_score == 12.0

    def test_handles_empty_inputs(self):
        assert reciprocal_rank_fusion([], [], k=5) == []


class TestBM25Index:
    @pytest.fixture
    def index(self) -> BM25Index:
        return BM25Index.build(
            ["a", "b", "c"],
            [
                "Total revenues were 96,773 million in fiscal 2023",
                "Risk factors include supply chain disruption",
                "Exhibit 10.1 equity incentive plan document",
            ],
            [
                {"ticker": "TSLA", "section": "Item 8"},
                {"ticker": "AAPL", "section": "Item 1A"},
                {"ticker": "AAPL", "section": "Exhibits"},
            ],
        )

    def test_finds_an_exact_figure(self, index):
        assert index.search("96,773", k=3)[0].chunk_id == "a"

    def test_matches_the_unpunctuated_form_of_a_figure(self, index):
        assert index.search("96773", k=3)[0].chunk_id == "a"

    def test_excludes_zero_scoring_chunks(self, index):
        """Padding results with chunks containing no query term would feed fusion noise."""
        assert index.search("96,773", k=10) != []
        assert all(h.sparse_score > 0 for h in index.search("96,773", k=10))

    def test_returns_nothing_for_an_unmatched_query(self, index):
        assert index.search("zzzzz nonexistent", k=5) == []

    def test_applies_metadata_filters(self, index):
        hits = index.search("revenues risk", k=5, filters=Filters(tickers=("AAPL",)))
        assert {h.chunk_id for h in hits} <= {"b", "c"}

    def test_excludes_exhibits_by_default(self, index):
        hits = index.search("equity incentive plan", k=5, filters=Filters())
        assert "c" not in {h.chunk_id for h in hits}

    def test_records_rank(self, index):
        hits = index.search("revenues", k=3)
        assert hits[0].ranks["sparse"] == 1

    def test_roundtrips_through_disk(self, index, tmp_path):
        path = tmp_path / "bm25.pkl"
        index.save(path)
        assert len(BM25Index.load(path)) == len(index)


def _index_is_built() -> bool:
    try:
        from finrag.ingestion.pipeline import open_vector_store

        return open_vector_store()._collection.count() > 0
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _index_is_built(), reason="requires a built index")
class TestRetrieveIntegration:
    QUERY = "What were Tesla's total revenues in 2023?"

    @pytest.mark.parametrize("strategy", list(Strategy))
    def test_every_strategy_returns_results(self, strategy):
        hits = retrieve(self.QUERY, strategy, k=3)
        assert 0 < len(hits) <= 3

    def test_finds_the_income_statement(self):
        hits = retrieve(self.QUERY, Strategy.HYBRID_RERANK, k=5)
        assert any("96,773" in h.text for h in hits)

    def test_ticker_filter_restricts_results(self):
        hits = retrieve("revenues", Strategy.DENSE, Filters(tickers=("AAPL",)), k=5)
        assert {h.metadata["ticker"] for h in hits} == {"AAPL"}

    def test_exhibits_are_excluded_by_default(self):
        hits = retrieve("equity incentive plan", Strategy.HYBRID, k=10)
        assert all(h.metadata["section"] != "Exhibits" for h in hits)

    def test_exhibits_can_be_included_explicitly(self):
        hits = retrieve(
            "equity incentive plan award agreement",
            Strategy.DENSE,
            Filters(exclude_exhibits=False),
            k=10,
        )
        assert any(h.metadata["section"] == "Exhibits" for h in hits)

    def test_reranking_widens_the_candidate_pool(self, monkeypatch):
        """A cross-encoder can only reorder what it is given, so handing it exactly k
        candidates would waste it entirely."""
        seen: list[int] = []
        import finrag.retrieval as retrieval_module

        original = retrieval_module.rerank_module.rerank

        def spy(query, chunks, top_n, model_name=None):
            seen.append(len(chunks))
            return original(query, chunks, top_n, model_name)

        monkeypatch.setattr(retrieval_module.rerank_module, "rerank", spy)
        results = retrieve(self.QUERY, Strategy.DENSE_RERANK, k=3)

        assert len(results) == 3
        assert seen, "rerank was never called"
        assert seen[0] > 3, f"reranker saw only {seen[0]} candidates for k=3"

    def test_results_carry_citations(self):
        for hit in retrieve(self.QUERY, Strategy.HYBRID, k=3):
            assert "FY" in hit.citation
            assert hit.metadata["ticker"]

    def test_k_is_respected(self):
        assert len(retrieve(self.QUERY, Strategy.HYBRID, k=2)) == 2

    def test_default_k_comes_from_settings(self):
        assert len(retrieve(self.QUERY, Strategy.DENSE)) == get_settings().rerank_top_n
