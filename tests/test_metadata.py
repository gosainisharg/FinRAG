"""Filing metadata extraction, checked against the real corpus. No network, no LLM."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from finrag.ingestion.metadata import (
    Confidence,
    FormType,
    _extract_company,
    extract_filing_metadata,
)

DATA = Path(__file__).resolve().parents[1] / "data"

# Ground truth, verified by reading the cover pages.
EXPECTED = {
    "10-Q4-2024-As-Filed.pdf": ("AAPL", FormType.TEN_K, 2024, date(2024, 9, 28), 0),
    "_10-K-Q4-2023-As-Filed.pdf": ("AAPL", FormType.TEN_K, 2023, date(2023, 9, 30), 0),
    "tsla-20231231-gen.pdf": ("TSLA", FormType.TEN_K, 2023, date(2023, 12, 31), 0),
    # 10-K wrapped in an annual report: the filing proper starts on page 6.
    "cost-annual-report-final-pdf-from-dfin.pdf": (
        "COST",
        FormType.TEN_K,
        2023,
        date(2023, 9, 3),
        6,
    ),
    "Meta-12-31-2023-Exhibit-99-1-FINAL.pdf": (
        "META",
        FormType.EARNINGS_RELEASE,
        2023,
        date(2023, 12, 31),
        0,
    ),
    "Meta-Reports-First-Quarter-2024-Results-2024.pdf": (
        "META",
        FormType.EARNINGS_RELEASE,
        2024,
        date(2024, 3, 31),
        0,
    ),
}


@pytest.fixture(scope="module")
def extracted() -> dict:
    return {name: extract_filing_metadata(DATA / name) for name in EXPECTED}


@pytest.mark.parametrize("filename", sorted(EXPECTED))
def test_metadata_matches_ground_truth(extracted, filename):
    ticker, form_type, fiscal_year, period_end, body_start = EXPECTED[filename]
    meta = extracted[filename]
    assert meta.ticker == ticker
    assert meta.form_type == form_type
    assert meta.fiscal_year == fiscal_year
    assert meta.period_end == period_end
    assert meta.body_start_page == body_start


def test_every_document_is_identified(extracted):
    """No document may fall through to unknown — that would make it unfilterable."""
    for name, meta in extracted.items():
        assert meta.ticker is not None, name
        assert meta.company is not None, name
        assert meta.confidence is not Confidence.NONE, name


def test_press_releases_are_not_labelled_as_filings(extracted):
    # They have no Item sections, so chunking must treat them differently.
    for name, meta in extracted.items():
        if "Meta" in name:
            assert meta.form_type is FormType.EARNINGS_RELEASE


class TestCompanyNameExtraction:
    def test_reads_the_line_above_the_registrant_marker(self):
        page = (
            "Commission file number 0-20355\n"
            "Costco Wholesale Corporation\n"
            "(\n(Exact name of registrant"
        )
        assert _extract_company(page) == "Costco Wholesale Corporation"

    def test_does_not_run_backwards_across_lines(self):
        """Regression: flattening whitespace let the match swallow the preceding line."""
        page = (
            "Commission file number 0-20355\n"
            "Costco Wholesale Corporation\n"
            "(Exact name of registrant"
        )
        assert "Commission file number" not in _extract_company(page)

    def test_handles_the_registrant_variant_wording(self):
        page = "Tesla, Inc.\n(Exact name of the Registrant as specified in its charter)"
        assert _extract_company(page) == "Tesla, Inc."

    def test_returns_none_when_absent(self):
        assert _extract_company("a page with no registrant block at all") is None


class TestChunkMetadata:
    def test_drops_none_values(self, extracted):
        # Chroma rejects None; an absent key filters more predictably than a sentinel.
        flat = extracted["tsla-20231231-gen.pdf"].as_chunk_metadata()
        assert None not in flat.values()

    def test_values_are_scalars_only(self, extracted):
        flat = extracted["tsla-20231231-gen.pdf"].as_chunk_metadata()
        assert all(isinstance(v, str | int | float | bool) for v in flat.values())

    def test_carries_the_fields_the_router_filters_on(self, extracted):
        flat = extracted["tsla-20231231-gen.pdf"].as_chunk_metadata()
        assert flat["ticker"] == "TSLA"
        assert flat["fiscal_year"] == 2023
        assert flat["form_type"] == "10-K"
