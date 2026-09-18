"""Chunking and section labelling. Real corpus; no network, no LLM."""

from __future__ import annotations

import collections
from pathlib import Path

import pytest

from finrag.ingestion.chunking import (
    EXHIBITS_SECTION,
    MAX_TABLE_CHARS,
    UNKNOWN_SECTION,
    _chunk_id,
    _split_table,
    chunk_document,
    detect_section,
    detect_section_in_table,
)
from finrag.ingestion.loaders import load_document

DATA = Path(__file__).resolve().parents[1] / "data"


class TestDetectSection:
    @pytest.mark.parametrize(
        "heading, expected",
        [
            ("ITEM 1A. RISK FACTORS", "Item 1A"),  # Tesla: uppercase, period
            ("Item 1A. \nRisk Factors", "Item 1A"),  # Apple: mixed case, newline
            ("Item 1A—Risk Factors", "Item 1A"),  # Costco: em dash
            ("Item 7. Management's Discussion and Analysis", "Item 7"),
        ],
    )
    def test_accepts_every_filing_style(self, heading, expected):
        assert detect_section(heading)[0] == expected

    def test_canonicalises_the_title_across_issuers(self):
        """Cross-company filters only work if the label is identical everywhere."""
        tesla = detect_section("ITEM 1A. RISK FACTORS")
        costco = detect_section("Item 1A—Risk Factors")
        assert tesla == costco == ("Item 1A", "Risk Factors")

    def test_rejects_toc_entries_with_a_trailing_page_number(self):
        assert detect_section("Item 1A. Risk Factors 14") is None

    def test_rejects_prose_cross_references(self):
        assert detect_section("as described in Part II, Item 8 of this report") is None

    def test_rejects_unknown_item_numbers(self):
        assert detect_section("Item 47. Something Invented") is None

    def test_rejects_long_text(self):
        assert detect_section("Item 1A. " + "x" * 200) is None


class TestDetectSectionInTable:
    def test_finds_a_heading_absorbed_into_a_table_caption(self):
        """Regression: Apple's Item 8 heading sat in a table caption and was missed,
        filing 24 pages of financial statements under Market Risk."""
        element = (
            "Item 8. \nFinancial Statements and Supplementary Data\n"
            "Index to Consolidated Financial Statements\n\n"
            "|  | 2024 |\n|---|---|\n| Revenue | 1 |"
        )
        assert detect_section_in_table(element) == (
            "Item 8",
            "Financial Statements and Supplementary Data",
        )

    def test_ignores_a_cross_reference_deeper_in_the_caption(self):
        element = "Some caption\nmore text\nsee Part II, Item 8\n\n|  | a |\n|---|---|\n| b | 1 |"
        assert detect_section_in_table(element) is None


class TestChunkId:
    def test_is_stable_for_identical_content(self):
        assert _chunk_id("a.pdf", "text") == _chunk_id("a.pdf", "text")

    def test_changes_when_content_changes(self):
        # This is what makes an edited document produce new ids.
        assert _chunk_id("a.pdf", "text") != _chunk_id("a.pdf", "text!")

    def test_same_text_in_different_filings_stays_distinct(self):
        # Shared boilerplate must remain separately attributable.
        assert _chunk_id("a.pdf", "boilerplate") != _chunk_id("b.pdf", "boilerplate")


class TestSplitTable:
    def test_repeats_the_header_in_every_part(self):
        table = "| | 2023 |\n|---|---|\n" + "\n".join(f"| row{i} | {i} |" for i in range(400))
        parts = _split_table(table, 1000)
        assert len(parts) > 1
        assert all(p.startswith("| | 2023 |\n|---|---|") for p in parts)

    def test_never_splits_a_row(self):
        table = "| | 2023 |\n|---|---|\n" + "\n".join(f"| row{i} | {i} |" for i in range(400))
        for part in _split_table(table, 1000):
            for line in part.splitlines():
                assert line.startswith("|") and line.endswith("|")


class TestChunkDocument:
    @pytest.fixture(scope="class")
    def tesla(self):
        meta, elements = load_document(DATA / "tsla-20231231-gen.pdf")
        return chunk_document(meta, elements)

    @pytest.fixture(scope="class")
    def apple(self):
        meta, elements = load_document(DATA / "10-Q4-2024-As-Filed.pdf")
        return chunk_document(meta, elements)

    def test_sections_are_contiguous_and_ordered(self, tesla):
        """A section's pages must not interleave with another's — that would mean a
        heading was detected in the wrong place."""
        pages = collections.defaultdict(list)
        for chunk in tesla:
            pages[chunk.metadata["section"]].append(chunk.metadata["page"])
        for section in ("Item 1A", "Item 7", "Item 8"):
            span = pages[section]
            assert max(span) - min(span) + 1 >= len(set(span))

    def test_risk_factors_span_matches_the_table_of_contents(self, tesla):
        # TOC says Risk Factors p14, Unresolved Staff Comments p28.
        pages = [c.metadata["page"] for c in tesla if c.metadata["section"] == "Item 1A"]
        assert min(pages) == 14
        assert max(pages) <= 28

    def test_financial_statements_are_labelled_item_8_not_market_risk(self, apple):
        """Regression: Apple's Item 8 heading was absorbed into a table caption."""
        item_8 = [c.metadata["page"] for c in apple if c.metadata["section"] == "Item 8"]
        assert item_8, "Item 8 was never detected"
        assert min(item_8) == 30
        item_7a = [c.metadata["page"] for c in apple if c.metadata["section"] == "Item 7A"]
        assert max(item_7a) < min(item_8)

    def test_no_section_is_taken_from_the_table_of_contents(self, apple):
        """Regression: a TOC entry on page 2 was accepted as a real heading."""
        early = [c for c in apple if c.metadata["page"] <= 2]
        assert all(c.metadata["section"] == UNKNOWN_SECTION for c in early)

    def test_exhibits_are_separated_from_the_filing_body(self, apple):
        exhibits = [c for c in apple if c.metadata["section"] == EXHIBITS_SECTION]
        assert exhibits, "exhibits were not identified"
        # They follow the signature page, so they sit at the end of the document.
        body_pages = [
            c.metadata["page"]
            for c in apple
            if c.metadata["section"] not in (EXHIBITS_SECTION, UNKNOWN_SECTION)
        ]
        assert min(c.metadata["page"] for c in exhibits) > max(body_pages)

    def test_press_releases_have_no_item_sections(self):
        meta, elements = load_document(DATA / "Meta-12-31-2023-Exhibit-99-1-FINAL.pdf")
        chunks = chunk_document(meta, elements)
        assert chunks
        assert all(c.metadata["section"] == UNKNOWN_SECTION for c in chunks)

    def test_every_chunk_carries_filtering_metadata(self, tesla):
        for chunk in tesla:
            assert chunk.metadata["ticker"] == "TSLA"
            assert chunk.metadata["fiscal_year"] == 2023
            assert "page" in chunk.metadata
            assert chunk.metadata["element_kind"] in {"prose", "table"}

    def test_chunk_ids_are_unique(self, tesla):
        ids = [c.chunk_id for c in tesla]
        assert len(ids) == len(set(ids))

    def test_no_chunk_exceeds_the_table_limit_by_much(self, tesla):
        # Prose flushes at TARGET_CHARS; tables are capped by MAX_TABLE_CHARS.
        assert max(len(c) for c in tesla) <= MAX_TABLE_CHARS

    def test_no_empty_chunks(self, tesla):
        assert all(c.text.strip() for c in tesla)
