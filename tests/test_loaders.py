"""PDF extraction. Runs against the real corpus; no network, no LLM."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from finrag.ingestion.loaders import (
    ElementKind,
    _clean_text,
    _compact_rows,
    _find_year_header,
    _is_page_furniture,
    _repair_ligatures,
    extract_page_elements,
    load_document,
)

DATA = Path(__file__).resolve().parents[1] / "data"
TESLA = DATA / "tsla-20231231-gen.pdf"
TESLA_INCOME_STATEMENT_PAGE = 50


class TestCompactRows:
    def test_drops_currency_cells_so_columns_align(self):
        """The bug this exists for: `$` cells shifted values into different columns."""
        rows = _compact_rows(
            [
                ["Automotive sales", "$", "78,509", "$", "67,210", "$", "44,125"],
                ["Automotive leasing", "2,120", "", "2,476", "", "1,642", ""],
            ]
        )
        assert rows[0] == ["Automotive sales", "78,509", "67,210", "44,125"]
        assert rows[1] == ["Automotive leasing", "2,120", "2,476", "1,642"]

    def test_all_rows_have_equal_width(self):
        rows = _compact_rows([["a", "$", "1", "2", "3"], ["b", "4", "5", "6"], ["heading"]])
        assert len({len(r) for r in rows}) == 1

    def test_width_is_the_mode_not_the_maximum(self):
        """One malformed row must not widen the whole table."""
        rows = _compact_rows(
            [["a", "1", "2"], ["b", "3", "4"], ["c", "5", "6"], ["junk", "7", "8", "9", "10"]]
        )
        assert all(len(r) == 3 for r in rows)

    def test_section_headings_survive_as_label_only_rows(self):
        rows = _compact_rows([["Revenues", "", ""], ["Automotive sales", "1", "2"]])
        assert rows[0][0] == "Revenues"
        assert rows[0][1:] == ["", ""]

    def test_empty_input(self):
        assert _compact_rows([]) == []


class TestYearHeader:
    def test_recovers_years_printed_above_the_table(self):
        preamble = "Consolidated Statements of Operations\n2023\n2022\n2021"
        assert _find_year_header(preamble, 4) == ["2023", "2022", "2021"]

    def test_rejects_a_run_that_does_not_match_the_column_count(self):
        # Guessing here would silently mislabel every figure in the table.
        assert _find_year_header("2023\n2022", 4) is None

    def test_ignores_years_embedded_in_prose(self):
        assert _find_year_header("during 2023 the company grew", 2) is None


class TestTextCleaning:
    def test_collapses_dot_leaders(self):
        assert _clean_text("Net sales . . . . . . . 237,710") == "Net sales 237,710"

    def test_preserves_decimals(self):
        assert "3.14" in _clean_text("ratio of 3.14 percent")

    @pytest.mark.parametrize(
        "corrupt, expected",
        [
            ("ínancial", "financial"),
            ("eìorts", "efforts"),
            ("îow", "flow"),
            ("artiícial", "artificial"),
            ("írst", "first"),
        ],
    )
    def test_repairs_misdecoded_ligatures(self, corrupt, expected):
        assert _repair_ligatures(corrupt) == expected

    @pytest.mark.parametrize("text", ["José", "naïve", "í"])
    def test_leaves_genuine_accented_characters_alone(self, text):
        assert _repair_ligatures(text) == text

    def test_expands_standard_unicode_ligatures(self):
        assert _repair_ligatures("ﬁnancial ﬂow") == "financial flow"

    @pytest.mark.parametrize("text", ["50", "  12  ", "---", "|"])
    def test_identifies_page_furniture(self, text):
        assert _is_page_furniture(text)

    @pytest.mark.parametrize("text", ["Revenues", "2023 2022 2021"])
    def test_keeps_real_content(self, text):
        assert not _is_page_furniture(text)


class TestPageExtraction:
    @pytest.fixture(scope="class")
    def income_statement(self) -> list:
        with pymupdf.open(TESLA) as doc:
            return extract_page_elements(doc[TESLA_INCOME_STATEMENT_PAGE], 50)

    def test_finds_the_table(self, income_statement):
        assert sum(e.is_table for e in income_statement) == 1

    def test_table_carries_its_units(self, income_statement):
        """Without the scale, 96,773 is wrong by six orders of magnitude."""
        table = next(e for e in income_statement if e.is_table)
        assert "in millions" in table.text

    def test_table_carries_year_headers(self, income_statement):
        table = next(e for e in income_statement if e.is_table)
        assert "| 2023 | 2022 | 2021 |" in table.text

    def test_row_labels_stay_attached_to_their_figures(self, income_statement):
        table = next(e for e in income_statement if e.is_table)
        assert "| Total revenues | 96,773 | 81,462 | 53,823 |" in table.text

    def test_preamble_is_not_also_emitted_as_prose(self, income_statement):
        """Regression: the caption was indexed twice, as table and as loose prose."""
        prose = [e.text for e in income_statement if e.kind is ElementKind.PROSE]
        assert not any("Consolidated Statements of Operations" in t for t in prose)

    def test_page_number_is_dropped(self, income_statement):
        assert "50" not in [e.text.strip() for e in income_statement]


class TestLoadDocument:
    def test_skips_annual_report_front_matter(self):
        # Costco's 10-K starts on page 6; the marketing pages before it answer nothing.
        meta, elements = load_document(DATA / "cost-annual-report-final-pdf-from-dfin.pdf")
        assert meta.body_start_page == 6
        assert min(e.page for e in elements) >= 6

    def test_returns_metadata_and_elements(self):
        meta, elements = load_document(TESLA)
        assert meta.ticker == "TSLA"
        assert len(elements) > 100
        assert any(e.is_table for e in elements)

    def test_no_empty_elements(self):
        _, elements = load_document(DATA / "Meta-12-31-2023-Exhibit-99-1-FINAL.pdf")
        assert all(e.text.strip() for e in elements)
