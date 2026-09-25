"""PDF extraction. Runs against the real corpus; no network, no LLM."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from finrag.ingestion.loaders import (
    ElementKind,
    _clean_text,
    _is_page_furniture,
    _repair_ligatures,
    _take_caption,
    load_document,
    parse_page_markdown,
)

DATA = Path(__file__).resolve().parents[1] / "data"
TESLA = DATA / "tsla-20231231-gen.pdf"

# The row that must survive extraction intact for every filing, and the kind of chunk it
# has to land in. Labels and figures on separate lines is the failure this guards against.
PRIMARY_REVENUE_ROW = {
    "tsla-20231231-gen.pdf": ("Total revenues", "96,773"),
    "10-Q4-2024-As-Filed.pdf": ("Total net sales", "391,035"),
    "_10-K-Q4-2023-As-Filed.pdf": ("Total net sales", "383,285"),
    "cost-annual-report-final-pdf-from-dfin.pdf": ("Total revenue", "242,290"),
    "Meta-12-31-2023-Exhibit-99-1-FINAL.pdf": ("Total revenue", "40,111"),
    "Meta-Reports-First-Quarter-2024-Results-2024.pdf": ("Total revenue", "36,455"),
}


class TestParsePageMarkdown:
    def test_splits_prose_and_table(self):
        page = "Some narrative text.\n\n|a|b|\n|---|---|\n|1|2|\n\nMore narrative."
        kinds = [e.kind for e in parse_page_markdown(page, 7)]
        assert ElementKind.TABLE in kinds
        assert kinds.count(ElementKind.PROSE) >= 1

    def test_table_carries_the_caption_above_it(self):
        """Without the units, a figure is wrong by six orders of magnitude."""
        page = (
            "Tesla, Inc.\nConsolidated Statements of Operations\n(in millions)\n\n"
            "|x|2023|\n|---|---|\n|Revenue|96,773|"
        )
        table = next(e for e in parse_page_markdown(page, 1) if e.is_table)
        assert "in millions" in table.text
        assert "96,773" in table.text

    def test_caption_stops_at_a_paragraph(self):
        """A long block above a table is the previous paragraph, not a caption."""
        paragraph = "This is a long narrative paragraph about tax positions. " * 4
        page = f"{paragraph}\nRevenues\n(in millions)\n\n|x|2023|\n|---|---|\n|Revenue|96,773|"
        table = next(e for e in parse_page_markdown(page, 1) if e.is_table)
        assert "tax positions" not in table.text
        assert "in millions" in table.text

    def test_caption_is_not_also_emitted_as_prose(self):
        """Regression: the caption was indexed twice, as table and as loose prose."""
        page = (
            "Consolidated Statements of Operations\n(in millions)\n\n"
            "|x|2023|\n|---|---|\n|Revenue|1|"
        )
        prose = [e.text for e in parse_page_markdown(page, 1) if e.kind is ElementKind.PROSE]
        assert not any("Consolidated Statements" in t for t in prose)

    def test_a_single_row_is_not_treated_as_a_table(self):
        page = "Narrative.\n\n|just one row|\n\nMore narrative."
        assert all(not e.is_table for e in parse_page_markdown(page, 1))

    def test_markdown_emphasis_is_stripped(self):
        """'**Total**' would not match a BM25 query for 'total'."""
        page = (
            "# **Apple Inc.**\n\n|**Total net sales**|**391,035**|\n|---|---|\n|Products|294,866|"
        )
        table = next(e for e in parse_page_markdown(page, 1) if e.is_table)
        assert "*" not in table.text
        assert "Total net sales" in table.text

    def test_page_furniture_is_dropped(self):
        assert parse_page_markdown("50", 50) == []

    def test_empty_page(self):
        assert parse_page_markdown("", 1) == []


class TestTakeCaption:
    def test_returns_trailing_short_lines(self):
        caption, remaining = _take_caption(["A paragraph " * 20, "Revenues", "(in millions)"])
        assert caption == "Revenues\n(in millions)"
        assert len(remaining) == 1

    def test_returns_nothing_when_preceded_only_by_a_paragraph(self):
        caption, remaining = _take_caption(["A long narrative paragraph. " * 10])
        assert caption == ""
        assert len(remaining) == 1

    def test_handles_empty_input(self):
        assert _take_caption([]) == ("", [])


class TestTextCleaning:
    def test_collapses_dot_leaders(self):
        assert _clean_text("Net sales . . . . . . . 237,710") == "Net sales 237,710"

    def test_preserves_decimals(self):
        assert "3.14" in _clean_text("ratio of 3.14 percent")

    def test_strips_headings_and_line_breaks(self):
        assert _clean_text("# **Apple Inc.**<br>") == "Apple Inc."

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


class TestLoadDocument:
    @pytest.mark.parametrize("filename", sorted(PRIMARY_REVENUE_ROW))
    def test_primary_revenue_row_is_a_table_row(self, filename):
        """The regression that motivated switching extraction backends.

        find_tables() only detected tables with ruling lines, so Apple's and Costco's
        statements arrived as prose with labels and figures on separate lines — half the
        corpus, and the half most likely to be asked about.
        """
        label, figure = PRIMARY_REVENUE_ROW[filename]
        _, elements = load_document(DATA / filename)

        rows = [
            line
            for element in elements
            if element.is_table
            for line in element.text.splitlines()
            if line.strip().startswith("|") and re.search(rf"\b{re.escape(label)}\b", line, re.I)
        ]
        assert rows, f"{label!r} never appeared as a table row"
        assert any(figure in row for row in rows), f"{figure} not found alongside {label!r}"

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

    def test_ligature_damage_is_repaired_in_the_real_filing(self):
        _, elements = load_document(DATA / "Meta-Reports-First-Quarter-2024-Results-2024.pdf")
        joined = " ".join(e.text for e in elements)
        assert "ínancial" not in joined
        assert "financial" in joined
