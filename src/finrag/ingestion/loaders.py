"""PDF extraction that keeps financial tables intact.

Plain text extraction flattens a table into an unlabelled column of numbers: `96,773`
survives, but its association with "Total revenues" and with the year 2023 does not. An
answer generated from that is a guess dressed as a citation.

This module splits each page into two kinds of element:

- **tables**, recovered structurally via ``find_tables()`` and rendered as markdown,
  carrying the text immediately above them — statement title, units, year headers — because
  that is where the scale and the period live. A figure without "(in millions)" is off by
  six orders of magnitude.
- **prose**, everything else, with the table regions removed so the same content is not
  indexed twice in two different shapes.

Known limitation: table detection uses the ``lines`` strategy, which needs ruling lines.
Costco's filing draws its statements with whitespace alignment alone, so its tables are
not detected and fall through to prose. The ``text`` strategy was measured and rejected —
it shatters Costco's headings mid-word (``CONSOLIDATED | STATEME | NTS OF I | NCOME``) and
hallucinates an 89-row table on every prose page of the Tesla filing. Costco's prose
extraction keeps label-value adjacency, so it degrades rather than breaks.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pymupdf

from finrag.ingestion.metadata import FilingMetadata, extract_filing_metadata

# How far above a table to look for its caption, in PDF points (~1/72 inch). 80pt covers
# the title / units / period / year-header stack seen on statement pages without reaching
# into the previous paragraph.
PREAMBLE_GAP_PT = 80.0
# A caption is a stack of short lines. A long block directly above a table is the end of
# the preceding paragraph, not a caption, and absorbing it buries the figures under
# unrelated prose and mislabels that prose as part of a table.
MAX_CAPTION_LINE_CHARS = 150
MAX_CAPTION_CHARS = 600

# A text block this fraction inside a table's bounding box is considered part of it.
# Partial overlap is common at table edges, so a bare intersection test drops neighbouring
# prose that merely touches the border.
TABLE_OVERLAP_RATIO = 0.5

_YEAR_LINE = re.compile(r"^\s*(19|20)\d{2}\s*$")
# Cells that carry formatting rather than data. Dropping them is what lets every row
# be packed to the same width, so a column means the same period in every row.
_NON_DATA_CELLS = frozenset({"$", "%", "(", ")", "", "—", "-"})
# Dot leaders ("Net sales . . . . . . . 237,710") are typographic filler. They cost tokens
# and add nothing an embedding can use, so they are collapsed on the way in.
_DOT_LEADER = re.compile(r"(?:\s*\.){3,}\s*")

# Standard Unicode ligatures. Always safe to expand.
_UNICODE_LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}

# One filing's font encoding maps ligature glyphs onto Latin-1 accented characters, so
# "financial" extracts as "ínancial" and "effect" as "eìect". Left alone this is a silent
# retrieval failure: BM25 can never match the query term against the corrupted token.
# Mappings were derived from the affected words themselves (ínancial, eìorts, îow, ...),
# not guessed. Applied only inside ASCII words so a genuine accented name is untouched.
_MISDECODED_LIGATURES = {"í": "fi", "ì": "ff", "î": "fl"}
_MISDECODED_PATTERN = re.compile(
    "|".join(f"(?<=[A-Za-z]){c}|{c}(?=[A-Za-z])" for c in _MISDECODED_LIGATURES)
)


class ElementKind(StrEnum):
    PROSE = "prose"
    TABLE = "table"


@dataclass(frozen=True)
class PageElement:
    """One extracted region of a page, ready to be chunked."""

    kind: ElementKind
    text: str
    page: int

    @property
    def is_table(self) -> bool:
        return self.kind is ElementKind.TABLE


def _compact_rows(rows: list[list[str | None]]) -> list[list[str]]:
    """Normalise a raw table into `[label, value, value, ...]` rows of uniform width.

    Raw extraction pads tables with spacer columns and puts currency symbols in cells of
    their own, so the same period lands in a different column on different rows::

        Automotive sales              | $ | 78,509 | $ | 67,210 | $ | 44,125
        Automotive regulatory credits | 1,790 |  | 1,776 |  | 1,465 |

    Read as markdown that invites pairing a figure with the wrong year. Dropping the
    non-data cells and left-packing what remains makes column position meaningful again.

    Width is the most common value count rather than the maximum, so one malformed row
    cannot widen the whole table. Rows carrying no values at all — section headings such
    as "Cost of revenues" — are kept as labels with empty cells.
    """
    if not rows:
        return []

    split: list[tuple[str, list[str]]] = []
    for row in rows:
        cells = [(cell or "").strip() for cell in row]
        label = _clean_text(cells[0]) if cells else ""
        values = [_clean_text(c) for c in cells[1:] if c not in _NON_DATA_CELLS]
        split.append((label, values))

    counts = Counter(len(values) for _, values in split if values)
    if not counts:
        return [[label] for label, _ in split]
    width = counts.most_common(1)[0][0]

    packed: list[list[str]] = []
    for label, values in split:
        fitted = values[:width] + [""] * max(0, width - len(values))
        packed.append([label, *fitted])
    return packed


def _find_year_header(preamble: str, column_count: int) -> list[str] | None:
    """Recover year column headers from the caption, when they line up.

    Statement pages print the periods above the table rather than inside it, so the
    extracted table has no usable header. When the caption contains a bare run of years
    matching the table's numeric column count, those years are the headers — attaching
    them is what stops the model reading a 2021 figure as the current year.
    """
    run: list[str] = []
    for line in preamble.splitlines():
        if _YEAR_LINE.match(line):
            run.append(line.strip())
            continue
        if run:
            # A run ended. Accept it only if it matches the value-column count exactly.
            if len(run) == column_count - 1:  # first column holds row labels
                return run
            run = []
    if len(run) == column_count - 1:
        return run
    return None


def _looks_like_header(row: list[str]) -> bool:
    """Whether a row reads as column names rather than data.

    Promoting a data row to the header is silent corruption: the figures survive but
    markdown presents them as column labels, and the row itself stops reading as data.
    """
    values = [cell for cell in row[1:] if cell]
    if not values:
        return False
    textual = sum(not re.search(r"\d", cell) for cell in values)
    return textual / len(values) >= 0.6


def _render_table(rows: list[list[str]], preamble: str) -> str:
    """Render a table as markdown, using year headers from the caption when available."""
    if not rows:
        return ""

    width = len(rows[0])
    years = _find_year_header(preamble, width)

    if years:
        header = ["", *years]
        body = rows
    elif _looks_like_header(rows[0]):
        header = rows[0]
        body = rows[1:]
    else:
        # No recoverable header. An empty one keeps every data row as data; the periods
        # are still named in the caption directly above.
        header = [""] * width
        body = rows

    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * width) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def _overlap_ratio(block: pymupdf.Rect, table: pymupdf.Rect) -> float:
    intersection = block & table
    if not intersection.is_valid or block.get_area() == 0:
        return 0.0
    return intersection.get_area() / block.get_area()


def _repair_ligatures(text: str) -> str:
    """Expand ligatures, including one filing's mis-decoded glyphs."""
    for glyph, expansion in _UNICODE_LIGATURES.items():
        text = text.replace(glyph, expansion)
    return _MISDECODED_PATTERN.sub(lambda m: _MISDECODED_LIGATURES[m.group(0)], text)


def _clean_text(text: str) -> str:
    """Strip typographic filler and repair encoding damage."""
    return _repair_ligatures(_DOT_LEADER.sub(" ", text)).strip()


def _is_page_furniture(text: str) -> bool:
    """A bare page number or rule line — retrievable noise that answers nothing."""
    stripped = text.strip()
    return len(stripped) <= 4 and bool(re.fullmatch(r"[\d\s|_.-]*", stripped))


def _preamble_for(blocks: list[tuple], table_rect: pymupdf.Rect) -> tuple[str, set[int]]:
    """Text sitting directly above a table: its title, units and period headers.

    Returns the caption and the indices of the blocks it consumed. The caller skips those
    in the prose pass — otherwise the caption is indexed twice, once inside the table
    element and once as standalone prose, producing near-duplicate retrievals that crowd
    out genuinely different context.
    """
    candidates: list[tuple[float, int, str]] = []
    for index, (x0, y0, x1, y1, text, *_) in enumerate(blocks):
        block = pymupdf.Rect(x0, y0, x1, y1)
        above = y1 <= table_rect.y0 + 2
        near = table_rect.y0 - y1 <= PREAMBLE_GAP_PT
        horizontally_related = block.x0 < table_rect.x1 and block.x1 > table_rect.x0
        if above and near and horizontally_related and text.strip():
            candidates.append((y0, index, text.strip()))

    # Walk upwards from the table. Stop at the first long block: that is the previous
    # paragraph, and everything above it belongs to the preceding text, not this table.
    found: list[tuple[float, str]] = []
    consumed: set[int] = set()
    total = 0
    for y0, index, text in sorted(candidates, reverse=True):
        if len(text) > MAX_CAPTION_LINE_CHARS or total + len(text) > MAX_CAPTION_CHARS:
            break
        found.append((y0, text))
        consumed.add(index)
        total += len(text)

    return "\n".join(text for _, text in sorted(found)), consumed


def extract_page_elements(page: pymupdf.Page, page_no: int) -> list[PageElement]:
    """Split a page into table and prose elements, in reading order."""
    elements: list[tuple[float, PageElement]] = []
    blocks = page.get_text("blocks")

    table_rects: list[pymupdf.Rect] = []
    consumed_blocks: set[int] = set()

    for table in page.find_tables().tables:
        rect = pymupdf.Rect(table.bbox)
        table_rects.append(rect)

        rows = _compact_rows(table.extract())
        if len(rows) < 2:  # a single row is a layout artefact, not a table
            continue

        preamble, consumed = _preamble_for(blocks, rect)
        consumed_blocks |= consumed
        rendered = _render_table(rows, preamble)
        text = f"{preamble}\n\n{rendered}" if preamble else rendered
        elements.append((rect.y0, PageElement(ElementKind.TABLE, text, page_no)))

    for index, (x0, y0, x1, y1, text, *_) in enumerate(blocks):
        if index in consumed_blocks:
            continue
        block = pymupdf.Rect(x0, y0, x1, y1)
        # Skip anything already captured structurally, so content is not indexed twice.
        if any(_overlap_ratio(block, r) >= TABLE_OVERLAP_RATIO for r in table_rects):
            continue
        cleaned = _clean_text(text)
        if cleaned and not _is_page_furniture(cleaned):
            elements.append((y0, PageElement(ElementKind.PROSE, cleaned, page_no)))

    return [element for _, element in sorted(elements, key=lambda pair: pair[0])]


def load_document(path: Path | str) -> tuple[FilingMetadata, list[PageElement]]:
    """Extract a filing's metadata and its page elements.

    Pages before ``body_start_page`` are skipped: for a 10-K wrapped in an annual report
    they are marketing front matter, and indexing them adds retrievable text that answers
    nothing.
    """
    path = Path(path)
    metadata = extract_filing_metadata(path)

    elements: list[PageElement] = []
    with pymupdf.open(path) as doc:
        for page_no in range(metadata.body_start_page, len(doc)):
            elements.extend(extract_page_elements(doc[page_no], page_no))

    return metadata, elements
