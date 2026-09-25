"""PDF extraction that keeps financial tables intact.

Plain text extraction flattens a table into an unlabelled column of numbers: `96,773`
survives, but its association with "Total revenues" and with the year 2023 does not. An
answer generated from that is a guess dressed as a citation.

Extraction runs through ``pymupdf4llm.to_markdown``, which does layout-aware conversion and
renders tables as markdown. The earlier implementation called ``page.find_tables()``
directly and detected tables only where the PDF draws ruling lines — which meant Apple's
and Costco's primary financial statements, roughly half the corpus, came through as
linearised prose with labels and figures on separate lines.

Each page is split into two kinds of element:

- **tables**, kept as markdown and carrying the text immediately above them — statement
  title, units, period — because that is where the scale lives. A figure without
  "(in millions)" is wrong by six orders of magnitude.
- **prose**, everything else, with caption lines removed so the same text is not indexed
  twice in two different shapes.

Note: importing ``pymupdf4llm`` also changes ``pymupdf.Page.find_tables`` behaviour
globally and degrades raw cell text — labels lose their spaces ("Automotivesales") and
commas migrate onto newlines. Nothing here calls ``find_tables``; if that ever changes,
read the extraction tests before trusting the output.
"""

from __future__ import annotations

import contextlib
import io
import logging
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pymupdf4llm

from finrag.ingestion.metadata import FilingMetadata, extract_filing_metadata

logger = logging.getLogger(__name__)

# A caption is a stack of short lines. A long block directly above a table is the end of
# the preceding paragraph, not a caption; absorbing it buries the figures under unrelated
# prose and mislabels that prose as part of a table.
MAX_CAPTION_LINE_CHARS = 150
MAX_CAPTION_CHARS = 600

# Dot leaders ("Net sales . . . . . . . 237,710") are typographic filler. They cost tokens
# and add nothing an embedding can use, so they are collapsed on the way in.
_DOT_LEADER = re.compile(r"(?:\s*\.){3,}\s*")

# Standard Unicode ligatures. Always safe to expand.
_UNICODE_LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
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

# The same filing also yields U+FFFD for those glyphs depending on the extraction path,
# and U+FFFD is ambiguous — it stands for fi, ff or fl alike. Context disambiguates every
# occurrence in this corpus:
#   word-initial + "o"  -> fl   (flow)
#   word-initial        -> fi   (financial, filed, find, first)
#   mid-word + vowel    -> ff   (efforts, differ, affecting)
#   mid-word + consonant-> fi   (artificial, specific, profile)
_REPLACEMENT_CHAR = re.compile(r"(?P<before>[A-Za-z]?)\ufffd(?P<after>[A-Za-z])")
_VOWELS = frozenset("aeiou")


def _expand_replacement_char(match: re.Match[str]) -> str:
    before, after = match.group("before"), match.group("after")
    if not before:  # word-initial
        return ("fl" if after.lower() == "o" else "fi") + after
    return before + ("ff" if after.lower() in _VOWELS else "fi") + after


# Markdown decoration carries no meaning for retrieval, and emphasis markers split tokens
# for BM25 — "**Total**" would not match a query for "total".
_EMPHASIS = re.compile(r"\*{1,3}")
_HEADING = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_LINE_BREAK = re.compile(r"<br\s*/?>", re.I)
_TABLE_SEPARATOR = re.compile(r"^\|[\s|:-]+\|$")
# pymupdf4llm emits a few inline HTML tags; they are decoration, not content.
_HTML_TAG = re.compile(r"</?[a-z]+\s*/?>", re.I)


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


def _repair_ligatures(text: str) -> str:
    """Expand ligatures, including one filing's mis-decoded glyphs."""
    for glyph, expansion in _UNICODE_LIGATURES.items():
        text = text.replace(glyph, expansion)
    text = _MISDECODED_PATTERN.sub(lambda m: _MISDECODED_LIGATURES[m.group(0)], text)
    return _REPLACEMENT_CHAR.sub(_expand_replacement_char, text)


def _clean_text(text: str) -> str:
    """Strip markdown decoration and typographic filler, and repair encoding damage."""
    text = _LINE_BREAK.sub(" ", text)
    text = _HTML_TAG.sub("", text)
    text = _HEADING.sub("", text)
    text = _EMPHASIS.sub("", text)
    text = _DOT_LEADER.sub(" ", text)
    return _repair_ligatures(text).strip()


def _is_page_furniture(text: str) -> bool:
    """A bare page number or rule line — retrievable noise that answers nothing."""
    stripped = text.strip()
    return len(stripped) <= 4 and bool(re.fullmatch(r"[\d\s|_.-]*", stripped))


def _is_table_row(line: str) -> bool:
    return line.lstrip().startswith("|")


def _take_caption(prose_lines: list[str]) -> tuple[str, list[str]]:
    """Split trailing caption lines away from the prose preceding a table.

    Walks upward from the table and stops at the first long line: that is the previous
    paragraph, and everything above it belongs to the preceding text. Returns the caption
    and the prose lines that remain.
    """
    caption: list[str] = []
    total = 0
    index = len(prose_lines)

    for line in reversed(prose_lines):
        stripped = _clean_text(line)
        if not stripped:
            index -= 1
            continue
        if len(stripped) > MAX_CAPTION_LINE_CHARS or total + len(stripped) > MAX_CAPTION_CHARS:
            break
        caption.insert(0, stripped)
        total += len(stripped)
        index -= 1

    return "\n".join(caption), prose_lines[:index]


def _prose_elements(lines: list[str], page_no: int) -> list[PageElement]:
    """Split prose into paragraph-level elements, one per blank-line-separated block.

    Granularity matters beyond tidiness: section headings such as "ITEM 1A. RISK FACTORS"
    are their own paragraph, and heading detection only considers short elements. Emitting
    one element per page buries every heading inside a multi-thousand-character block and
    silently disables section labelling for the whole document.
    """
    elements: list[PageElement] = []
    paragraph: list[str] = []

    def flush() -> None:
        nonlocal paragraph
        text = _clean_text("\n".join(paragraph))
        if text and not _is_page_furniture(text):
            elements.append(PageElement(ElementKind.PROSE, text, page_no))
        paragraph = []

    for line in lines:
        if line.strip():
            paragraph.append(line)
        else:
            flush()
    flush()
    return elements


def parse_page_markdown(markdown: str, page_no: int) -> list[PageElement]:
    """Split one page's markdown into table and prose elements, in reading order."""
    elements: list[PageElement] = []
    prose: list[str] = []
    table: list[str] = []

    def close_table() -> None:
        nonlocal prose, table
        if not table:
            return

        # A lone row is a layout artefact, not a table; keep it as prose rather than
        # emitting a one-row table that fusion would treat as a financial statement.
        rows = [r for r in table if not _TABLE_SEPARATOR.match(r.strip())]
        if len(rows) < 2:
            prose.extend(table)
            table = []
            return

        caption, remaining = _take_caption(prose)
        elements.extend(_prose_elements(remaining, page_no))
        prose = []

        body = "\n".join(_clean_text(row) for row in table)
        elements.append(
            PageElement(ElementKind.TABLE, f"{caption}\n\n{body}" if caption else body, page_no)
        )
        table = []

    for line in markdown.splitlines():
        if _is_table_row(line):
            table.append(line)
            continue
        if table:
            close_table()
        prose.append(line)

    close_table()
    elements.extend(_prose_elements(prose, page_no))
    return elements


def load_document(path: Path | str) -> tuple[FilingMetadata, list[PageElement]]:
    """Extract a filing's metadata and its page elements.

    Pages before ``body_start_page`` are skipped: for a 10-K wrapped in an annual report
    they are marketing front matter, and indexing them adds retrievable text that answers
    nothing.
    """
    path = Path(path)
    metadata = extract_filing_metadata(path)

    # to_markdown writes a progress bar to stdout, which would corrupt CLI output.
    with contextlib.redirect_stdout(io.StringIO()):
        pages = pymupdf4llm.to_markdown(str(path), page_chunks=True, show_progress=False)

    elements: list[PageElement] = []
    for page_no, page in enumerate(pages):
        if page_no < metadata.body_start_page:
            continue
        elements.extend(parse_page_markdown(page["text"], page_no))

    return metadata, elements
