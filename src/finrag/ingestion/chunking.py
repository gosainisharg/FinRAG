"""Grouping page elements into retrievable chunks.

Two rules shape this:

- **A table is never split mid-row.** Splitting one strands figures from their labels and
  period headers, which is the failure the extraction work existed to prevent. Oversized
  tables are divided by row groups with the caption and header repeated in each part.
- **Prose carries its section.** A 10-K is a fixed document structure, and "Item 1A Risk
  Factors" versus "Item 7 MD&A" is the single most useful filter the router has. Section
  labels are canonicalised so Tesla's ``ITEM 1A. RISK FACTORS`` and Costco's
  ``Item 1A—Risk Factors`` become the same value and cross-company queries work.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from finrag.ingestion.loaders import ElementKind, PageElement
from finrag.ingestion.metadata import FilingMetadata

# ~300 tokens of prose. Small enough that a chunk is about one idea, large enough that a
# sentence keeps the context that disambiguates it.
TARGET_CHARS = 1200
# Trailing context repeated into the next chunk, so a fact split across a boundary is still
# retrievable from either side.
OVERLAP_CHARS = 200
# Above this, a table is split by row groups rather than indexed whole.
MAX_TABLE_CHARS = 4000

# Canonical 10-K item titles. Filings word and punctuate these differently; normalising
# means a filter on "Item 1A" behaves identically across issuers.
_ITEM_TITLES: dict[str, str] = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity",
    "6": "Reserved",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}

# Separators vary by filing: "Item 1A." (Tesla, Apple) and "Item 1A—" (Costco).
_ITEM_HEADING = re.compile(r"^\s*Item\s+(\d{1,2}[A-C]?)\s*[.:—–-]?\s*(.{0,90})", re.I | re.S)
# A table-of-contents line ends with the page it points at. Real headings do not.
_TRAILING_PAGE_NUMBER = re.compile(r"\d{1,3}\s*$")

# A page carrying this many Item headings is a table of contents, not a section start.
# Filtering on a trailing page number alone is not enough: some filings put the page
# number in a separate layout block, so the entry looks exactly like a real heading.
TOC_HEADING_DENSITY = 5

UNKNOWN_SECTION = "unknown"
# Content after the signature page is attached exhibits — debt indentures, stock plans,
# trading policies. Legally part of the filing, but separate documents, and they would
# otherwise inherit the last Item label: half of Apple's chunks were filed under
# "Item 16 Form 10-K Summary", which is a one-line section. Labelling them lets the
# router exclude them without deleting content that might occasionally be asked about.
EXHIBITS_SECTION = "Exhibits"
EXHIBITS_TITLE = "Signatures and Exhibits"
_SIGNATURES_HEADING = re.compile(r"^\s*SIGNATURES?\s*\Z", re.I)


@dataclass(frozen=True)
class Chunk:
    """One indexable unit of text plus everything needed to filter and cite it."""

    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.text)


def _chunk_id(source: str, text: str) -> str:
    """Stable identifier derived from content.

    Re-ingesting an unchanged document yields identical ids, so the write is a no-op.
    Editing a document changes the text and therefore the id — which is why the pipeline
    deletes a source's chunks before re-adding rather than upserting, otherwise the
    superseded chunk would linger and stay retrievable.

    The source is part of the hash so that identical boilerplate appearing in two filings
    remains two distinct, separately attributable chunks.
    """
    digest = hashlib.sha256(f"{source}|{text}".encode())
    return digest.hexdigest()[:16]


def detect_section(text: str) -> tuple[str, str] | None:
    """Return ``(item, canonical_title)`` if the text is a section heading.

    Table-of-contents entries are rejected: they match the same pattern but point at a
    page they are not in, so treating one as a heading would label the entire front of the
    document with the wrong section.
    """
    if len(text) > 120:
        return None

    match = _ITEM_HEADING.match(text)
    if not match:
        return None

    item = match.group(1).upper()
    if item not in _ITEM_TITLES:
        return None

    if _TRAILING_PAGE_NUMBER.search(text.strip()):
        return None

    return f"Item {item}", _ITEM_TITLES[item]


def _table_caption(text: str) -> str:
    """The caption lines of a rendered table element, before the markdown grid."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("|"):
            return "\n".join(lines[:index]).strip()
    return text


def detect_section_in_table(text: str) -> tuple[str, str] | None:
    """Detect a section heading absorbed into a table element.

    Some filings place a heading immediately above an index table, so caption capture
    swallows it. Missing one is expensive: every page until the next heading inherits
    the previous section's label, which put 24 pages of Apple's financial statements
    under "Market Risk".

    Only the first two caption lines are considered, so a prose cross-reference such as
    "see Part II, Item 8" cannot be mistaken for a heading.
    """
    caption = _table_caption(text)
    if not caption:
        return None
    return detect_section("\n".join(caption.splitlines()[:2]))


def _toc_pages(elements: list[PageElement]) -> set[int]:
    """Pages dense with Item headings, i.e. tables of contents.

    Counts headings inside table rows as well as standalone ones: a filing's index is
    often laid out as a table, and missing it means every entry is treated as a real
    section start — and the word "Signatures" in that index is read as the end of the
    filing body.
    """
    per_page: dict[int, int] = {}
    for element in elements:
        lines = element.text.splitlines()
        hits = sum(1 for line in lines if _ITEM_HEADING.match(line.strip().strip("|").strip()))
        if hits:
            per_page[element.page] = per_page.get(element.page, 0) + hits
    return {page for page, count in per_page.items() if count >= TOC_HEADING_DENSITY}


def _signature_page(elements: list[PageElement], toc_pages: set[int]) -> int | None:
    """Page of the signature block, which ends the filing body.

    Table-of-contents pages are excluded: they list "Signatures" as an entry, and
    treating that as the boundary would label the whole document as exhibits.
    """
    # Take the last match, not the first. An index lists "Signatures" as an entry, and a
    # cross-reference can mention it too; the real signature block is the final one.
    found: int | None = None
    for element in elements:
        if element.page in toc_pages or element.is_table:
            continue
        if _SIGNATURES_HEADING.match(element.text.strip()):
            found = element.page
    return found


def _split_table(text: str, limit: int) -> list[str]:
    """Divide an oversized table by row groups, repeating caption and header in each part.

    Without the repeat, the second part is a wall of numbers with no labels — retrievable
    but useless, which is worse than not retrieving it.
    """
    lines = text.splitlines()
    header_end = 0
    for index, line in enumerate(lines):
        if set(line.strip()) <= set("|-") and "-" in line:
            header_end = index + 1
            break

    if header_end == 0:  # no markdown header found; fall back to a plain character split
        return [text[i : i + limit] for i in range(0, len(text), limit)]

    header = "\n".join(lines[:header_end])
    body = lines[header_end:]

    parts: list[str] = []
    current: list[str] = []
    for row in body:
        candidate = len(header) + sum(len(r) + 1 for r in current) + len(row)
        if current and candidate > limit:
            parts.append(header + "\n" + "\n".join(current))
            current = []
        current.append(row)
    if current:
        parts.append(header + "\n" + "\n".join(current))
    return parts


def _overlap_tail(elements: list[PageElement]) -> list[PageElement]:
    """Trailing elements to repeat at the start of the next chunk.

    Whole elements are carried over rather than a character slice, so a chunk never begins
    mid-sentence.
    """
    tail: list[PageElement] = []
    total = 0
    for element in reversed(elements):
        if total + len(element.text) > OVERLAP_CHARS:
            break
        tail.insert(0, element)
        total += len(element.text)
    return tail


def chunk_document(metadata: FilingMetadata, elements: list[PageElement]) -> list[Chunk]:
    """Group a document's elements into chunks carrying filing, page and section metadata."""
    base = metadata.as_chunk_metadata()
    chunks: list[Chunk] = []

    section, section_title = UNKNOWN_SECTION, UNKNOWN_SECTION
    buffer: list[PageElement] = []
    toc_pages = _toc_pages(elements)
    signature_page = _signature_page(elements, toc_pages)

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        text = "\n".join(e.text for e in buffer).strip()
        if text:
            chunks.append(_build(text, buffer[0].page, ElementKind.PROSE))
        buffer = _overlap_tail(buffer)

    def _build(text: str, page: int, kind: ElementKind) -> Chunk:
        return Chunk(
            chunk_id=_chunk_id(metadata.source, text),
            text=text,
            metadata={
                **base,
                "page": page,
                "section": section,
                "section_title": section_title,
                "element_kind": str(kind),
            },
        )

    for element in elements:
        crossed_into_exhibits = (
            signature_page is not None
            and element.page >= signature_page
            and section != EXHIBITS_SECTION
        )
        if crossed_into_exhibits:
            # Flush first, exactly as for a heading change. A buffer that began in the
            # filing body would otherwise be stamped with the section current at flush
            # time and be labelled an exhibit.
            flush()
            buffer = []
            section, section_title = EXHIBITS_SECTION, EXHIBITS_TITLE
        on_toc = element.page in toc_pages
        heading = None
        if not on_toc:
            heading = (
                detect_section_in_table(element.text)
                if element.is_table
                else detect_section(element.text)
            )

        if heading and not element.is_table:
            # Flush before switching, so no chunk straddles two sections and carries the
            # wrong label for half its content.
            flush()
            buffer = []
            section, section_title = heading
            continue

        if element.is_table:
            flush()
            buffer = []
            if heading:
                # The heading was absorbed into this table's caption. Apply it, then still
                # index the table — it is the section's own index or opening exhibit.
                section, section_title = heading
            for part in (
                _split_table(element.text, MAX_TABLE_CHARS)
                if len(element.text) > MAX_TABLE_CHARS
                else [element.text]
            ):
                chunks.append(_build(part, element.page, ElementKind.TABLE))
            continue

        buffer.append(element)
        if sum(len(e.text) for e in buffer) >= TARGET_CHARS:
            flush()

    flush()
    return _deduplicate(chunks)


def _deduplicate(chunks: list[Chunk]) -> list[Chunk]:
    """Drop chunks whose id has already been seen.

    Overlap can reproduce an identical chunk when a section contains a single short
    element, and repeated boilerplate can collide. Duplicates waste index space and crowd
    the top-k with the same text.
    """
    seen: set[str] = set()
    unique: list[Chunk] = []
    for chunk in chunks:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        unique.append(chunk)
    return unique
