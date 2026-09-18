"""Filing-level metadata extraction.

Every chunk carries the identity of the document it came from, so the router can filter
("only Tesla, only FY2023") and so a retrieved number can be attributed. Without this the
index is an undifferentiated pile of text and a question about Apple can be answered with
Tesla's figures.

The corpus is deliberately heterogeneous — plain 10-Ks, a 10-K wrapped inside a glossy
annual report, and earnings press releases that are not SEC filings at all — so extraction
is layered: try the structured SEC cover page, then looser text patterns, then the
filename. Each layer records how it succeeded, because "we guessed from the filename" and
"we read the SEC cover" deserve different amounts of trust.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path

import pymupdf

# How many leading pages to scan for a cover. Costco's 10-K begins on page 6 behind annual
# report front matter, so scanning only the first page or two misses it.
COVER_SCAN_PAGES = 12

# Last-resort ticker lookup. Press releases carry no SEC cover block, and inferring a
# ticker from prose is unreliable. Small and explicit beats clever and wrong.
_KNOWN_TICKERS: dict[str, str] = {
    "apple": "AAPL",
    "tesla": "TSLA",
    "meta platforms": "META",
    "meta": "META",
    "costco": "COST",
}

_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"


class FormType(StrEnum):
    TEN_K = "10-K"
    TEN_Q = "10-Q"
    EARNINGS_RELEASE = "earnings-release"
    ANNUAL_REPORT = "annual-report"
    UNKNOWN = "unknown"


class Confidence(StrEnum):
    """How the value was obtained, not how likely it is to be right.

    Surfaced so ingestion can warn on anything weaker than `COVER`, and so a wrong answer
    can be traced back to a guess rather than a misreading.
    """

    COVER = "sec-cover"
    TEXT = "text-pattern"
    FILENAME = "filename"
    NONE = "not-found"


@dataclass(frozen=True)
class FilingMetadata:
    """Identity of a source document, attached to every chunk derived from it."""

    source: str
    company: str | None
    ticker: str | None
    form_type: FormType
    fiscal_year: int | None
    period_end: date | None
    body_start_page: int
    confidence: Confidence

    def as_chunk_metadata(self) -> dict[str, str | int]:
        """Flatten for Chroma, which only stores scalar metadata values.

        Nones are dropped: Chroma rejects them, and an absent key filters more predictably
        than a sentinel string.
        """
        raw: dict[str, str | int | None] = {
            "source": self.source,
            "company": self.company,
            "ticker": self.ticker,
            "form_type": str(self.form_type),
            "fiscal_year": self.fiscal_year,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "metadata_confidence": str(self.confidence),
        }
        return {k: v for k, v in raw.items() if v is not None}


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _find_body_start(doc: pymupdf.Document) -> tuple[int, str]:
    """Return the page index where the filing proper begins, and that page's text.

    Falls back to page 0 for documents with no SEC cover, such as press releases.
    """
    for i in range(min(COVER_SCAN_PAGES, len(doc))):
        text = doc[i].get_text()
        if "SECURITIES AND EXCHANGE COMMISSION" in text.upper():
            return i, text
    return 0, doc[0].get_text() if len(doc) else ""


def _extract_company(text: str) -> str | None:
    # On an SEC cover the registrant sits on its own line directly above the
    # "(Exact name of registrant...)" parenthetical. Use that line structure rather than
    # a regex over flattened text: flattening lets the match run backwards across the line
    # boundary and swallow the preceding "Commission file number ..." line.
    marker = re.search(r"\(Exact name of (?:the )?[Rr]egistrant", text)
    if marker:
        preceding = [line.strip() for line in text[: marker.start()].splitlines()]
        for line in reversed(preceding):
            candidate = line.strip(" (,")
            # Skip the stray "(" that PDF extraction leaves on its own line.
            if len(candidate) >= 3 and re.match(r"^[A-Z]", candidate):
                return candidate

    normalised = _normalise(text)

    # Press releases lead with the company in the headline or dateline.
    for pattern in (
        r"\b([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)?),? Inc\.\s*\(Nasdaq",
        r"^([A-Z][A-Za-z ]{2,40}?) Reports",
    ):
        match = re.search(pattern, normalised)
        if match:
            return match.group(1).strip(" ,")
    return None


def _extract_ticker(text: str, company: str | None) -> str | None:
    normalised = _normalise(text)

    for pattern in (
        r"Trading\s*Symbol\(?s?\)?.{0,120}?\b([A-Z]{1,5})\b\s+(?:The\s+)?(?:Nasdaq|New York)",
        r"\b([A-Z]{1,5})\b\s+(?:The\s+)?(?:Nasdaq|New York Stock Exchange)",
        r"\((?:Nasdaq|NASDAQ|NYSE):\s*([A-Z]{1,5})\)",
    ):
        match = re.search(pattern, normalised)
        if match:
            return match.group(1)

    if company:
        lowered = company.lower()
        for name, ticker in _KNOWN_TICKERS.items():
            if name in lowered:
                return ticker
    return None


def _extract_form_type(text: str, filename: str) -> FormType:
    upper = _normalise(text).upper()
    if "FORM 10-K" in upper:
        return FormType.TEN_K
    if "FORM 10-Q" in upper:
        return FormType.TEN_Q
    if re.search(r"REPORTS? (?:FIRST|SECOND|THIRD|FOURTH|FULL)", upper) or "NEWS RELEASE" in upper:
        return FormType.EARNINGS_RELEASE
    if "ANNUAL REPORT" in upper:
        return FormType.ANNUAL_REPORT
    if "10-K" in filename.upper():
        return FormType.TEN_K
    return FormType.UNKNOWN


def _parse_date(month: str, day: str, year: str) -> date | None:
    months = {m.lower(): i for i, m in enumerate(_MONTHS.split("|"), start=1)}
    try:
        return date(int(year), months[month.lower()], int(day))
    except (KeyError, ValueError):  # pragma: no cover - guards malformed OCR
        return None


def _extract_period_end(text: str) -> date | None:
    normalised = _normalise(text)
    for pattern in (
        rf"fiscal year ended\s+({_MONTHS})\s+(\d{{1,2}}),?\s+(\d{{4}})",
        rf"(?:quarter|year) ended\s+({_MONTHS})\s+(\d{{1,2}}),?\s+(\d{{4}})",
    ):
        match = re.search(pattern, normalised, re.IGNORECASE)
        if match:
            return _parse_date(*match.groups())
    return None


def extract_filing_metadata(path: Path | str) -> FilingMetadata:
    """Read identifying metadata from a filing PDF."""
    path = Path(path)
    with pymupdf.open(path) as doc:
        body_start, cover_text = _find_body_start(doc)
        # Press releases put the company in the headline but the period a page or two in,
        # so widen the window past the cover for the looser fields.
        wide_text = " ".join(doc[i].get_text() for i in range(min(body_start + 3, len(doc))))

    company = _extract_company(cover_text) or _extract_company(wide_text)
    ticker = _extract_ticker(cover_text, company) or _extract_ticker(wide_text, company)
    form_type = _extract_form_type(cover_text, path.name)
    period_end = _extract_period_end(cover_text) or _extract_period_end(wide_text)

    if company and ticker and period_end:
        confidence = Confidence.COVER if "Exact name" in cover_text else Confidence.TEXT
    elif company or ticker:
        confidence = Confidence.TEXT
    else:
        confidence = Confidence.NONE

    return FilingMetadata(
        source=path.name,
        company=company,
        ticker=ticker,
        form_type=form_type,
        # Fiscal year is the year the period ends in. True for all four issuers here;
        # it breaks for companies whose fiscal year ends in January (retailers such as
        # Walmart label FY2024 as ending Jan 2024), so revisit if the corpus grows.
        fiscal_year=period_end.year if period_end else None,
        period_end=period_end,
        body_start_page=body_start,
        confidence=confidence,
    )
