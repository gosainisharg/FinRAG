"""The golden question set.

Ground truth for retrieval is defined by **the answer appearing in the chunk**, not by
chunk id and not by page number:

- Chunk ids are content hashes, so any change to chunking invalidates every label. A
  golden set that has to be rebuilt whenever the chunker changes cannot measure whether
  the chunker improved.
- Page numbers are close but not exact: a chunk accumulates elements until it reaches its
  size target, so it can begin on one page and continue onto the next while carrying only
  the first page in its metadata.

An answer token plus a ticker match is stable across re-chunking and is what a human would
check anyway. Pages are still recorded, for human review and for citations in the report.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

GOLDEN_PATH = Path(__file__).resolve().parents[3] / "eval" / "golden" / "finrag_golden.jsonl"


class Verification(StrEnum):
    """Who confirmed a question's ground truth.

    Recorded rather than collapsed into a boolean because the distinction is a claim
    about the evidence. A model-checked set is far stronger than an unchecked one and
    strictly weaker than a human-checked one, and the README has to say which it is.
    """

    NONE = "unverified"
    MODEL = "model-verified"
    HUMAN = "human-verified"


class QuestionType(StrEnum):
    """Deliberately mixed, because each exercises a different failure mode.

    ``UNANSWERABLE`` matters most and is the one usually left out: without it, a system
    that confidently fabricates an answer scores identically to one that correctly
    declines.
    """

    SINGLE_FACT = "single_fact"
    MULTI_HOP = "multi_hop"
    CROSS_COMPANY = "cross_company"
    UNANSWERABLE = "unanswerable"


def normalise_number(text: str) -> str:
    """Strip formatting that varies between question and document.

    "96,773", "$96,773" and "96773" are the same figure; a literal comparison would count
    a correct retrieval as a miss.
    """
    return re.sub(r"[,$\s]", "", text.lower())


@dataclass
class GoldenQuestion:
    """One evaluation question with its verified ground truth."""

    id: str
    question: str
    question_type: QuestionType
    answer: str | None
    answer_tokens: list[str] = field(default_factory=list)
    tickers: list[str] = field(default_factory=list)
    fiscal_years: list[int] = field(default_factory=list)
    expected_pages: list[dict] = field(default_factory=list)
    verified: bool = False
    verified_by: Verification = Verification.NONE
    notes: str = ""

    @property
    def is_answerable(self) -> bool:
        return self.question_type is not QuestionType.UNANSWERABLE

    def is_relevant(self, chunk_text: str, chunk_metadata: dict) -> bool:
        """Whether a retrieved chunk counts as relevant to this question.

        Requires both an answer token and a ticker match. The ticker guard matters for
        cross-company work: "96,773" could appear coincidentally in another filing, and
        without it a retrieval from the wrong company would score as a hit.
        """
        if not self.is_answerable or not self.answer_tokens:
            return False

        if self.tickers and chunk_metadata.get("ticker") not in self.tickers:
            return False

        haystack = normalise_number(chunk_text)
        return any(normalise_number(token) in haystack for token in self.answer_tokens)

    def to_json(self) -> str:
        payload = asdict(self)
        payload["question_type"] = str(self.question_type)
        payload["verified_by"] = str(self.verified_by)
        return json.dumps(payload, ensure_ascii=False)

    @classmethod
    def from_dict(cls, payload: dict) -> GoldenQuestion:
        payload = dict(payload)
        payload["question_type"] = QuestionType(payload["question_type"])
        payload["verified_by"] = Verification(payload.get("verified_by", Verification.NONE))
        return cls(**payload)


def load_golden(path: Path | None = None, verified_only: bool = True) -> list[GoldenQuestion]:
    """Read the golden set.

    Defaults to verified questions only: an unverified, LLM-generated question is a draft,
    and scoring against drafts produces a number that looks measured but is not.
    """
    path = path or GOLDEN_PATH
    if not path.exists():
        return []

    questions = [
        GoldenQuestion.from_dict(json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    return [q for q in questions if q.verified] if verified_only else questions


class GoldenWriteError(RuntimeError):
    """Raised when a write would discard verified ground truth."""


def save_golden(
    questions: list[GoldenQuestion],
    path: Path | None = None,
    *,
    allow_verified_loss: bool = False,
) -> Path:
    """Write the golden set, refusing to silently discard verified questions.

    The guard exists because of a real incident: a long-running generation process had
    loaded the file before verification ran, and when it finally finished it wrote back
    its own stale copy — erasing 39 verified questions with no error. Whole-file writes
    from concurrent processes are the hazard, and verified ground truth is the expensive
    thing to lose.

    Callers that intentionally prune questions — the review tools, which delete rejects —
    pass ``allow_verified_loss=True``.
    """
    path = path or GOLDEN_PATH

    if not allow_verified_loss and path.exists():
        existing = {q.id for q in load_golden(path, verified_only=True)}
        surviving = {q.id for q in questions}
        lost = existing - surviving
        if lost:
            raise GoldenWriteError(
                f"refusing to write {path}: would drop {len(lost)} verified question(s) "
                f"({sorted(lost)[:5]}...). Pass allow_verified_loss=True if intended."
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(q.to_json() for q in questions) + "\n")
    return path


def iter_by_type(questions: list[GoldenQuestion]) -> Iterator[tuple[QuestionType, list]]:
    for question_type in QuestionType:
        subset = [q for q in questions if q.question_type is question_type]
        if subset:
            yield question_type, subset
