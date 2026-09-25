#!/usr/bin/env python
"""Check golden-set drafts against their source text with a frontier model.

    python scripts/auto_verify_golden.py --dry-run     # report, change nothing
    python scripts/auto_verify_golden.py               # apply verdicts
    python scripts/auto_verify_golden.py --model openai:gpt-5.4

**This is not human verification, and the two are not interchangeable.** The result is
recorded as `model-verified`, never `human-verified`, so the distinction survives into any
claim made about the set. What it is worth:

- the drafts were written by a local 8B model; the checker is a frontier model from a
  different family, so this is not a model grading its own output
- the checker sees the source chunk and judges the answer against it, rather than judging
  plausibility
- deterministic checks run first and need no model at all

What it is not worth: a model can agree with a wrong answer for the same reason another
model produced it. Anything the checker is unsure about is marked `needs-review` rather
than accepted, so a short human pass can cover the genuinely doubtful cases.

The most common real defect in the drafts is not a wrong answer — it is answer_tokens
carrying figures from a period the question did not ask about. Relevance matches on *any*
token, so one stray figure makes a wrong-year chunk score as a hit.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field

from finrag.config import Settings
from finrag.evaluation.golden import (
    GoldenQuestion,
    QuestionType,
    Verification,
    load_golden,
    normalise_number,
    save_golden,
)
from finrag.ingestion.pipeline import open_vector_store
from finrag.llm import build_chat_model
from finrag.retrieval import Filters, Strategy, retrieve

BOLD, DIM, OFF = "\033[1m", "\033[2m", "\033[0m"
GREEN, RED, YELLOW = "\033[32m", "\033[31m", "\033[33m"

Verdict = Literal["accept", "fix", "reject", "needs-review"]


class Judgement(BaseModel):
    """The checker's assessment of one answerable question."""

    answerable_from_source: bool = Field(
        description="True only if the SOURCE text alone contains everything the question asks for"
    )
    answer_is_correct: bool = Field(
        description="True if the proposed answer matches what the source says"
    )
    tokens_are_minimal: bool = Field(
        description="True if answer_tokens contains exactly the figures a correct answer "
        "must include — no figures from other periods or unrelated line items"
    )
    corrected_tokens: list[str] = Field(
        default_factory=list,
        description="If tokens are not minimal, the corrected list, copied literally from "
        "the source. Empty if no correction is needed.",
    )
    confident: bool = Field(description="False if this needs a human to look at it for any reason")
    reason: str = Field(description="One short sentence explaining the judgement")


class AbsenceJudgement(BaseModel):
    """The checker's assessment of an unanswerable question."""

    genuinely_unanswerable: bool = Field(
        description="True if none of the retrieved excerpts answers the question"
    )
    confident: bool = Field(description="False if a human should look at this")
    reason: str = Field(description="One short sentence explaining the judgement")


ANSWERABLE_PROMPT = """You are auditing an evaluation question written by a weaker model.
Judge it strictly against the SOURCE text. Do not use outside knowledge.

Be particularly strict about answer_tokens. They are used to decide whether a retrieved
passage is relevant, by matching ANY one of them. A token belonging to a different fiscal
period, or to a line item the question did not ask about, therefore makes an incorrect
passage count as correct. Tokens must be copied literally from the source.

QUESTION: {question}
PROPOSED ANSWER: {answer}
ANSWER TOKENS: {tokens}

SOURCE:
{source}"""

ABSENCE_PROMPT = """You are auditing an evaluation question that claims to be UNANSWERABLE
from a corpus of SEC filings.

Below are the passages the search system returns for it. Decide whether any of them
actually answers the question. Do not use outside knowledge — judge only these passages.

The stated reason it should be unanswerable: {reason}

QUESTION: {question}

RETRIEVED PASSAGES:
{passages}"""


def source_text(store, question: GoldenQuestion) -> str:
    """The chunks the question was written from."""
    parts: list[str] = []
    for reference in question.expected_pages:
        result = store.get(
            where={"$and": [{"source": reference["source"]}, {"page": reference["page"]}]},
            include=["documents"],
        )
        parts.extend(result["documents"])
    return "\n\n---\n\n".join(parts)[:6000]


def judge_answerable(
    model, question: GoldenQuestion, source: str
) -> tuple[Verdict, str, list[str]]:
    if not source:
        return "needs-review", "source chunk not found in the index", []

    result: Judgement = model.invoke(
        ANSWERABLE_PROMPT.format(
            question=question.question,
            answer=question.answer,
            tokens=question.answer_tokens,
            source=source,
        )
    )

    if not result.answerable_from_source or not result.answer_is_correct:
        return "reject", result.reason, []
    if not result.confident:
        return "needs-review", result.reason, []
    if not result.tokens_are_minimal and result.corrected_tokens:
        # Only accept a correction whose tokens actually appear in the source; otherwise
        # the checker has invented figures, which is the failure it exists to catch.
        haystack = normalise_number(source)
        if all(normalise_number(t) in haystack for t in result.corrected_tokens):
            return "fix", result.reason, result.corrected_tokens
        return "needs-review", "proposed token correction is not present in the source", []
    if not result.tokens_are_minimal:
        return "needs-review", result.reason, []
    return "accept", result.reason, []


def judge_absence(model, question: GoldenQuestion) -> tuple[Verdict, str]:
    hits = retrieve(question.question, Strategy.HYBRID, Filters(), k=6)
    passages = "\n\n---\n\n".join(h.text[:900] for h in hits)

    result: AbsenceJudgement = model.invoke(
        ABSENCE_PROMPT.format(
            reason=question.notes or "not stated",
            question=question.question,
            passages=passages,
        )
    )
    if not result.genuinely_unanswerable:
        return "reject", result.reason
    if not result.confident:
        return "needs-review", result.reason
    return "accept", result.reason


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="override the checker model")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    settings = Settings(local_only=False)
    spec = args.model or settings.judge_model
    print(f"{DIM}checking with {spec}{OFF}")

    base = build_chat_model(spec, settings)
    answerable_model = base.with_structured_output(Judgement)
    absence_model = base.with_structured_output(AbsenceJudgement)

    questions = load_golden(verified_only=False)
    store = open_vector_store()

    verdicts: Counter[str] = Counter()
    kept: list[GoldenQuestion] = []
    review: list[tuple[str, str]] = []

    for index, question in enumerate(questions, start=1):
        try:
            if question.question_type is QuestionType.UNANSWERABLE:
                verdict, reason = judge_absence(absence_model, question)
                tokens: list[str] = []
            else:
                verdict, reason, tokens = judge_answerable(
                    answerable_model, question, source_text(store, question)
                )
        except Exception as exc:  # noqa: BLE001 - one failure must not lose the run
            verdict, reason, tokens = "needs-review", f"{type(exc).__name__}: {exc}", []

        verdicts[verdict] += 1
        colour = {"accept": GREEN, "fix": YELLOW, "reject": RED, "needs-review": YELLOW}[verdict]
        print(f"  {index:>3}. {colour}{verdict:<13}{OFF} {question.id}  {reason[:74]}")

        if verdict == "reject":
            continue
        if verdict == "fix":
            question.answer_tokens = tokens
        if verdict in ("accept", "fix"):
            question.verified = True
            question.verified_by = Verification.MODEL
            question.notes = (question.notes + f" | auto: {reason}").strip(" |")
        else:
            review.append((question.id, reason))
        kept.append(question)

    print(f"\n{BOLD}{dict(verdicts)}{OFF}")
    verified = sum(q.verified for q in kept)
    print(f"{verified}/{len(kept)} usable, {len(review)} need a human look")

    for question_id, reason in review:
        print(f"  {YELLOW}{question_id}{OFF} {reason[:88]}")

    if args.dry_run:
        print(f"\n{DIM}dry run — nothing written{OFF}")
        return 0

    path = save_golden(kept, allow_verified_loss=True)
    print(f"\nwritten to {path}")
    if review:
        print(f"{DIM}review the flagged ones: python scripts/verify_golden.py{OFF}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
