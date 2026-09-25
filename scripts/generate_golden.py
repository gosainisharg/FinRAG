#!/usr/bin/env python
"""Draft golden-set candidates from indexed chunks, using the local model.

    python scripts/generate_golden.py --n 40
    python scripts/generate_golden.py --n 10 --type multi_hop --append

Runs on the locally-served model, so generating costs nothing and can be repeated freely.

Every candidate is written with `verified: false`. They are **drafts**, not ground truth —
`load_golden()` ignores unverified entries by design. An LLM writing both the question and
the answer, then being scored against its own answer, measures nothing. The verification
pass in `verify_golden.py` is what makes the numbers real.

One automated filter runs first: a candidate whose answer tokens do not literally appear in
its source chunk is discarded. That catches the common failure where the model paraphrases
a figure or invents one, and it removes most of the manual work before a human looks.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys

from pydantic import BaseModel, Field

from finrag.config import Settings
from finrag.evaluation.golden import (
    GoldenQuestion,
    QuestionType,
    load_golden,
    normalise_number,
    save_golden,
)
from finrag.ingestion.pipeline import open_vector_store
from finrag.llm import build_chat_model

BOLD, DIM, OFF = "\033[1m", "\033[2m", "\033[0m"

# Sections where the answerable facts live. Exhibits are excluded — a question about a
# stock plan's vesting schedule is not what this system is for.
USEFUL_SECTIONS = ("Item 1", "Item 1A", "Item 5", "Item 6", "Item 7", "Item 7A", "Item 8")


class DraftQA(BaseModel):
    """What the model is asked to produce for one chunk."""

    question: str = Field(description="A specific question answerable only from this text")
    answer: str = Field(description="The exact answer, quoting figures as they appear")
    answer_tokens: list[str] = Field(
        description="Literal strings copied from the text that an answer must contain, "
        "such as '96,773'. Copy them exactly, do not reformat."
    )


SINGLE_FACT_PROMPT = """You write evaluation questions for a financial-filings search system.

Below is one excerpt from {company}'s {form} for fiscal {year}.

Write ONE question that:
- can be answered using only this excerpt
- asks for a specific fact, figure or named item — not a summary
- names the company and the period, so it is unambiguous on its own
- a financial analyst would plausibly ask

Then give the exact answer, and copy the literal strings from the excerpt that any correct
answer must contain. Copy figures exactly as written, including commas.

EXCERPT:
{text}"""

MULTI_HOP_PROMPT = """You write evaluation questions for a financial-filings search system.

Below are TWO excerpts from {company}'s {form} for fiscal {year}.

Write ONE question that requires BOTH excerpts to answer — a comparison, a sum, or a
change between two figures. It must be impossible to answer from either excerpt alone.

Name the company and the period. Give the exact answer, and copy the literal strings from
the excerpts that any correct answer must contain, exactly as written.

EXCERPT A:
{text_a}

EXCERPT B:
{text_b}"""


def sample_chunks(store, n: int, seed: int, tables_only: bool = False) -> list[dict]:
    """Draw chunks worth asking about, spread across the filings."""
    where = {"$and": [{"section": {"$in": list(USEFUL_SECTIONS)}}]}
    if tables_only:
        where["$and"].append({"element_kind": "table"})

    data = store.get(where=where, include=["documents", "metadatas"])
    pool = [
        {"text": t, "metadata": m}
        for t, m in zip(data["documents"], data["metadatas"], strict=True)
        # Very short chunks rarely contain a self-contained fact.
        if len(t) > 400
    ]

    rng = random.Random(seed)
    rng.shuffle(pool)

    # Round-robin by ticker so one large filing does not dominate the set.
    by_ticker: dict[str, list[dict]] = {}
    for item in pool:
        by_ticker.setdefault(item["metadata"].get("ticker", "?"), []).append(item)

    spread: list[dict] = []
    while len(spread) < n and any(by_ticker.values()):
        for bucket in by_ticker.values():
            if bucket and len(spread) < n:
                spread.append(bucket.pop())
    return spread


def tokens_are_grounded(tokens: list[str], *texts: str) -> bool:
    """Whether every answer token literally appears in the source text."""
    if not tokens:
        return False
    haystack = " ".join(normalise_number(t) for t in texts)
    return all(normalise_number(token) in haystack for token in tokens)


def draft(model, prompt: str) -> DraftQA | None:
    try:
        return model.invoke(prompt)
    except Exception as exc:  # noqa: BLE001 - one bad generation must not stop the run
        logging.warning("generation failed: %s: %s", type(exc).__name__, exc)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=30, help="candidates to attempt")
    parser.add_argument(
        "--type",
        default=QuestionType.SINGLE_FACT.value,
        choices=[QuestionType.SINGLE_FACT.value, QuestionType.MULTI_HOP.value],
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--append", action="store_true", help="add to the existing file")
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    # Generation is free on the local backend; force it regardless of .env.
    settings = Settings(local_only=True)
    model = build_chat_model(settings.local_model, settings).with_structured_output(DraftQA)
    print(f"{DIM}generating with {settings.local_model}{OFF}")

    store = open_vector_store()
    question_type = QuestionType(args.type)
    per_question = 2 if question_type is QuestionType.MULTI_HOP else 1
    chunks = sample_chunks(store, args.n * per_question, args.seed, args.tables_only)

    existing = load_golden(verified_only=False) if args.append else []
    start_index = len(existing)
    accepted: list[GoldenQuestion] = []
    rejected = 0

    for i in range(args.n):
        group = chunks[i * per_question : (i + 1) * per_question]
        if len(group) < per_question:
            break
        meta = group[0]["metadata"]

        if question_type is QuestionType.MULTI_HOP:
            # Both excerpts must come from the same filing or the question is not multi-hop
            # within one document; skip mismatched pairs rather than produce a bad prompt.
            if group[1]["metadata"].get("source") != meta.get("source"):
                continue
            prompt = MULTI_HOP_PROMPT.format(
                company=meta.get("company"),
                form=meta.get("form_type"),
                year=meta.get("fiscal_year"),
                text_a=group[0]["text"][:2000],
                text_b=group[1]["text"][:2000],
            )
        else:
            prompt = SINGLE_FACT_PROMPT.format(
                company=meta.get("company"),
                form=meta.get("form_type"),
                year=meta.get("fiscal_year"),
                text=group[0]["text"][:2500],
            )

        result = draft(model, prompt)
        if result is None:
            rejected += 1
            continue

        if not tokens_are_grounded(result.answer_tokens, *(g["text"] for g in group)):
            rejected += 1
            print(f"  {DIM}reject (ungrounded tokens): {result.answer_tokens}{OFF}")
            continue

        accepted.append(
            GoldenQuestion(
                id=f"q{start_index + len(accepted) + 1:03d}",
                question=result.question.strip(),
                question_type=question_type,
                answer=result.answer.strip(),
                answer_tokens=result.answer_tokens,
                tickers=[meta["ticker"]] if meta.get("ticker") else [],
                fiscal_years=[meta["fiscal_year"]] if meta.get("fiscal_year") else [],
                expected_pages=[
                    {"source": g["metadata"]["source"], "page": g["metadata"]["page"]}
                    for g in group
                ],
                verified=False,
            )
        )
        print(f"  {len(accepted):>3}. {result.question[:88]}")

    path = save_golden(existing + accepted)
    print(
        f"\n{BOLD}{len(accepted)} drafted, {rejected} auto-rejected{OFF} → {path}"
        f"\n{DIM}all unverified. run: python scripts/verify_golden.py{OFF}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
