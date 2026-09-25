#!/usr/bin/env python
"""Review golden-set drafts and mark them verified.

    python scripts/verify_golden.py              # review everything unverified
    python scripts/verify_golden.py --type unanswerable
    python scripts/verify_golden.py --review-verified   # re-check accepted ones

This is the step that makes the evaluation numbers real. Everything before it is an LLM
writing questions and answers from a chunk; scoring a system against that measures
agreement with a draft, not correctness. A human confirming each answer against the source
is what the generated set cannot provide for itself.

Each question is shown with its source chunk and the answer tokens highlighted, so the
judgement is "does this figure actually answer this question", not "does this look
plausible". Progress is saved on quit, so review can be done in several sittings.
"""

from __future__ import annotations

import argparse
import re
import sys

from finrag.evaluation.golden import (
    GoldenQuestion,
    QuestionType,
    Verification,
    load_golden,
    normalise_number,
    save_golden,
)
from finrag.ingestion.pipeline import open_vector_store

BOLD, DIM, OFF = "\033[1m", "\033[2m", "\033[0m"
GREEN, RED, YELLOW, CYAN = "\033[32m", "\033[31m", "\033[33m", "\033[36m"


def fetch_source_chunks(store, question: GoldenQuestion) -> list[dict]:
    """The chunks the question was written from, by source and page."""
    chunks: list[dict] = []
    for reference in question.expected_pages:
        result = store.get(
            where={"$and": [{"source": reference["source"]}, {"page": reference["page"]}]},
            include=["documents", "metadatas"],
        )
        for text, metadata in zip(result["documents"], result["metadatas"], strict=True):
            chunks.append({"text": text, "metadata": metadata})
    return chunks


def highlight(text: str, tokens: list[str]) -> str:
    for token in tokens:
        if token.strip():
            text = re.sub(
                f"({re.escape(token.strip())})", f"{GREEN}\\1{OFF}", text, flags=re.IGNORECASE
            )
    return text


def token_report(question: GoldenQuestion, chunks: list[dict]) -> str:
    """Whether each answer token is actually present in the source chunks."""
    if not question.answer_tokens:
        return f"{DIM}no answer tokens (expected for unanswerable){OFF}"

    haystack = " ".join(normalise_number(c["text"]) for c in chunks)
    parts = []
    for token in question.answer_tokens:
        found = normalise_number(token) in haystack
        parts.append(f"{GREEN if found else RED}{'✓' if found else '✗'} {token}{OFF}")
    return "  ".join(parts)


def show(question: GoldenQuestion, chunks: list[dict], index: int, total: int) -> None:
    print("\n" + "─" * 100)
    print(f"{BOLD}[{index}/{total}] {question.id}  {question.question_type}{OFF}")
    print(f"\n{BOLD}Q:{OFF} {question.question}")
    answer = question.answer or f"{YELLOW}(none — unanswerable){OFF}"
    print(f"{BOLD}A:{OFF} {answer}")
    print(f"{BOLD}tokens:{OFF} {token_report(question, chunks)}")
    if question.notes:
        print(f"{DIM}note: {question.notes}{OFF}")

    if question.question_type is QuestionType.UNANSWERABLE:
        print(
            f"\n{DIM}No source chunk to check. Confirm the corpus genuinely cannot answer "
            f"this.{OFF}"
        )
        return

    for chunk in chunks[:2]:
        metadata = chunk["metadata"]
        print(
            f"\n{CYAN}source:{OFF} {metadata.get('ticker')} FY{metadata.get('fiscal_year')} "
            f"{metadata.get('section')} p{metadata.get('page')} [{metadata.get('element_kind')}]"
        )
        body = highlight(chunk["text"][:1100], question.answer_tokens)
        for line in body.splitlines():
            print("   " + line)
    if not chunks:
        print(f"\n{RED}source chunk not found — the index may have been rebuilt{OFF}")


def prompt_edit(question: GoldenQuestion) -> GoldenQuestion:
    print(f"{DIM}blank keeps the current value{OFF}")
    new_question = input("  question: ").strip()
    new_answer = input("  answer:   ").strip()
    new_tokens = input("  tokens (comma-separated): ").strip()

    if new_question:
        question.question = new_question
    if new_answer:
        question.answer = new_answer
    if new_tokens:
        question.answer_tokens = [t.strip() for t in new_tokens.split(",") if t.strip()]
    return question


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--type", choices=[t.value for t in QuestionType])
    parser.add_argument("--review-verified", action="store_true")
    args = parser.parse_args()

    everything = load_golden(verified_only=False)
    if not everything:
        print("no golden questions found; run scripts/generate_golden.py first")
        return 1

    queue = [q for q in everything if q.verified == args.review_verified]
    if args.type:
        queue = [q for q in queue if q.question_type is QuestionType(args.type)]

    if not queue:
        verified = sum(q.verified for q in everything)
        print(f"nothing to review. {verified}/{len(everything)} verified.")
        return 0

    store = open_vector_store()
    rejected: set[str] = set()
    print(f"{BOLD}{len(queue)} to review{OFF}   [a]ccept  [e]dit  [r]eject  [s]kip  [q]uit+save")

    for index, question in enumerate(queue, start=1):
        chunks = fetch_source_chunks(store, question)
        show(question, chunks, index, len(queue))

        while True:
            choice = input(f"\n  {BOLD}[a/e/r/s/q]{OFF} ").strip().lower() or "s"
            if choice == "a":
                question.verified = True
                question.verified_by = Verification.HUMAN
                break
            if choice == "e":
                prompt_edit(question)
                show(question, chunks, index, len(queue))
                continue
            if choice == "r":
                rejected.add(question.id)
                break
            if choice == "s":
                break
            if choice == "q":
                kept = [q for q in everything if q.id not in rejected]
                save_golden(kept)
                verified = sum(q.verified for q in kept)
                print(f"\nsaved. {verified}/{len(kept)} verified, {len(rejected)} rejected.")
                return 0
            print("  choose a, e, r, s or q")

    kept = [q for q in everything if q.id not in rejected]
    save_golden(kept)
    verified = sum(q.verified for q in kept)
    print(f"\n{BOLD}done.{OFF} {verified}/{len(kept)} verified, {len(rejected)} rejected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
