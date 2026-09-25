"""Score every retrieval strategy against the golden set.

    python -m finrag.evaluation.run_retrieval_eval
    python -m finrag.evaluation.run_retrieval_eval --k 10 --provisional

Deterministic and free: no LLM is called, so this can run on every pull request. It is
also the *baseline* — these numbers are taken before any agentic machinery exists, so the
Phase 5 loop has something to be measured against. Without that, "the agentic loop helps"
is an assertion rather than a finding.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from finrag.evaluation.golden import GoldenQuestion, QuestionType, load_golden
from finrag.evaluation.retrieval_metrics import (
    QuestionResult,
    RetrievalMetrics,
    abstention_rate,
    aggregate,
    score_question,
)
from finrag.retrieval import Filters, Strategy, retrieve
from finrag.retrieval.types import RetrievedChunk

RESULTS_DIR = Path(__file__).resolve().parents[3] / "eval" / "results"

logger = logging.getLogger(__name__)


def run_strategy(
    questions: list[GoldenQuestion], strategy: Strategy, k: int
) -> tuple[list[QuestionResult], dict[str, list[RetrievedChunk]], float]:
    """Retrieve for every question under one strategy."""
    results: list[QuestionResult] = []
    retrieved: dict[str, list[RetrievedChunk]] = {}

    started = time.perf_counter()
    for question in questions:
        # No metadata filters: this measures retrieval on the raw question, which is what
        # the system faces before the Phase 5 router exists to narrow it.
        hits = retrieve(question.question, strategy, Filters(), k=k)
        retrieved[question.id] = hits
        if question.is_answerable:
            results.append(score_question(question, hits))

    elapsed = (time.perf_counter() - started) / max(len(questions), 1)
    return results, retrieved, elapsed * 1000


def render_report(
    rows: dict[str, RetrievalMetrics],
    by_type: dict[str, dict[str, RetrievalMetrics]],
    abstention: dict[str, float],
    latency: dict[str, float],
    questions: list[GoldenQuestion],
    k: int,
    provisional: bool,
) -> str:
    counts = {t.value: sum(1 for q in questions if q.question_type is t) for t in QuestionType}
    lines = [
        "# Retrieval baseline",
        "",
        f"_Generated {datetime.now(UTC):%Y-%m-%d %H:%M UTC} · k={k} · {len(questions)} questions_",
        "",
    ]
    if provisional:
        lines += [
            "> **Provisional.** Includes unverified questions, so these numbers measure "
            "agreement with an LLM's drafts rather than correctness. Not for the README.",
            "",
        ]

    lines += [
        "Question mix: " + " · ".join(f"{name} {count}" for name, count in counts.items() if count),
        "",
        "## Overall (answerable questions)",
        "",
        "| strategy | hit-rate@k | MRR | precision | recall | n |",
        "|---|---|---|---|---|---|",
    ]
    lines += [metrics.as_row(name) for name, metrics in rows.items()]

    lines += [
        "",
        "## Mean latency per query",
        "",
        "| strategy | ms |",
        "|---|---|",
    ]
    lines += [f"| {name} | {value:.0f} |" for name, value in latency.items()]

    if any(abstention.values()) or counts.get(QuestionType.UNANSWERABLE.value):
        lines += [
            "",
            "## Unanswerable questions",
            "",
            "Fraction where retrieval surfaced no chunk containing a plausible answer. "
            "Retrieval cannot refuse — that is the generator's job — but a low score here "
            "means the grader will be handed confident-looking noise.",
            "",
            "| strategy | clean |",
            "|---|---|",
        ]
        lines += [f"| {name} | {value:.3f} |" for name, value in abstention.items()]

    for question_type, per_strategy in by_type.items():
        lines += [
            "",
            f"## {question_type}",
            "",
            "| strategy | hit-rate@k | MRR | precision | recall | n |",
            "|---|---|---|---|---|---|",
        ]
        lines += [metrics.as_row(name) for name, metrics in per_strategy.items()]

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--provisional",
        action="store_true",
        help="include unverified questions (marks the report accordingly)",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    questions = load_golden(verified_only=not args.provisional)
    if not questions:
        print(
            "no verified golden questions. run scripts/generate_golden.py, "
            "scripts/seed_golden.py, then scripts/verify_golden.py — "
            "or pass --provisional to score the drafts.",
            file=sys.stderr,
        )
        return 1

    rows: dict[str, RetrievalMetrics] = {}
    latency: dict[str, float] = {}
    abstention: dict[str, float] = {}
    per_type: dict[str, dict[str, RetrievalMetrics]] = {}

    for strategy in Strategy:
        results, retrieved, elapsed = run_strategy(questions, strategy, args.k)
        rows[strategy.value] = aggregate(results)
        latency[strategy.value] = elapsed
        abstention[strategy.value] = abstention_rate(questions, retrieved)

        for question_type in QuestionType:
            if question_type is QuestionType.UNANSWERABLE:
                continue
            subset = [r for r in results if r.question_type == question_type.value]
            if subset:
                per_type.setdefault(question_type.value, {})[strategy.value] = aggregate(subset)

        print(f"  {strategy.value:<16} {rows[strategy.value].as_row('')[2:]}", file=sys.stderr)

    report = render_report(rows, per_type, abstention, latency, questions, args.k, args.provisional)
    print()
    print(report)

    out = args.out or RESULTS_DIR / (
        "retrieval_baseline_provisional.md" if args.provisional else "retrieval_baseline.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"written to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
