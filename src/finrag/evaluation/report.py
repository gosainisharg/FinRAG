"""Combine the recorded runs and judge scores into one report.

    python -m finrag.evaluation.report

The headline is the before/after: the Phase 3 baseline retrieved once from the raw
question and stopped; the graph routes, grades its own context and searches again when the
grade is weak. Scoring the graph's *final* context with the same retrieval metrics is what
makes "the agentic loop helps" a measurement rather than a claim — and it is equally able
to show that it does not.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from finrag.evaluation.golden import load_golden
from finrag.evaluation.retrieval_metrics import aggregate, score_question
from finrag.evaluation.run_graph_eval import RUNS_DIR, RunManifest
from finrag.retrieval.types import RetrievedChunk

RESULTS_DIR = Path(__file__).resolve().parents[3] / "eval" / "results"


def _usable(value) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def _mean(values: list[float]) -> float | None:
    usable = [v for v in values if _usable(v)]
    return statistics.mean(usable) if usable else None


def graph_retrieval_metrics(manifest: RunManifest):
    """Score the graph's final context with the Phase 3 retrieval metrics."""
    questions = {q.id: q for q in load_golden()}
    results = []
    for run in manifest.runs:
        question = questions.get(run.question_id)
        if question is None or not question.is_answerable:
            continue
        chunks = [
            RetrievedChunk(
                chunk_id=str(index),
                text=text,
                metadata={"ticker": label.split()[0] if label else ""},
                score=0.0,
            )
            for index, (text, label) in enumerate(
                zip(run.contexts, run.context_labels, strict=False)
            )
        ]
        results.append(score_question(question, chunks))
    return aggregate(results)


def build(manifest: RunManifest, scores: dict[str, dict], baseline: str | None) -> str:
    runs = manifest.runs
    answerable = [r for r in runs if r.question_type != "unanswerable"]
    unanswerable = [r for r in runs if r.question_type == "unanswerable"]

    lines = [
        "# Agentic RAG evaluation",
        "",
        f"_Generated {datetime.now(UTC):%Y-%m-%d %H:%M UTC} · {len(runs)} questions_",
        "",
        "Ground truth is **model-verified**, not human-verified: a frontier model from a "
        "different family than the drafter checked each answer against its source chunk. "
        "Stronger than unchecked, weaker than reviewed.",
        "",
        "## Configuration",
        "",
        "| role | model |",
        "|---|---|",
    ]
    lines += [f"| {role} | `{model}` |" for role, model in manifest.models.items()]
    lines += ["", "| prompt | version |", "|---|---|"]
    lines += [f"| {name} | `{ref}` |" for name, ref in manifest.prompts.items()]

    metrics = graph_retrieval_metrics(manifest)
    lines += [
        "",
        "## Retrieval: before and after the agentic loop",
        "",
        "Both rows score the *same* metrics over the *same* questions. The baseline "
        "retrieved once from the raw question. The graph routes, grades its own context, "
        "and re-queries when the grade is weak.",
        "",
        "| | hit-rate@k | MRR | precision | recall |",
        "|---|---|---|---|---|",
    ]
    if baseline:
        lines.append(baseline)
    lines.append(
        f"| **agentic graph** | {metrics.hit_rate:.3f} | {metrics.mrr:.3f} | "
        f"{metrics.precision:.3f} | {metrics.recall:.3f} |"
    )

    answered = sum(r.answerable for r in answerable)
    refused_correctly = sum(not r.answerable for r in unanswerable)
    retried = sum(1 for r in runs if r.attempts > 0)
    lines += [
        "",
        "## Behaviour",
        "",
        f"- **{answered}/{len(answerable)}** answerable questions answered",
        f"- **{refused_correctly}/{len(unanswerable)}** unanswerable questions correctly "
        "refused — no fabricated figures",
        f"- **{retried}/{len(runs)}** runs triggered at least one corrective re-query",
        f"- mean latency **{statistics.mean(r.seconds for r in runs):.1f}s**",
    ]

    lines += [
        "",
        "## Answer quality (LLM-as-judge)",
        "",
        "Judge is a different model family than the generator, so no model scores its own output.",
        "",
        "| metric | source | scored | mean |",
        "|---|---|---|---|",
    ]
    for key, label, source in (
        ("faithfulness", "faithfulness", "RAGAS"),
        ("response_relevancy", "response relevancy", "RAGAS"),
        ("context_precision", "context precision", "RAGAS"),
        ("context_recall", "context recall", "RAGAS"),
        ("financial_rigour", "financial rigour (units · period · citation)", "DeepEval G-Eval"),
    ):
        values = [s.get(key) for s in scores.values()]
        usable = [v for v in values if _usable(v)]
        mean = _mean(values)
        note = "" if len(usable) > len(values) * 0.5 else " ⚠ too few to report"
        cell = f"{mean:.3f}{note}" if mean is not None else "—"
        lines.append(f"| {label} | {source} | {len(usable)}/{len(values)} | {cell} |")

    by_type: dict[str, list[float]] = defaultdict(list)
    for question_id, row in scores.items():
        del question_id
        if _usable(row.get("financial_rigour")):
            by_type[row["question_type"]].append(row["financial_rigour"])
    if by_type:
        lines += [
            "",
            "### Financial rigour by question type",
            "",
            "| type | n | mean |",
            "|---|---|---|",
        ]
        lines += [
            f"| {name} | {len(values)} | {statistics.mean(values):.3f} |"
            for name, values in by_type.items()
        ]

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=RUNS_DIR / "graph_runs.json")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "agentic_eval.md")
    args = parser.parse_args(argv)

    if not args.runs.exists():
        print(f"no runs at {args.runs}", file=sys.stderr)
        return 1

    manifest = RunManifest.load(args.runs)

    scores: dict[str, dict] = {}
    for path in (RESULTS_DIR / "ragas_scores.json", RESULTS_DIR / "deepeval_scores.json"):
        if path.exists():
            for row in json.loads(path.read_text())["scores"]:
                scores.setdefault(row["question_id"], {}).update(
                    {k: v for k, v in row.items() if v is not None}
                )

    baseline = None
    baseline_path = RESULTS_DIR / "retrieval_baseline.md"
    if baseline_path.exists():
        for line in baseline_path.read_text().splitlines():
            if line.startswith("| hybrid |"):
                cells = [c.strip() for c in line.strip("|").split("|")]
                baseline = (
                    f"| retrieval baseline (hybrid, no loop) | {cells[1]} | {cells[2]} | "
                    f"{cells[3]} | {cells[4]} |"
                )
                break

    report = build(manifest, scores, baseline)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(report)
    print(f"written to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
