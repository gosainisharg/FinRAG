"""Run the golden set through the agentic graph and record what it produced.

    python -m finrag.evaluation.run_graph_eval                 # hosted models
    python -m finrag.evaluation.run_graph_eval --local         # free, for a smoke run
    python -m finrag.evaluation.run_graph_eval --limit 5

Execution is separated from scoring on purpose. Running the graph costs money and takes
minutes; scoring the result is iterated on repeatedly while thresholds and rubrics are
tuned. Writing the runs to disk means a scoring change never re-pays for generation.

The record keeps the retrieval attempts, the grader's verdict and the node path alongside
the answer, because the useful question about a bad score is which stage caused it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from finrag.config import get_settings
from finrag.evaluation.golden import GoldenQuestion, load_golden
from finrag.llm import describe_configuration
from finrag.prompts.registry import describe_prompts

RUNS_DIR = Path(__file__).resolve().parents[3] / "eval" / "runs"

logger = logging.getLogger(__name__)


@dataclass
class GraphRun:
    """What the graph did for one question."""

    question_id: str
    question: str
    question_type: str
    expected_answer: str | None
    answer_tokens: list[str]

    answer: str
    answerable: bool
    citations: list[str]
    contexts: list[str]
    context_labels: list[str]

    attempts: int
    graded_sufficient: bool | None
    grade_missing: str
    filters_used: str
    path: list[str]
    errors: list[str]
    seconds: float


@dataclass
class RunManifest:
    """Everything needed to say what produced these numbers."""

    created_at: str
    models: dict[str, str]
    prompts: dict[str, str]
    local_only: bool
    runs: list[GraphRun] = field(default_factory=list)

    def to_json(self) -> str:
        payload = asdict(self)
        return json.dumps(payload, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> RunManifest:
        payload = json.loads(path.read_text())
        runs = [GraphRun(**r) for r in payload.pop("runs")]
        return cls(**payload, runs=runs)


def execute(question: GoldenQuestion) -> GraphRun:
    from finrag.graph.build import answer as run_graph

    started = time.perf_counter()
    state = run_graph(question.question, thread_id=question.id)
    chunks = state.get("chunks", [])
    grade = state.get("grade")

    return GraphRun(
        question_id=question.id,
        question=question.question,
        question_type=str(question.question_type),
        expected_answer=question.answer,
        answer_tokens=question.answer_tokens,
        answer=state.get("answer", ""),
        answerable=bool(state.get("answerable")),
        citations=state.get("citations", []),
        contexts=[c.text for c in chunks],
        context_labels=[c.citation for c in chunks],
        attempts=state.get("attempts", 0),
        graded_sufficient=grade.sufficient if grade else None,
        grade_missing=grade.missing if grade else "",
        filters_used=state.get("filters_used", ""),
        path=state.get("path", []),
        errors=state.get("errors", []),
        seconds=time.perf_counter() - started,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", action="store_true", help="run on the local model, free")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip questions already recorded in the output file",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.local:
        import os

        os.environ["FINRAG_LOCAL_ONLY"] = "true"
        get_settings.cache_clear()

    questions = load_golden()
    if not questions:
        print("no verified golden questions", file=sys.stderr)
        return 1

    out = args.out or RUNS_DIR / ("graph_runs_local.json" if args.local else "graph_runs.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.resume and out.exists():
        manifest = RunManifest.load(out)
        done = {r.question_id for r in manifest.runs}
        questions = [q for q in questions if q.id not in done]
        print(f"resuming: {len(done)} already recorded, {len(questions)} to go", file=sys.stderr)
    else:
        manifest = RunManifest(
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            models=describe_configuration(),
            prompts=describe_prompts(),
            local_only=get_settings().local_only,
        )

    # Applied after the resume filter, so --limit means "process this many more" and
    # repeated batches make progress instead of re-selecting the same head of the list.
    if args.limit:
        questions = questions[: args.limit]

    print(f"models: {manifest.models}", file=sys.stderr)
    print(f"prompts: {manifest.prompts}", file=sys.stderr)

    for index, question in enumerate(questions, start=1):
        run = execute(question)
        manifest.runs.append(run)
        # Written after every question: a run that is interrupted — and in this project
        # long jobs have been — resumes instead of starting over and paying again.
        out.write_text(manifest.to_json())

        flag = "ok " if run.answerable else "no "
        print(
            f"  {index:>3}/{len(questions)} {flag} {run.question_id} "
            f"att={run.attempts} {run.seconds:5.1f}s  {run.answer[:58]}",
            file=sys.stderr,
        )

    answered = sum(r.answerable for r in manifest.runs)
    total_time = sum(r.seconds for r in manifest.runs)
    print(
        f"\n{answered}/{len(manifest.runs)} answered, {total_time / len(manifest.runs):.1f}s mean"
        f"\nwritten to {out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
