"""Metric-based retrieval scoring.

Deterministic and free — no LLM is involved — which is why these gate every pull request
while the LLM-as-judge suite runs on a schedule.

Four metrics, because they fail in different directions:

- **hit-rate@k** — did any relevant chunk make the cut? The blunt question: can the system
  answer this at all?
- **MRR** — how high was the first relevant chunk? A hit at rank 5 survives hit-rate but
  will be crowded out of a prompt that only takes the top 3.
- **precision@k** — how much of what was retrieved was relevant? Low precision means the
  generator is reading mostly noise, which is where hallucination starts.
- **recall@k** — how much of the available evidence was found? Multi-hop questions need
  several chunks; retrieving one of three looks fine to hit-rate and still cannot be
  answered.
"""

from __future__ import annotations

from dataclasses import dataclass

from finrag.evaluation.golden import GoldenQuestion
from finrag.retrieval.types import RetrievedChunk


@dataclass(frozen=True)
class QuestionResult:
    """Per-question outcome, kept so failures can be inspected individually."""

    question_id: str
    question_type: str
    hit: bool
    reciprocal_rank: float
    precision: float
    recall: float
    first_relevant_rank: int | None
    retrieved: int


@dataclass(frozen=True)
class RetrievalMetrics:
    hit_rate: float
    mrr: float
    precision: float
    recall: float
    questions: int

    def as_row(self, label: str) -> str:
        return (
            f"| {label} | {self.hit_rate:.3f} | {self.mrr:.3f} | "
            f"{self.precision:.3f} | {self.recall:.3f} | {self.questions} |"
        )


def score_question(question: GoldenQuestion, retrieved: list[RetrievedChunk]) -> QuestionResult:
    """Score one question's retrieved chunks against its ground truth."""
    flags = [question.is_relevant(c.text, c.metadata) for c in retrieved]
    first = next((i for i, ok in enumerate(flags, start=1) if ok), None)
    relevant_found = sum(flags)

    # Recall needs a denominator: how much evidence exists. The expected pages recorded at
    # verification time are that denominator, falling back to one for a single-fact answer.
    expected = max(len(question.expected_pages), 1)

    return QuestionResult(
        question_id=question.id,
        question_type=str(question.question_type),
        hit=first is not None,
        reciprocal_rank=1.0 / first if first else 0.0,
        precision=relevant_found / len(retrieved) if retrieved else 0.0,
        recall=min(relevant_found / expected, 1.0),
        first_relevant_rank=first,
        retrieved=len(retrieved),
    )


def aggregate(results: list[QuestionResult]) -> RetrievalMetrics:
    if not results:
        return RetrievalMetrics(0.0, 0.0, 0.0, 0.0, 0)
    n = len(results)
    return RetrievalMetrics(
        hit_rate=sum(r.hit for r in results) / n,
        mrr=sum(r.reciprocal_rank for r in results) / n,
        precision=sum(r.precision for r in results) / n,
        recall=sum(r.recall for r in results) / n,
        questions=n,
    )


def abstention_rate(
    questions: list[GoldenQuestion], retrieved: dict[str, list[RetrievedChunk]]
) -> float:
    """Fraction of unanswerable questions for which retrieval surfaced no strong evidence.

    Reported separately because the other four metrics are undefined when no chunk can be
    relevant. Retrieval cannot itself refuse — that is the generator's job in Phase 7 —
    but a low score here means the grader will be handed confident-looking noise.
    """
    unanswerable = [q for q in questions if not q.is_answerable]
    if not unanswerable:
        return 0.0

    clean = sum(
        1
        for q in unanswerable
        if not any(q.is_relevant(c.text, c.metadata) for c in retrieved.get(q.id, []))
    )
    return clean / len(unanswerable)
