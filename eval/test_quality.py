"""Quality gates.

Two tiers, because they cost very different things:

- The **retrieval** gate is deterministic and free. It runs on every pull request.
- The **answer** gate reads recorded judge scores rather than calling a judge, so it is
  also free to run; producing those scores costs money and happens on a schedule.

Thresholds are set below the measured values with deliberate headroom. A gate that sits
exactly at the current number fails on noise and gets disabled, which is worse than having
no gate at all.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from finrag.evaluation.golden import QuestionType, load_golden
from finrag.evaluation.retrieval_metrics import aggregate, score_question
from finrag.retrieval import Filters, Strategy, retrieve

RESULTS = Path(__file__).resolve().parent / "results"
RUNS = Path(__file__).resolve().parent / "runs" / "graph_runs.json"

# Measured 2026-09-25 with hybrid retrieval at k=5: hit-rate 0.970, MRR 0.755,
# recall 0.939. Gates sit clear of those so ordinary variation does not trip them.
MIN_HIT_RATE = 0.85
MIN_MRR = 0.65
MIN_RECALL = 0.80

# Measured: faithfulness 0.921, financial rigour 0.603.
MIN_FAITHFULNESS = 0.80
MIN_FINANCIAL_RIGOUR = 0.45


def _usable(value) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


@pytest.fixture(scope="module")
def golden():
    questions = load_golden()
    if not questions:
        pytest.skip("no verified golden set")
    return questions


class TestRetrievalGate:
    """Free and deterministic — runs on every pull request."""

    @pytest.fixture(scope="class")
    def metrics(self):
        questions = load_golden()
        if not questions:
            pytest.skip("no verified golden set")
        answerable = [q for q in questions if q.is_answerable]
        results = [
            score_question(q, retrieve(q.question, Strategy.HYBRID, Filters(), k=5))
            for q in answerable
        ]
        return aggregate(results)

    def test_hit_rate(self, metrics):
        assert metrics.hit_rate >= MIN_HIT_RATE, f"hit-rate {metrics.hit_rate:.3f}"

    def test_mrr(self, metrics):
        assert metrics.mrr >= MIN_MRR, f"MRR {metrics.mrr:.3f}"

    def test_recall(self, metrics):
        assert metrics.recall >= MIN_RECALL, f"recall {metrics.recall:.3f}"


class TestGoldenSetHealth:
    """An eval set that drifts silently is worse than none."""

    def test_covers_every_question_type(self, golden):
        present = {q.question_type for q in golden}
        assert QuestionType.UNANSWERABLE in present, (
            "without unanswerable questions, a system that fabricates confidently scores "
            "the same as one that correctly declines"
        )
        assert len(present) >= 3

    def test_every_question_is_verified(self, golden):
        assert all(q.verified for q in golden)

    def test_answerable_questions_carry_tokens(self, golden):
        for question in golden:
            if question.is_answerable:
                assert question.answer_tokens, f"{question.id} has no answer tokens"

    def test_unanswerable_questions_carry_none(self, golden):
        for question in golden:
            if not question.is_answerable:
                assert not question.answer_tokens, f"{question.id} should have no tokens"


@pytest.mark.llm
class TestAnswerGate:
    """Reads recorded scores. Producing them costs money; checking them does not."""

    @pytest.fixture(scope="class")
    def scores(self):
        rows: dict[str, dict] = {}
        for name in ("ragas_scores.json", "deepeval_scores.json"):
            path = RESULTS / name
            if path.exists():
                for row in json.loads(path.read_text())["scores"]:
                    # Filter None on merge: each file carries the full schema with the
                    # metrics it did not compute set to null, so a plain update would
                    # let the second file erase the first file's scores.
                    rows.setdefault(row["question_id"], {}).update(
                        {k: v for k, v in row.items() if v is not None}
                    )
        if not rows:
            pytest.skip("no recorded judge scores; run finrag.evaluation.answer_metrics")
        return rows

    def test_faithfulness(self, scores):
        values = [s["faithfulness"] for s in scores.values() if _usable(s.get("faithfulness"))]
        assert values, "faithfulness was never scored"
        mean = sum(values) / len(values)
        assert mean >= MIN_FAITHFULNESS, f"faithfulness {mean:.3f}"

    def test_financial_rigour(self, scores):
        values = [
            s["financial_rigour"] for s in scores.values() if _usable(s.get("financial_rigour"))
        ]
        assert values, "financial rigour was never scored"
        mean = sum(values) / len(values)
        assert mean >= MIN_FINANCIAL_RIGOUR, f"financial rigour {mean:.3f}"


@pytest.mark.llm
class TestRefusalGate:
    """The safety property: never fabricate an answer to an unanswerable question."""

    def test_every_unanswerable_question_is_refused(self):
        if not RUNS.exists():
            pytest.skip("no recorded graph runs")
        runs = json.loads(RUNS.read_text())["runs"]
        unanswerable = [r for r in runs if r["question_type"] == "unanswerable"]
        if not unanswerable:
            pytest.skip("no unanswerable questions in the recorded run")
        fabricated = [r["question_id"] for r in unanswerable if r["answerable"]]
        assert not fabricated, f"fabricated answers for {fabricated}"
