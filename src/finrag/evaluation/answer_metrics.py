"""LLM-as-judge scoring of recorded graph runs.

    python -m finrag.evaluation.answer_metrics
    python -m finrag.evaluation.answer_metrics --limit 5 --skip-ragas

Scores answers, not retrieval — the retrieval metrics in `retrieval_metrics` are
deterministic and free and gate every pull request; these cost money and run on a
schedule.

Two frameworks, deliberately:

- **RAGAS** supplies the standard quartet — faithfulness, response relevancy, context
  precision and recall — so the numbers are comparable to how everyone else reports RAG
  quality.
- **DeepEval G-Eval** supplies a rubric this domain needs and no general metric covers:
  whether a figure carries its units and period, and whether it is cited. "$96,773" and
  "$96,773 million" score identically on faithfulness and differ by six orders of
  magnitude.

The judge is configured to a different model family than the generator. A model scoring
its own output rates it more highly than a neutral judge does.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from finrag.evaluation import compat  # noqa: F401  # must precede any ragas import
from finrag.evaluation.run_graph_eval import RUNS_DIR, GraphRun, RunManifest

RESULTS_DIR = Path(__file__).resolve().parents[3] / "eval" / "results"

logger = logging.getLogger(__name__)

FINANCIAL_RUBRIC = """Evaluate whether the ANSWER states its figures usably and honestly.

Award credit for:
- every monetary figure carrying its scale (million/billion) and its fiscal period
- figures matching the expected answer exactly, where one is given
- at least one citation when a figure is stated

Penalise heavily:
- a figure with no units — "$96,773" instead of "$96,773 million" is wrong by six orders
  of magnitude, and reads as correct
- a figure attributed to the wrong period
- a stated figure with no citation
Do not penalise a refusal that is correct."""


@dataclass
class AnswerScores:
    question_id: str
    question_type: str
    faithfulness: float | None = None
    response_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    financial_rigour: float | None = None


def _judge_llm():
    """A LangChain judge wrapped for RAGAS."""
    from ragas.llms import LangchainLLMWrapper

    from finrag.config import ModelRole, get_settings
    from finrag.llm import build_chat_model

    settings = get_settings()
    return LangchainLLMWrapper(build_chat_model(settings.model_for(ModelRole.JUDGE), settings))


def _judge_embeddings():
    from ragas.embeddings import LangchainEmbeddingsWrapper

    from finrag.ingestion.embeddings import get_embeddings

    return LangchainEmbeddingsWrapper(get_embeddings())


def score_with_ragas(runs: list[GraphRun]) -> dict[str, dict]:
    """Faithfulness, relevancy, context precision and recall."""
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    samples, ids = [], []
    for run in runs:
        if not run.contexts:
            continue
        samples.append(
            SingleTurnSample(
                user_input=run.question,
                response=run.answer,
                retrieved_contexts=run.contexts,
                reference=run.expected_answer or "Not answerable from these filings.",
            )
        )
        ids.append(run.question_id)

    if not samples:
        return {}

    llm = _judge_llm()
    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[
            Faithfulness(llm=llm),
            ResponseRelevancy(llm=llm, embeddings=_judge_embeddings()),
            LLMContextPrecisionWithReference(llm=llm),
            LLMContextRecall(llm=llm),
        ],
        show_progress=True,
    )

    frame = result.to_pandas()
    return {question_id: frame.iloc[index].to_dict() for index, question_id in enumerate(ids)}


def score_with_deepeval(runs: list[GraphRun]) -> dict[str, float]:
    """The finance-specific rubric: units, period, citation."""
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    from finrag.config import ModelRole, Provider, get_settings
    from finrag.llm import parse_model_spec

    settings = get_settings()
    provider, judge_model = parse_model_spec(settings.model_for(ModelRole.JUDGE))

    # DeepEval constructs its own client and reads the key from the process environment,
    # not from our settings object. Exporting it here keeps .env as the single place a key
    # is written while still satisfying a library that insists on os.environ.
    if provider is Provider.OPENAI and settings.openai_api_key:
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key.get_secret_value())
    elif provider is Provider.ANTHROPIC and settings.anthropic_api_key:
        os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key.get_secret_value())

    metric = GEval(
        name="FinancialRigour",
        criteria=FINANCIAL_RUBRIC,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=judge_model,
    )

    scores: dict[str, float] = {}
    for run in runs:
        case = LLMTestCase(
            input=run.question,
            actual_output=run.answer,
            expected_output=run.expected_answer or "Not answerable from these filings.",
        )
        try:
            metric.measure(case)
            scores[run.question_id] = float(metric.score)
        except Exception as exc:  # noqa: BLE001 - one judge failure must not lose the run
            logger.warning("G-Eval failed for %s: %s", run.question_id, exc)
    return scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=RUNS_DIR / "graph_runs.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--skip-deepeval", action="store_true")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "answer_scores.json")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if not args.runs.exists():
        print(f"no runs at {args.runs}; run run_graph_eval first", file=sys.stderr)
        return 1

    manifest = RunManifest.load(args.runs)
    runs = manifest.runs[: args.limit] if args.limit else manifest.runs
    print(f"scoring {len(runs)} runs", file=sys.stderr)

    ragas_scores = {} if args.skip_ragas else score_with_ragas(runs)
    deepeval_scores = {} if args.skip_deepeval else score_with_deepeval(runs)

    scored = []
    for run in runs:
        row = ragas_scores.get(run.question_id, {})
        scored.append(
            AnswerScores(
                question_id=run.question_id,
                question_type=run.question_type,
                faithfulness=row.get("faithfulness"),
                response_relevancy=row.get("answer_relevancy") or row.get("response_relevancy"),
                context_precision=row.get("llm_context_precision_with_reference"),
                context_recall=row.get("context_recall"),
                financial_rigour=deepeval_scores.get(run.question_id),
            )
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "models": manifest.models,
                "prompts": manifest.prompts,
                "scores": [s.__dict__ for s in scored],
            },
            indent=2,
        )
    )
    print(f"written to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
