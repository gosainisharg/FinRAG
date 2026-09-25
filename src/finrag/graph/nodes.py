"""The graph's nodes.

Five steps: route the question, retrieve, grade what came back, rewrite and retry if it is
weak, then generate. The grading step is what makes this corrective rather than a pipeline
— the system judges its own retrieval and acts on the judgement.

Every node is defensive in the same way: an LLM failure degrades to a sensible default and
records the error, rather than aborting the run. A router that fails should fall back to
searching everything, not return nothing. The one thing that must *not* degrade quietly is
the grader — see `grade_node`.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from finrag.config import ModelRole, get_settings
from finrag.graph.schemas import FinalAnswer, GradeResult, RewrittenQuery, RouteDecision
from finrag.graph.state import GraphState
from finrag.llm import get_chat_model
from finrag.prompts.registry import get_prompt
from finrag.retrieval import Filters, Strategy, retrieve
from finrag.retrieval.types import RetrievedChunk

logger = logging.getLogger(__name__)

# Shown to the router so it can only filter on companies that exist. Without it the model
# invents plausible tickers and the filter silently matches nothing.
CORPUS_COMPANIES = """- AAPL — Apple Inc. (10-K, FY2023 and FY2024)
- TSLA — Tesla, Inc. (10-K, FY2023)
- COST — Costco Wholesale Corporation (10-K, FY2023)
- META — Meta Platforms (quarterly earnings releases, Q4 2023 and Q1 2024)"""


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render chunks for a prompt, each labelled with the citation to use for it.

    The label is inlined rather than left to the model to construct, so a citation can be
    checked against the chunk it came from instead of being plausible-looking prose.
    """
    return "\n\n---\n\n".join(f"[{chunk.citation}]\n{chunk.text}" for chunk in chunks)


def _note(state: GraphState, node: str) -> list[str]:
    return [*state.get("path", []), node]


def route_node(state: GraphState) -> GraphState:
    """Decide how to search: strategy plus metadata filters."""
    question = state["question"]
    try:
        model = get_chat_model(ModelRole.ROUTER).with_structured_output(RouteDecision)
        decision = model.invoke(
            get_prompt("route").render(question=question, companies=CORPUS_COMPANIES)
        )
    except Exception as exc:  # noqa: BLE001 - a routing failure must not end the run
        logger.warning("router failed, searching unfiltered: %s", exc)
        return {
            "route": RouteDecision(query_type="single_fact", strategy=Strategy.HYBRID),
            "path": _note(state, "route"),
            "errors": [*state.get("errors", []), f"route: {exc}"],
        }

    return {"route": decision, "path": _note(state, "route")}


def _filters_from(decision: RouteDecision | None) -> Filters:
    if decision is None:
        return Filters()
    return Filters(
        tickers=tuple(t.upper() for t in decision.tickers),
        fiscal_years=tuple(decision.fiscal_years),
    )


def _relaxations(filters: Filters) -> list[tuple[str, Filters]]:
    """Progressively weaker versions of a filter set, most specific first.

    Sections are dropped before years, and years before tickers, in order of how likely
    the router is to get each wrong. Section names are the worst guess available to it —
    a question about membership fees is not obviously Item 6 rather than Item 7 — while a
    named company is nearly always right.
    """
    steps: list[tuple[str, Filters]] = [("as routed", filters)]
    if filters.sections:
        steps.append(("without sections", replace(filters, sections=())))
    if filters.fiscal_years:
        steps.append(("without years", replace(filters, sections=(), fiscal_years=())))
    if not filters.is_empty():
        steps.append(("unfiltered", Filters()))
    return steps


def retrieve_node(state: GraphState) -> GraphState:
    """Fetch candidates for the current search query, relaxing filters if none match.

    A filter that matches nothing is worse than no filter: the right document is never
    considered at all, the grader sees an empty context, and the generator — given the
    chance — invents an answer with a fabricated citation. Relaxing progressively keeps
    the benefit of a correct filter without letting a wrong one end the run.
    """
    settings = get_settings()
    decision = state.get("route")
    strategy = Strategy(decision.strategy) if decision else Strategy.HYBRID

    filters = _filters_from(decision)

    # A comparison needs evidence from every company it names. One pooled search does not
    # provide that: measured over four cross-company questions, a single top-5 returned
    # both companies' figures zero times, because the strongest matches all come from
    # whichever filing phrases the metric most like the question. Searching per company
    # and merging takes that to 3/4. This is still one retrieval step, not a sub-agent.
    if len(filters.tickers) > 1:
        per_company = settings.comparison_chunks_per_company
        chunks = []
        for ticker in filters.tickers:
            chunks.extend(
                retrieve(
                    state["search_query"],
                    strategy,
                    replace(filters, tickers=(ticker,)),
                    k=per_company,
                )
            )
        if chunks:
            return {
                "chunks": chunks,
                "filters_used": f"per-company ({len(filters.tickers)})",
                "path": _note(state, "retrieve"),
            }

    chunks: list[RetrievedChunk] = []
    used = "as routed"
    for label, candidate in _relaxations(filters):
        chunks = retrieve(state["search_query"], strategy, candidate, k=settings.rerank_top_n)
        used = label
        if chunks:
            break

    if used != "as routed":
        logger.info("routed filters matched nothing; retrieved %s", used)

    return {"chunks": chunks, "filters_used": used, "path": _note(state, "retrieve")}


def grade_node(state: GraphState) -> GraphState:
    """Judge whether the retrieved context can answer the question.

    On failure this defaults to ``sufficient=False``, which costs one more retrieval.
    Defaulting the other way would let an LLM outage silently turn the corrective loop off
    while the system kept answering — the expensive failure, and an invisible one.
    """
    chunks = state.get("chunks", [])
    if not chunks:
        return {
            "grade": GradeResult(sufficient=False, missing="nothing was retrieved"),
            "path": _note(state, "grade"),
        }

    try:
        model = get_chat_model(ModelRole.GRADER).with_structured_output(GradeResult)
        result = model.invoke(
            get_prompt("grade").render(question=state["question"], context=format_context(chunks))
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("grader failed, assuming insufficient: %s", exc)
        return {
            "grade": GradeResult(sufficient=False, missing="grader unavailable"),
            "path": _note(state, "grade"),
            "errors": [*state.get("errors", []), f"grade: {exc}"],
        }

    return {"grade": result, "path": _note(state, "grade")}


def rewrite_node(state: GraphState) -> GraphState:
    """Reformulate the query in the vocabulary a filing would use."""
    grade = state.get("grade")
    previous = [*state.get("previous_queries", []), state["search_query"]]

    try:
        model = get_chat_model(ModelRole.REWRITER).with_structured_output(RewrittenQuery)
        rewritten = model.invoke(
            get_prompt("rewrite").render(
                question=state["question"],
                missing=grade.missing if grade else "unknown",
                previous_queries="\n".join(f"- {q}" for q in previous),
            )
        )
        new_query = rewritten.query.strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("rewriter failed, reusing the original question: %s", exc)
        return {
            "search_query": state["question"],
            "attempts": state.get("attempts", 0) + 1,
            "previous_queries": previous,
            "path": _note(state, "rewrite"),
            "errors": [*state.get("errors", []), f"rewrite: {exc}"],
        }

    # A rewriter that returns the query it was told not to repeat would burn an attempt
    # retrieving exactly the same chunks.
    if not new_query or new_query in previous:
        new_query = f"{state['question']} {grade.missing if grade else ''}".strip()

    return {
        "search_query": new_query,
        "attempts": state.get("attempts", 0) + 1,
        "previous_queries": previous,
        "path": _note(state, "rewrite"),
    }


# A model that fills a required string field with the field's own name has not answered.
_DEGENERATE_ANSWERS = {"answerable", "answer", "none", "null", "n/a", ""}


def generate_node(state: GraphState) -> GraphState:
    """Answer from the retrieved context, or refuse.

    Refusal is decided in code, not asked of the model. Measured on this corpus: handed
    context the grader had already rejected, the local generator answered "$2,337 million"
    with `answerable=True` — a figure lifted from an unrelated line item, and a citation to
    go with it. Instructing a model not to do that is a request; declining to call it is a
    guarantee.

    The cost is that a wrong grader now blocks a good answer, so grader accuracy becomes
    load-bearing. That is the right place for the risk to sit: a missed answer is visible
    and recoverable, a fabricated financial figure is neither.
    """
    chunks = state.get("chunks", [])

    if not chunks:
        return {
            "answer": "I could not find anything in these filings that addresses that.",
            "answerable": False,
            "citations": [],
            "path": _note(state, "generate"),
        }

    # When the grader rejected the context, generation still happens — but under a prompt
    # that names what was reported missing and requires citations. Measured: hard-refusing
    # on the grader's verdict lost 9 of 33 answerable questions whose evidence was in fact
    # present, because the grader has a high false-negative rate on long contexts. Making
    # it a single point of failure traded one error for a more common one. Both the grader
    # and the generator must now fail together, and an uncited answer is discarded either
    # way.
    grade = state.get("grade")
    cautious = grade is not None and not grade.sufficient
    prompt = (
        get_prompt("generate_cautious").render(
            question=state["question"],
            context=format_context(chunks),
            missing=grade.missing or "not specified",
        )
        if cautious
        else get_prompt("generate").render(
            question=state["question"], context=format_context(chunks)
        )
    )

    try:
        model = get_chat_model(ModelRole.GENERATOR).with_structured_output(FinalAnswer)
        result = model.invoke(prompt)
    except Exception as exc:  # noqa: BLE001
        logger.error("generator failed: %s", exc)
        return {
            "answer": "The answer could not be generated. Please retry.",
            "answerable": False,
            "citations": [],
            "path": _note(state, "generate"),
            "errors": [*state.get("errors", []), f"generate: {exc}"],
        }

    # Keep only citations that name a chunk actually retrieved. A fabricated citation is
    # more dangerous than a missing one, because it survives casual checking.
    valid = {chunk.citation for chunk in chunks}
    citations = [c for c in result.citations if c in valid]

    answer = result.answer.strip()
    if answer.lower() in _DEGENERATE_ANSWERS:
        answer = "I could not produce a supported answer from these filings."
        return {
            "answer": answer,
            "answerable": False,
            "citations": [],
            "path": _note(state, "generate"),
        }

    return {
        "answer": answer,
        "answerable": result.answerable and bool(citations),
        "citations": citations,
        "path": _note(state, "generate"),
    }


def should_retry(state: GraphState) -> str:
    """Route out of the grader: retry the search, or go on and answer.

    Bounded by `max_retrieval_attempts`, which caps latency and spend and guarantees the
    loop terminates even if the grader is never satisfied. On exhaustion the run proceeds
    to generation rather than erroring — the generator is instructed to say when the
    context does not contain the answer, which is the honest outcome.
    """
    grade = state.get("grade")
    if grade is not None and grade.sufficient:
        return "generate"
    if state.get("attempts", 0) >= get_settings().max_retrieval_attempts:
        logger.info("retry budget exhausted; answering from what was found")
        return "generate"
    return "rewrite"
