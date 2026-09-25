"""The graph's shared state.

Everything a node reads or writes lives here, including the fields that exist purely to
explain a run afterwards. The diagnostic fields are not incidental: the difference between
"the agentic loop helps" as a claim and as a finding is being able to point at how many
times the grader rejected context and what it asked for instead.
"""

from __future__ import annotations

from typing import TypedDict

from finrag.graph.schemas import GradeResult, RouteDecision
from finrag.retrieval.types import RetrievedChunk


class GraphState(TypedDict, total=False):
    """State threaded through every node.

    `total=False` because nodes fill their own fields as the run progresses; requiring all
    keys up front would mean inventing empty values that are indistinguishable from real
    ones at the point a node reads them.
    """

    # --- input ---
    question: str

    # --- routing ---
    route: RouteDecision | None

    # --- retrieval ---
    # The query actually sent to the retriever. Starts as the question and is replaced by
    # the rewriter, which is why it is separate from `question`: citations and grading must
    # keep referring to what the user actually asked.
    search_query: str
    chunks: list[RetrievedChunk]
    # Which relaxation of the routed filters actually returned results. 'as routed'
    # means the router was right; anything else means it over-filtered.
    filters_used: str

    # --- grading loop ---
    grade: GradeResult | None
    attempts: int
    previous_queries: list[str]

    # --- output ---
    answer: str
    answerable: bool
    citations: list[str]

    # --- diagnostics ---
    # Node visit order, so a trace can be read without a tracing backend attached.
    path: list[str]
    errors: list[str]


def initial_state(question: str) -> GraphState:
    return GraphState(
        question=question,
        search_query=question,
        chunks=[],
        attempts=0,
        previous_queries=[],
        path=[],
        errors=[],
        route=None,
        grade=None,
    )
