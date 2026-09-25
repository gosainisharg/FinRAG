"""Assembling the corrective-RAG graph.

    route → retrieve → grade → [rewrite → retrieve]* → generate

The cycle between `grade` and `retrieve` is the whole point: the system judges its own
retrieval and searches again when it is weak, instead of answering from whatever the first
search returned. Everything else is plumbing around that decision.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from finrag.graph.nodes import (
    generate_node,
    grade_node,
    retrieve_node,
    rewrite_node,
    route_node,
    should_retry,
)
from finrag.graph.state import GraphState, initial_state

logger = logging.getLogger(__name__)


def build_graph(checkpointer=None):
    """Wire and compile the graph.

    A checkpointer makes runs resumable and gives the same `thread_id` a shared history.
    `InMemorySaver` is the development default; production would use a Postgres-backed
    saver so state survives a restart.
    """
    graph = StateGraph(GraphState)

    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "route")
    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", "grade")

    # The corrective branch. `should_retry` is bounded by max_retrieval_attempts, so the
    # cycle always terminates even when the grader is never satisfied.
    graph.add_conditional_edges(
        "grade",
        should_retry,
        {"rewrite": "rewrite", "generate": "generate"},
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())


@lru_cache(maxsize=1)
def get_graph():
    """Process-wide compiled graph. Compilation is pure setup and need happen once."""
    return build_graph()


def answer(question: str, thread_id: str = "default") -> GraphState:
    """Run one question through the graph and return the final state.

    The whole state is returned rather than just the answer: the retrieval attempts, the
    grader's verdict and the node path are what make a wrong answer diagnosable.
    """
    return get_graph().invoke(
        initial_state(question),
        config={"configurable": {"thread_id": thread_id}},
    )
