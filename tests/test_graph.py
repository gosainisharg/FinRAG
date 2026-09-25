"""The corrective-RAG graph. All LLM calls are faked; no network, no cost."""

from __future__ import annotations

import pytest

from finrag.config import get_settings
from finrag.graph import nodes
from finrag.graph.nodes import (
    _filters_from,
    _relaxations,
    format_context,
    generate_node,
    grade_node,
    retrieve_node,
    rewrite_node,
    should_retry,
)
from finrag.graph.schemas import FinalAnswer, GradeResult, RewrittenQuery, RouteDecision
from finrag.graph.state import GraphState, initial_state
from finrag.retrieval.types import Filters, RetrievedChunk, Strategy


def chunk(citation_ticker="TSLA", page=50, text="Total revenues 96,773") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"c{page}",
        text=text,
        metadata={
            "ticker": citation_ticker,
            "fiscal_year": 2023,
            "section": "Item 8",
            "page": page,
            "element_kind": "table",
        },
        score=1.0,
    )


class FakeModel:
    """Stands in for a structured-output chat model."""

    def __init__(self, result):
        self.result = result
        self.calls: list[str] = []

    def with_structured_output(self, _schema):
        return self

    def invoke(self, prompt):
        self.calls.append(prompt)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.fixture
def fake_llm(monkeypatch):
    def install(result):
        model = FakeModel(result)
        monkeypatch.setattr(nodes, "get_chat_model", lambda _role: model)
        return model

    return install


class TestShouldRetry:
    def test_sufficient_context_goes_straight_to_generate(self):
        state = GraphState(grade=GradeResult(sufficient=True), attempts=0)
        assert should_retry(state) == "generate"

    def test_insufficient_context_triggers_a_rewrite(self):
        state = GraphState(grade=GradeResult(sufficient=False, missing="x"), attempts=0)
        assert should_retry(state) == "rewrite"

    def test_the_loop_is_bounded(self):
        """Without this the cycle never terminates when the grader is never satisfied."""
        limit = get_settings().max_retrieval_attempts
        state = GraphState(grade=GradeResult(sufficient=False, missing="x"), attempts=limit)
        assert should_retry(state) == "generate"

    def test_missing_grade_is_treated_as_insufficient(self):
        assert should_retry(GraphState(attempts=0)) == "rewrite"


class TestFilterRelaxation:
    def test_sections_are_dropped_before_years(self):
        """Section is the router's least reliable guess — membership fees are not
        obviously Item 6 rather than Item 7 — while a named company nearly always is."""
        steps = _relaxations(Filters(tickers=("COST",), fiscal_years=(2023,), sections=("Item 7",)))
        labels = [label for label, _ in steps]
        assert labels == ["as routed", "without sections", "without years", "unfiltered"]

    def test_relaxation_order_progressively_weakens(self):
        steps = _relaxations(Filters(tickers=("COST",), fiscal_years=(2023,), sections=("Item 7",)))
        assert steps[1][1].sections == ()
        assert steps[2][1].fiscal_years == ()
        assert steps[3][1].is_empty()

    def test_no_relaxation_needed_when_unfiltered(self):
        assert len(_relaxations(Filters())) == 1

    def test_route_decision_converts_to_filters(self):
        decision = RouteDecision(query_type="single_fact", tickers=["tsla"], fiscal_years=[2023])
        filters = _filters_from(decision)
        assert filters.tickers == ("TSLA",)  # normalised to upper case
        assert filters.fiscal_years == (2023,)


class TestRetrieveNode:
    def test_relaxes_filters_when_nothing_matches(self, monkeypatch):
        attempts: list[Filters] = []

        def fake_retrieve(_query, _strategy, filters, k):
            attempts.append(filters)
            return [chunk()] if filters.is_empty() else []

        monkeypatch.setattr(nodes, "retrieve", fake_retrieve)
        state = initial_state("q")
        state["route"] = RouteDecision(
            query_type="single_fact", tickers=["COST"], sections=["Item 7"]
        )

        result = retrieve_node(state)
        assert result["chunks"]
        assert result["filters_used"] == "unfiltered"
        assert len(attempts) > 1

    def test_keeps_the_routed_filters_when_they_work(self, monkeypatch):
        monkeypatch.setattr(nodes, "retrieve", lambda *a, **k: [chunk()])
        state = initial_state("q")
        state["route"] = RouteDecision(query_type="single_fact", tickers=["TSLA"])
        assert retrieve_node(state)["filters_used"] == "as routed"


class TestGradeNode:
    def test_empty_context_is_insufficient_without_calling_a_model(self, fake_llm):
        model = fake_llm(GradeResult(sufficient=True))
        state = initial_state("q")
        result = grade_node(state)
        assert result["grade"].sufficient is False
        assert not model.calls, "should not spend a call on an empty context"

    def test_a_grader_failure_defaults_to_insufficient(self, fake_llm):
        """Defaulting the other way would let an outage silently disable the corrective
        loop while the system kept answering — expensive and invisible."""
        fake_llm(RuntimeError("provider down"))
        state = initial_state("q")
        state["chunks"] = [chunk()]
        result = grade_node(state)
        assert result["grade"].sufficient is False
        assert result["errors"]


class TestRewriteNode:
    def test_records_the_previous_query_and_increments_attempts(self, fake_llm):
        fake_llm(RewrittenQuery(query="total net sales fiscal 2023"))
        state = initial_state("revenue?")
        state["grade"] = GradeResult(sufficient=False, missing="revenue figure")
        result = rewrite_node(state)
        assert result["search_query"] == "total net sales fiscal 2023"
        assert result["attempts"] == 1
        assert "revenue?" in result["previous_queries"]

    def test_a_repeated_query_is_replaced(self, fake_llm):
        """Re-issuing a query already tried burns an attempt on identical chunks."""
        fake_llm(RewrittenQuery(query="revenue?"))
        state = initial_state("revenue?")
        state["grade"] = GradeResult(sufficient=False, missing="2023 R&D expense")
        result = rewrite_node(state)
        assert result["search_query"] != "revenue?"
        assert "2023 R&D expense" in result["search_query"]

    def test_a_rewriter_failure_still_advances_the_counter(self, fake_llm):
        # Otherwise a persistently failing rewriter loops forever.
        fake_llm(RuntimeError("down"))
        state = initial_state("q")
        assert rewrite_node(state)["attempts"] == 1


class TestGenerateNode:
    def test_refuses_when_the_grader_rejected_the_context(self, fake_llm):
        """Measured: given context the grader had rejected, the local generator answered
        '$2,337 million' with answerable=True. Declining to call it is the guarantee."""
        model = fake_llm(FinalAnswer(answer="$2,337 million", answerable=True, citations=[]))
        state = initial_state("q")
        state["chunks"] = [chunk()]
        state["grade"] = GradeResult(sufficient=False, missing="membership fee revenue")

        result = generate_node(state)
        assert result["answerable"] is False
        assert "membership fee revenue" in result["answer"]
        assert not model.calls, "the model must not be asked to answer rejected context"

    def test_refusal_cites_what_was_searched(self, fake_llm):
        fake_llm(FinalAnswer(answer="x", answerable=True))
        state = initial_state("q")
        state["chunks"] = [chunk()]
        state["grade"] = GradeResult(sufficient=False, missing="y")
        citations = generate_node(state)["citations"]
        assert "TSLA FY2023 Item 8 p50" in citations

    def test_drops_fabricated_citations(self, fake_llm):
        """A fabricated citation is worse than a missing one: it survives casual checking."""
        fake_llm(
            FinalAnswer(
                answer="96,773 million",
                answerable=True,
                citations=["TSLA FY2023 Item 8 p50", "AAPL FY2099 Item 8 p1"],
            )
        )
        state = initial_state("q")
        state["chunks"] = [chunk()]
        state["grade"] = GradeResult(sufficient=True)
        assert generate_node(state)["citations"] == ["TSLA FY2023 Item 8 p50"]

    def test_a_field_name_as_the_answer_is_rejected(self, fake_llm):
        """Observed with the local model: it filled `answer` with the word 'answerable'."""
        fake_llm(FinalAnswer(answer="answerable", answerable=False, citations=[]))
        state = initial_state("q")
        state["chunks"] = [chunk()]
        state["grade"] = GradeResult(sufficient=True)
        result = generate_node(state)
        assert result["answer"] != "answerable"
        assert result["answerable"] is False

    def test_an_uncited_answer_is_not_marked_answerable(self, fake_llm):
        fake_llm(FinalAnswer(answer="96,773 million", answerable=True, citations=[]))
        state = initial_state("q")
        state["chunks"] = [chunk()]
        state["grade"] = GradeResult(sufficient=True)
        assert generate_node(state)["answerable"] is False

    def test_no_chunks_means_no_model_call(self, fake_llm):
        model = fake_llm(FinalAnswer(answer="invented", answerable=True))
        result = generate_node(initial_state("q"))
        assert result["answerable"] is False
        assert not model.calls


class TestFormatContext:
    def test_labels_each_chunk_with_its_citation(self):
        """Inlining the label lets a citation be checked against its chunk rather than
        left to the model to construct."""
        text = format_context([chunk()])
        assert "[TSLA FY2023 Item 8 p50]" in text


class TestGraphWiring:
    def test_compiles_with_the_corrective_cycle(self):
        from finrag.graph.build import build_graph

        compiled = build_graph().get_graph()
        names = {node.name for node in compiled.nodes.values()}
        assert {"route", "retrieve", "grade", "rewrite", "generate"} <= names
        # The cycle that makes this corrective rather than a pipeline.
        assert any(e.source == "rewrite" and e.target == "retrieve" for e in compiled.edges)
        assert any(e.source == "grade" for e in compiled.edges)

    def test_strategy_enum_round_trips_through_the_schema(self):
        decision = RouteDecision(query_type="single_fact", strategy=Strategy.SPARSE)
        assert Strategy(decision.strategy) is Strategy.SPARSE
