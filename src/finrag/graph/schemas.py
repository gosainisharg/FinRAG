"""Structured outputs for the graph's decision nodes.

Every LLM decision in the graph returns a validated object rather than prose the code has
to parse. Parsing free text is where agentic systems rot: a model that answers "yes, the
context looks sufficient" instead of `true` either crashes the loop or, worse, is coerced
into a default that silently disables it.

Field descriptions are part of the prompt — the model sees them — so they are written as
instructions, not as notes to a reader.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from finrag.evaluation.golden import QuestionType
from finrag.retrieval.types import Strategy


class RouteDecision(BaseModel):
    """How to search for this question."""

    query_type: Literal[
        QuestionType.SINGLE_FACT,
        QuestionType.MULTI_HOP,
        QuestionType.CROSS_COMPANY,
        QuestionType.UNANSWERABLE,
    ] = Field(description="What kind of question this is")

    strategy: Literal[Strategy.DENSE, Strategy.SPARSE, Strategy.HYBRID] = Field(
        default=Strategy.HYBRID,
        description="sparse for an exact figure or defined term, dense for a conceptual "
        "question, hybrid when unsure",
    )

    tickers: list[str] = Field(
        default_factory=list,
        description="Ticker symbols the question is restricted to. Leave empty unless the "
        "question names a company.",
    )
    fiscal_years: list[int] = Field(
        default_factory=list,
        description="Fiscal years the question is restricted to. Leave empty unless the "
        "question names a period.",
    )
    # Section filtering was measured and removed. Over 33 answerable questions, letting
    # the router choose a 10-K section dropped evidence-in-top-5 from 0.848 to 0.606 — it
    # put Japan net sales in Item 8 when the discussion is in Item 7, and once emitted
    # "10-K", which is not a section at all. Tickers cost nothing; sections cost 24 points.
    reasoning: str = Field(default="", description="One sentence explaining the choice")


class GradeResult(BaseModel):
    """Whether the retrieved context can answer the question."""

    sufficient: bool = Field(
        description="True ONLY if every fact the question asks for appears in the context"
    )
    missing: str = Field(
        default="",
        description="What specific fact is absent. Empty when sufficient. This is handed "
        "to the query rewriter, so name the thing, do not describe the problem.",
    )
    confidence: Literal["low", "medium", "high"] = Field(
        default="medium", description="Confidence in this judgement"
    )


class RewrittenQuery(BaseModel):
    """A reformulated search query after retrieval came back insufficient."""

    query: str = Field(description="The new search query, using the vocabulary a filing would use")
    reasoning: str = Field(default="", description="One sentence on what changed and why")


class FinalAnswer(BaseModel):
    """The graph's answer, with the provenance needed to check it."""

    answer: str = Field(
        description="The answer, or a plain statement that it is not in the filings"
    )
    # Defaulted rather than required: one run failed validation entirely because the
    # model omitted the field, losing an otherwise good answer. Optimistic is safe here
    # because generate_node independently requires a surviving citation before marking
    # anything answerable.
    answerable: bool = Field(
        default=True,
        description="False if the context did not contain the answer. Set this honestly — "
        "an unanswerable question answered confidently is the worst outcome.",
    )
    citations: list[str] = Field(
        default_factory=list,
        description="Citation labels for every figure used, e.g. 'TSLA FY2023 Item 8 p50'",
    )
