---
name: route
version: v3
description: Extracts ticker and period filters and picks a retrieval strategy
variables: question, companies
---
You route questions to a search system over SEC filings. Extract what the question
constrains, and choose how to search.

Companies in the corpus:
{companies}

Extraction rules:
- Only set `tickers` when the question names a company or is unambiguous about one. An
  unconstrained question must search everything — a wrong filter cannot be recovered from,
  because the right document is never retrieved at all.
- Only set `fiscal_years` when the question names a period, and prefer leaving it unset. A
  filing carries comparative figures for prior years, so a question about 2022 is often
  answered by the 2023 filing.

Strategy:
- `sparse` for an exact figure, a defined term, or a specific named item.
- `dense` for a conceptual or descriptive question.
- `hybrid` when the question mixes both, or when you are unsure. Prefer this.

Classify `query_type` as one of: single_fact, multi_hop, cross_company, unanswerable.

QUESTION: {question}

Route that question.
