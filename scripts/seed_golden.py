#!/usr/bin/env python
"""Append the hand-authored golden questions.

    python scripts/seed_golden.py

Two question types cannot come from the generator, which works from one chunk at a time:

- **cross_company** needs figures from two filings and a comparison the model cannot see.
- **multi_hop** needs two facts from different chunks of one filing. The local model was
  tried for this and abandoned: it took over 150 seconds per question on the two-excerpt
  prompt and produced nothing usable, so these are authored from verified figures instead.
- **unanswerable** needs a question the corpus genuinely cannot answer, which means knowing
  what is *absent* — the opposite of what generating from a chunk produces.

Unanswerable questions are the ones usually left out of an eval set, and they are the most
important here. Without them a system that fabricates a confident answer scores identically
to one that correctly declines. They carry no answer tokens, so `is_relevant` is False for
every chunk by construction and they are scored through `abstention_rate` instead.

Every figure below was read from the indexed table rows during Phase 3, not recalled. The
entries are still written unverified so they pass through the same review as the generated
ones.
"""

from __future__ import annotations

import sys

from finrag.evaluation.golden import GoldenQuestion, QuestionType, load_golden, save_golden

APPLE_FY24 = "10-Q4-2024-As-Filed.pdf"
APPLE_FY23 = "_10-K-Q4-2023-As-Filed.pdf"
TESLA = "tsla-20231231-gen.pdf"
COSTCO = "cost-annual-report-final-pdf-from-dfin.pdf"
META_Q4 = "Meta-12-31-2023-Exhibit-99-1-FINAL.pdf"

CROSS_COMPANY = [
    (
        "Which reported higher total revenue in fiscal 2023, Tesla or Costco, and by how much?",
        "Costco $242,290M versus Tesla $96,773M — Costco higher by $145,517M",
        ["242,290", "96,773"],
        ["COST", "TSLA"],
        [2023],
        [{"source": COSTCO, "page": 40}, {"source": TESLA, "page": 50}],
    ),
    (
        "How did Apple's fiscal 2024 total net sales compare with Tesla's fiscal 2023 "
        "total revenues?",
        "Apple $391,035M versus Tesla $96,773M",
        ["391,035", "96,773"],
        ["AAPL", "TSLA"],
        [2023, 2024],
        [{"source": APPLE_FY24, "page": 31}, {"source": TESLA, "page": 50}],
    ),
    (
        "Compare Apple's fiscal 2023 total net sales with Costco's fiscal 2023 total revenue.",
        "Apple $383,285M versus Costco $242,290M",
        ["383,285", "242,290"],
        ["AAPL", "COST"],
        [2023],
        [{"source": APPLE_FY23, "page": 23}, {"source": COSTCO, "page": 40}],
    ),
    (
        "What total revenue did Meta report for the fourth quarter of 2023, and how does it "
        "compare with Tesla's full fiscal year 2023 revenue?",
        "Meta Q4 2023 $40,111M versus Tesla FY2023 $96,773M",
        ["40,111", "96,773"],
        ["META", "TSLA"],
        [2023],
        [{"source": META_Q4, "page": 9}, {"source": TESLA, "page": 50}],
    ),
]

# Each pair of figures lives in a *different* chunk of the same filing, which is what makes
# these multi-hop: retrieving one chunk answers half the question. They stress recall, where
# hit-rate and MRR both look healthy while the answer is still incomplete.
MULTI_HOP = [
    (
        "What were Tesla's total revenues and total assets at the end of fiscal 2023?",
        "Total revenues $96,773M; total assets $106,618M",
        ["96,773", "106,618"],
        ["TSLA"],
        [2023],
        [{"source": TESLA, "page": 50}, {"source": TESLA, "page": 49}],
    ),
    (
        "How much did Tesla spend on research and development in 2023, and what were its "
        "total revenues that year?",
        "R&D $3,969M against total revenues of $96,773M",
        ["3,969", "96,773"],
        ["TSLA"],
        [2023],
        [{"source": TESLA, "page": 41}, {"source": TESLA, "page": 50}],
    ),
    (
        "What were Apple's net income and total assets in fiscal 2024?",
        "Net income $93,736M; total assets $364,980M",
        ["93,736", "364,980"],
        ["AAPL"],
        [2024],
        [{"source": APPLE_FY24, "page": 31}, {"source": APPLE_FY24, "page": 33}],
    ),
    (
        "How much did Apple spend on research and development in fiscal 2024, and what were "
        "its total net sales?",
        "R&D $31,370M against total net sales of $391,035M",
        ["31,370", "391,035"],
        ["AAPL"],
        [2024],
        [{"source": APPLE_FY24, "page": 26}, {"source": APPLE_FY24, "page": 24}],
    ),
    (
        "What did Costco collect in membership fees in fiscal 2023, and what was its total "
        "revenue?",
        "Membership fees $4,580M of $242,290M total revenue",
        ["4,580", "242,290"],
        ["COST"],
        [2023],
        [{"source": COSTCO, "page": 29}, {"source": COSTCO, "page": 40}],
    ),
]


# Each is absent from these six filings for a stated reason, so a correct system declines.
UNANSWERABLE = [
    (
        "What was Tesla's employee headcount in Germany at the end of 2023?",
        "Headcount is disclosed only in total, never by country.",
        ["TSLA"],
    ),
    (
        "How much did Costco spend on television advertising in fiscal 2023?",
        "Costco does not disclose an advertising expense line.",
        ["COST"],
    ),
    (
        "What is Apple's projected total net sales for fiscal 2027?",
        "A 10-K reports historical results; it contains no multi-year revenue projection.",
        ["AAPL"],
    ),
    (
        "What was Meta's total revenue in 2019?",
        "The corpus holds only Meta's 2023 and 2024 results.",
        ["META"],
    ),
    (
        "Who is Tesla's largest individual shareholder by percentage owned?",
        "Item 12 incorporates beneficial ownership by reference to the proxy statement.",
        ["TSLA"],
    ),
    (
        "What was Apple's gross margin percentage in the first quarter of fiscal 2025?",
        "That quarter falls after the FY2024 filing period.",
        ["AAPL"],
    ),
]


def build(start_index: int) -> list[GoldenQuestion]:
    questions: list[GoldenQuestion] = []

    for kind, group in (
        (QuestionType.MULTI_HOP, MULTI_HOP),
        (QuestionType.CROSS_COMPANY, CROSS_COMPANY),
    ):
        for question, answer, tokens, tickers, years, pages in group:
            questions.append(
                GoldenQuestion(
                    id=f"q{start_index + len(questions) + 1:03d}",
                    question=question,
                    question_type=kind,
                    answer=answer,
                    answer_tokens=tokens,
                    tickers=tickers,
                    fiscal_years=years,
                    expected_pages=pages,
                    verified=False,
                    notes="hand-authored; figures read from indexed table rows",
                )
            )

    for question, reason, tickers in UNANSWERABLE:
        questions.append(
            GoldenQuestion(
                id=f"q{start_index + len(questions) + 1:03d}",
                question=question,
                question_type=QuestionType.UNANSWERABLE,
                answer=None,
                answer_tokens=[],
                tickers=tickers,
                fiscal_years=[],
                expected_pages=[],
                verified=False,
                notes=f"unanswerable: {reason}",
            )
        )

    return questions


def main() -> int:
    existing = load_golden(verified_only=False)
    if any(q.question_type is QuestionType.CROSS_COMPANY for q in existing):
        print("seeds already present; nothing to do")
        return 0

    seeds = build(len(existing))
    path = save_golden(existing + seeds)
    print(
        f"added {len(seeds)} hand-authored questions ({len(existing) + len(seeds)} total) → {path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
