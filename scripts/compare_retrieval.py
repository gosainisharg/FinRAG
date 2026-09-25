#!/usr/bin/env python
"""Run one question through every retrieval strategy, side by side.

    python scripts/compare_retrieval.py "What were Tesla's total revenues in 2023?"
    python scripts/compare_retrieval.py "..." --ticker TSLA --k 5
    python scripts/compare_retrieval.py "..." --strategies sparse hybrid_rerank

Exists so strategy choice is made by looking at what each actually returns. Phase 3 will
score these against a golden set; until then this is how the behaviour is inspected, and
it stays useful afterwards for explaining why a particular answer came out wrong.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from finrag.retrieval import Filters, Strategy, retrieve

BOLD, DIM, OFF = "\033[1m", "\033[2m", "\033[0m"


def fmt_scores(chunk) -> str:
    parts = []
    if chunk.dense_score is not None:
        parts.append(f"d={chunk.dense_score:.3f}")
    if chunk.sparse_score is not None:
        parts.append(f"s={chunk.sparse_score:.2f}")
    if chunk.rerank_score is not None:
        parts.append(f"r={chunk.rerank_score:.3f}")
    return " ".join(parts) or f"{chunk.score:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--ticker", action="append", default=[])
    parser.add_argument("--year", action="append", type=int, default=[])
    parser.add_argument("--section", action="append", default=[])
    parser.add_argument("--include-exhibits", action="store_true")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=[s.value for s in Strategy],
        choices=[s.value for s in Strategy],
    )
    parser.add_argument("--reranker", help="override the cross-encoder model")
    args = parser.parse_args()

    # sentence-transformers logs an HTTP request per config file at INFO.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    filters = Filters(
        tickers=tuple(t.upper() for t in args.ticker),
        fiscal_years=tuple(args.year),
        sections=tuple(args.section),
        exclude_exhibits=not args.include_exhibits,
    )

    print(f"\n{BOLD}query:{OFF} {args.query}")
    if not filters.is_empty() or not filters.exclude_exhibits:
        print(f"{DIM}filters: {filters}{OFF}")

    # Chunks that every strategy agrees on are the stable core; the differences are where
    # strategy choice actually matters.
    per_strategy: dict[str, list[str]] = {}

    for name in args.strategies:
        strategy = Strategy(name)
        started = time.perf_counter()
        try:
            hits = retrieve(args.query, strategy, filters, k=args.k, reranker_model=args.reranker)
        except Exception as exc:  # noqa: BLE001 - a broken strategy must not hide the rest
            print(f"\n{BOLD}{name}{OFF} — failed: {type(exc).__name__}: {exc}")
            continue
        elapsed = (time.perf_counter() - started) * 1000

        per_strategy[name] = [h.chunk_id for h in hits]
        print(f"\n{BOLD}{name}{OFF} {DIM}({elapsed:.0f}ms){OFF}")
        for rank, hit in enumerate(hits, start=1):
            kind = hit.metadata.get("element_kind", "?")
            # Table chunks all begin with the company name; the informative line is the
            # statement title or the first data row.
            lines = [ln for ln in hit.text.strip().splitlines() if ln.strip()]
            first_line = " / ".join(lines[:2])[:64] if lines else ""
            print(
                f"  {rank}. {hit.citation:<26} {kind:<6} {DIM}{fmt_scores(hit):<34}{OFF} "
                f"{first_line!r}"
            )

    if len(per_strategy) > 1:
        common = set.intersection(*(set(v) for v in per_strategy.values()))
        union = set.union(*(set(v) for v in per_strategy.values()))
        print(
            f"\n{DIM}{len(common)} of {len(union)} distinct chunks returned by every "
            f"strategy — the rest is where strategy choice matters{OFF}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
