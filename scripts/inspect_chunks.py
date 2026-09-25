#!/usr/bin/env python
"""Eyeball what actually went into the index.

    python scripts/inspect_chunks.py                      # summary + 8 random chunks
    python scripts/inspect_chunks.py -n 20                # more samples
    python scripts/inspect_chunks.py --ticker TSLA --tables
    python scripts/inspect_chunks.py --search "total revenues"

Chunk quality is the ceiling on every retrieval number, and it is not something a test can
fully assert — a chunk can be well-formed and still be useless to read. This exists so that
judgement is made on real output rather than on statistics.
"""

from __future__ import annotations

import argparse
import collections
import random
import sys
import textwrap

from finrag.config import get_settings
from finrag.ingestion.pipeline import open_vector_store

DIM = "\033[2m"
BOLD = "\033[1m"
OFF = "\033[0m"


def summarise(metadatas: list[dict]) -> None:
    total = len(metadatas)
    print(f"\n{BOLD}collection: {get_settings().collection_name} — {total} chunks{OFF}")

    for field in ("ticker", "form_type", "element_kind"):
        counts = collections.Counter(m.get(field, "—") for m in metadatas)
        line = "  ".join(f"{k}={v}" for k, v in counts.most_common())
        print(f"  {field:<13} {line}")

    sections = collections.Counter(m.get("section", "—") for m in metadatas)
    print(f"  {'sections':<13} {len(sections)} distinct")
    for name, count in sections.most_common(8):
        bar = "█" * max(1, round(count / total * 60))
        print(f"      {name:<12} {count:>5}  {DIM}{bar}{OFF}")


def show(text: str, metadata: dict, index: int, width: int, full: bool) -> None:
    head = (
        f"{metadata.get('ticker')} · FY{metadata.get('fiscal_year')} · "
        f"{metadata.get('section')} ({metadata.get('section_title')}) · "
        f"p{metadata.get('page')} · {metadata.get('element_kind')} · {len(text)} chars"
    )
    print(f"\n{BOLD}[{index}] {head}{OFF}")
    print(f"{DIM}{metadata.get('source')}{OFF}")

    body = text if full else text[:width]
    for line in body.splitlines():
        # Table rows are wide; truncating them keeps column alignment readable.
        print("   " + (line if line.startswith("|") else textwrap.shorten(line, 110)))
    if not full and len(text) > width:
        print(f"   {DIM}... {len(text) - width} more chars{OFF}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--num", type=int, default=8, help="how many chunks to show")
    parser.add_argument("--ticker", help="filter by ticker, e.g. TSLA")
    parser.add_argument("--section", help="filter by section, e.g. 'Item 1A'")
    parser.add_argument("--tables", action="store_true", help="only table chunks")
    parser.add_argument("--search", help="show nearest neighbours to a query instead")
    parser.add_argument("--full", action="store_true", help="print whole chunks")
    parser.add_argument("--seed", type=int, default=None, help="fix the random sample")
    args = parser.parse_args()

    store = open_vector_store()

    if args.search:
        print(f"\n{BOLD}nearest to {args.search!r}{OFF}")
        for i, (doc, distance) in enumerate(
            store.similarity_search_with_score(args.search, k=args.num), start=1
        ):
            show(doc.page_content, doc.metadata, i, 600, args.full)
            print(f"   {DIM}cosine distance {distance:.4f}{OFF}")
        return 0

    where: dict = {}
    if args.ticker:
        where["ticker"] = args.ticker.upper()
    if args.section:
        where["section"] = args.section
    if args.tables:
        where["element_kind"] = "table"
    # Chroma requires an explicit $and for more than one condition.
    if len(where) > 1:
        where = {"$and": [{k: v} for k, v in where.items()]}

    result = store.get(where=where or None, include=["documents", "metadatas"])
    documents, metadatas = result["documents"], result["metadatas"]
    if not documents:
        print("no chunks matched that filter", file=sys.stderr)
        return 1

    summarise(metadatas)

    rng = random.Random(args.seed)
    picks = rng.sample(range(len(documents)), min(args.num, len(documents)))
    for i, pick in enumerate(picks, start=1):
        show(documents[pick], metadatas[pick], i, 600, args.full)
    return 0


if __name__ == "__main__":
    sys.exit(main())
