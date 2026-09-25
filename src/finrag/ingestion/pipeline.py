"""Build the vector index from source filings.

    python -m finrag.ingestion.pipeline              # ingest data/, replacing each source
    python -m finrag.ingestion.pipeline --reset      # drop the collection and rebuild
    python -m finrag.ingestion.pipeline --dry-run    # chunk and report, write nothing

Re-ingesting a source deletes its existing chunks first rather than upserting. Chunk ids
are content hashes, so an edited document produces *new* ids and the superseded chunks
would otherwise survive and stay retrievable — silently answering from a stale filing.
That was the defect in the original implementation, which skipped any chunk whose id
already existed and therefore never picked up a change.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from langchain_chroma import Chroma

from finrag.config import get_settings
from finrag.ingestion.chunking import Chunk, chunk_document
from finrag.ingestion.embeddings import get_embeddings
from finrag.ingestion.loaders import load_document

logger = logging.getLogger(__name__)


@dataclass
class SourceReport:
    source: str
    ticker: str | None
    form_type: str
    chunks: int
    tables: int
    deleted: int
    seconds: float


@dataclass
class IngestionReport:
    sources: list[SourceReport] = field(default_factory=list)
    collection: str = ""
    total_in_collection: int = 0

    @property
    def total_chunks(self) -> int:
        return sum(s.chunks for s in self.sources)

    def render(self) -> str:
        lines = [
            f"{'source':<34} {'ticker':<7} {'form':<17} "
            f"{'chunks':>7} {'tables':>7} {'del':>5} {'sec':>6}",
            "-" * 88,
        ]
        for s in self.sources:
            lines.append(
                f"{s.source[:33]:<34} {str(s.ticker):<7} {s.form_type:<17} "
                f"{s.chunks:>7} {s.tables:>7} {s.deleted:>5} {s.seconds:>6.1f}"
            )
        lines.append("-" * 88)
        lines.append(
            f"{'total':<34} {'':<7} {'':<17} {self.total_chunks:>7}"
            f"\ncollection {self.collection!r} now holds {self.total_in_collection} chunks"
        )
        return "\n".join(lines)


def open_vector_store() -> Chroma:
    """Open (or create) the persistent Chroma collection."""
    settings = get_settings()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_dir),
        # Vectors are normalised, so cosine distance is a dot product. Chroma's default is
        # squared L2, which on unit vectors ranks identically but reports distances on a
        # different scale — awkward when fusing with BM25 later.
        collection_metadata={"hnsw:space": "cosine"},
    )


def _delete_existing(store: Chroma, source: str) -> int:
    """Remove a source's chunks so a re-ingest replaces rather than accumulates."""
    existing = store.get(where={"source": source}, include=[])
    ids = existing.get("ids", [])
    if ids:
        store.delete(ids=ids)
    return len(ids)


def ingest_path(store: Chroma | None, path: Path) -> tuple[SourceReport, list[Chunk]]:
    """Chunk one filing and, unless `store` is None, write it to the index."""
    started = time.perf_counter()
    metadata, elements = load_document(path)
    chunks = chunk_document(metadata, elements)

    deleted = 0
    if store is not None and chunks:
        deleted = _delete_existing(store, metadata.source)
        store.add_texts(
            texts=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
            ids=[c.chunk_id for c in chunks],
        )

    report = SourceReport(
        source=metadata.source,
        ticker=metadata.ticker,
        form_type=str(metadata.form_type),
        chunks=len(chunks),
        tables=sum(c.metadata.get("element_kind") == "table" for c in chunks),
        deleted=deleted,
        seconds=time.perf_counter() - started,
    )
    return report, chunks


def build_index(
    data_dir: Path | None = None, *, reset: bool = False, dry_run: bool = False
) -> IngestionReport:
    settings = get_settings()
    data_dir = data_dir or settings.data_dir

    pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"no PDFs found in {data_dir}")

    store: Chroma | None = None
    if not dry_run:
        store = open_vector_store()
        if reset:
            logger.info("resetting collection %s", settings.collection_name)
            store.reset_collection()

    report = IngestionReport(collection=settings.collection_name)
    for pdf in pdfs:
        source_report, _ = ingest_path(store, pdf)
        report.sources.append(source_report)
        logger.info("%s → %d chunks", source_report.source, source_report.chunks)

    if store is not None:
        report.total_in_collection = store._collection.count()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="drop the collection first")
    parser.add_argument("--dry-run", action="store_true", help="chunk and report, write nothing")
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = build_index(args.data_dir, reset=args.reset, dry_run=args.dry_run)
    print()
    print(report.render())
    if args.dry_run:
        print("\ndry run — nothing was written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
