"""CLI entry point for the ingestion pipeline.

Usage:
    python scripts/ingest_cli.py [--reset] [--limit N]

Prints the IngestResponse as indented JSON on completion.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python scripts/ingest_cli.py` from any cwd.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.core.deps import get_embedder, get_sparse_retriever, get_vector_store  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.services.ingestion import run_ingest  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the InfoQuest ingestion pipeline: DB -> embed -> vector/BM25 index.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop the existing vector collection and BM25 index before ingesting.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of ROLE documents ingested (for dev/testing).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    # Console-friendly logs for CLI usage; the FastAPI app uses JSON logs.
    configure_logging(level="INFO", fmt="console")

    response = run_ingest(
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        sparse=get_sparse_retriever(),
        reset=args.reset,
        limit=args.limit,
    )
    print(json.dumps(response.model_dump(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
