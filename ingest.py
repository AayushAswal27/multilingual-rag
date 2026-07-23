"""
CLI entry point for building the vector index.

Usage:
    python ingest.py
"""

import logging
import sys

from src.loaders import load_documents
from src.chunking import chunk_documents
from src.vector_store import build_vector_store, save_vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> int:
    try:
        documents = load_documents()
        chunks = chunk_documents(documents)
        store = build_vector_store(chunks)
        save_vector_store(store)
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc)
        return 1

    logger.info("Done. %d chunk(s) indexed.", len(chunks))
    return 0


if __name__ == "__main__":
    sys.exit(main())