"""Build, save, and load the FAISS vector index."""

import logging
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import VECTORDB_DIR
from src.embeddings import get_embeddings

logger = logging.getLogger(__name__)

INDEX_NAME = "faiss_index"


def build_vector_store(chunks: List[Document]) -> FAISS:
    """Embed chunks and build an in-memory FAISS index."""
    if not chunks:
        raise ValueError("Cannot build a vector store from zero chunks")

    logger.info("Embedding %d chunk(s)...", len(chunks))
    store = FAISS.from_documents(chunks, get_embeddings())
    logger.info("Vector store built")
    return store


def save_vector_store(store: FAISS, directory: Path = VECTORDB_DIR) -> None:
    """Persist the index to disk."""
    directory.mkdir(parents=True, exist_ok=True)
    store.save_local(str(directory), index_name=INDEX_NAME)
    logger.info("Vector store saved to %s", directory)


def load_vector_store(directory: Path = VECTORDB_DIR) -> FAISS:
    """
    Load a previously saved index.

    Raises FileNotFoundError if the index does not exist, which
    normally means ingest.py has not been run yet.
    """
    index_file = directory / f"{INDEX_NAME}.faiss"
    if not index_file.exists():
        raise FileNotFoundError(
            f"No index at {index_file}. Run `python ingest.py` first."
        )

    store = FAISS.load_local(
        str(directory),
        get_embeddings(),
        index_name=INDEX_NAME,
        allow_dangerous_deserialization=True,
    )
    logger.info("Vector store loaded from %s", directory)
    return store