"""Load documents from disk into LangChain Document objects."""

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_core.documents import Document

from src.config import DATA_DIR, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


def _loader_for(path: Path):
    """Return the appropriate LangChain loader for a file extension."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path))
    if suffix == ".docx":
        return Docx2txtLoader(str(path))
    if suffix in {".txt", ".md"}:
        return TextLoader(str(path), encoding="utf-8")
    raise ValueError(f"Unsupported file type: {suffix}")


def load_documents(directory: Path = DATA_DIR) -> List[Document]:
    """
    Load every supported document in a directory.

    Returns a flat list of Document objects. For PDFs, each page
    becomes one Document with page number in metadata.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    documents: List[Document] = []

    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            docs = _loader_for(path).load()
        except Exception as exc:
            logger.error("Failed to load %s: %s", path.name, exc)
            continue

        for doc in docs:
            doc.metadata["source_file"] = path.name

        documents.extend(docs)
        logger.info("Loaded %s -> %d document(s)", path.name, len(docs))

    if not documents:
        raise ValueError(f"No supported documents found in {directory}")

    return documents