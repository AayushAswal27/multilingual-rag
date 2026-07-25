"""Load documents from disk into LangChain Document objects."""

import logging
import os
import tempfile
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


def load_uploaded_file(file, suffix: str) -> List[Document]:
    """
    Load a single file uploaded through Streamlit (in-memory).

    Streamlit hands us a file-like object, not a path, so we write
    it to a temporary file the LangChain loaders can open, then
    delete it once loaded.
    """
    suffix = suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.getvalue())
        tmp_path = Path(tmp.name)

    try:
        docs = _loader_for(tmp_path).load()
        for doc in docs:
            doc.metadata["source_file"] = file.name
        logger.info("Loaded uploaded file %s -> %d document(s)", file.name, len(docs))
        return docs
    finally:
        os.remove(tmp_path)