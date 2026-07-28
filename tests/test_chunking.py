"""Tests for the chunking module."""

from langchain_core.documents import Document

from src.chunking import chunk_documents
from src.config import CHUNK_SIZE


def _make_doc(text: str) -> Document:
    return Document(page_content=text, metadata={"source_file": "test.pdf", "page": 0})


def test_chunking_splits_long_text():
    """A long document should split into multiple chunks."""
    long_text = "This is a sentence. " * 200  # ~4000 chars
    chunks = chunk_documents([_make_doc(long_text)])
    assert len(chunks) > 1


def test_chunks_respect_max_size():
    """No chunk should greatly exceed the configured chunk size."""
    long_text = "word " * 1000
    chunks = chunk_documents([_make_doc(long_text)])
    for c in chunks:
        # allow a small margin — splitter breaks at boundaries
        assert len(c.page_content) <= CHUNK_SIZE + 200


def test_metadata_is_preserved():
    """Chunks must inherit source_file and page from the parent."""
    chunks = chunk_documents([_make_doc("Some text. " * 100)])
    for c in chunks:
        assert c.metadata["source_file"] == "test.pdf"
        assert c.metadata["page"] == 0


def test_chunk_id_is_assigned():
    """Every chunk should get a unique chunk_id."""
    chunks = chunk_documents([_make_doc("Some text. " * 100)])
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert ids == list(range(len(chunks)))


def test_empty_input_returns_empty():
    """No documents in, no chunks out."""
    assert chunk_documents([]) == []