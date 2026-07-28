"""Tests for the retriever, using a small in-memory index."""

import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.embeddings import get_embeddings
from src.retriever import Retriever


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    """Build a tiny FAISS index from a few known documents."""
    docs = [
        Document(page_content="Farmers who pay income tax are excluded from the scheme.",
                 metadata={"source_file": "test.pdf", "page": 0}),
        Document(page_content="The benefit is six thousand rupees per year in three installments.",
                 metadata={"source_file": "test.pdf", "page": 1}),
        Document(page_content="Land records are used to identify eligible beneficiaries.",
                 metadata={"source_file": "test.pdf", "page": 2}),
    ]
    store = FAISS.from_documents(docs, get_embeddings())
    return Retriever(store=store, top_k=2)


def test_retrieve_returns_chunks(retriever):
    results = retriever.retrieve("Who is excluded from the scheme?")
    assert len(results) == 2
    assert all(hasattr(r, "distance") for r in results)


def test_relevant_query_survives_threshold(retriever):
    """A relevant question should keep at least one chunk."""
    results = retriever.retrieve_relevant("Who is excluded?", max_distance=1.6)
    assert len(results) >= 1


def test_irrelevant_query_filtered(retriever):
    """An unrelated question should be filtered out by the threshold."""
    results = retriever.retrieve_relevant("How do I bake a chocolate cake?", max_distance=1.0)
    assert len(results) == 0


def test_empty_query_returns_nothing(retriever):
    assert retriever.retrieve("") == []


def test_citation_format(retriever):
    results = retriever.retrieve("exclusions")
    assert "test.pdf" in results[0].citation