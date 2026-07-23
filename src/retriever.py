"""Retrieve relevant chunks for a query."""

import logging
from dataclasses import dataclass
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import TOP_K
from src.vector_store import load_vector_store

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A retrieved chunk with its distance score and provenance."""

    content: str
    distance: float  # L2 distance; lower means more similar
    source_file: str
    page: Optional[int]

    @property
    def citation(self) -> str:
        page_part = f", page {self.page + 1}" if self.page is not None else ""
        return f"{self.source_file}{page_part}"


class Retriever:
    """Wraps a FAISS store and returns scored, citation-ready chunks."""

    def __init__(self, store: Optional[FAISS] = None, top_k: int = TOP_K):
        self.store = store if store is not None else load_vector_store()
        self.top_k = top_k

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """Return the most similar chunks to the query, nearest first."""
        if not query.strip():
            return []

        k = top_k if top_k is not None else self.top_k
        results = self.store.similarity_search_with_score(query, k=k)

        chunks = [
            RetrievedChunk(
                content=doc.page_content,
                distance=float(score),
                source_file=doc.metadata.get("source_file", "unknown"),
                page=doc.metadata.get("page"),
            )
            for doc, score in results
        ]

        logger.info("Retrieved %d chunk(s) for query: %.60s", len(chunks), query)
        return chunks

    def retrieve_relevant(
        self, query: str, max_distance: float = 1.6, top_k: Optional[int] = None
    ) -> List[RetrievedChunk]:
        """
        Retrieve, then discard chunks beyond the distance threshold.

        An empty list is a meaningful result: the knowledge base has
        nothing relevant, and the LLM should say so rather than
        improvise from weak matches.
        """
        chunks = self.retrieve(query, top_k=top_k)
        kept = [c for c in chunks if c.distance <= max_distance]

        if not kept:
            logger.info("No chunks within distance %.2f", max_distance)

        return kept