"""End-to-end RAG pipeline."""

import logging
from dataclasses import dataclass
from typing import List, Optional

from src.language import detect_language
from src.llm import get_llm
from src.prompts import RAG_PROMPT, NO_CONTEXT_MESSAGE, format_context
from src.retriever import Retriever, RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """A complete answer with everything needed to display it."""

    answer: str
    sources: List[RetrievedChunk]
    language: str
    grounded: bool  # False when no relevant context was found


class RAGPipeline:
    """Query -> detect language -> retrieve -> prompt -> answer."""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        max_distance: float = 1.6,
    ):
        self.retriever = retriever if retriever is not None else Retriever()
        self.llm = get_llm()
        self.max_distance = max_distance

    def answer(self, question: str) -> RAGResponse:
        """Answer a question using only the indexed documents."""
        language = detect_language(question)
        chunks = self.retriever.retrieve_relevant(
            question, max_distance=self.max_distance
        )

        if not chunks:
            logger.info("No relevant context; returning canned response")
            code = "hi" if language == "Hindi" else "en"
            return RAGResponse(
                answer=NO_CONTEXT_MESSAGE[code],
                sources=[],
                language=language,
                grounded=False,
            )

        messages = RAG_PROMPT.format_messages(
            language=language,
            context=format_context(chunks),
            question=question,
        )

        response = self.llm.invoke(messages)

        return RAGResponse(
            answer=response.content,
            sources=chunks,
            language=language,
            grounded=True,
        )