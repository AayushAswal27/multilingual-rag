"""LLM client wrapper."""

import logging
from functools import lru_cache

from langchain_groq import ChatGroq

from src.config import LLM_MODEL, LLM_TEMPERATURE, GROQ_API_KEY

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """
    Return the shared LLM client.

    Temperature is 0 because this is an extraction task, not a
    generative one — we want the same answer for the same context
    every time, and no sampling-induced drift from the source text.
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to your .env file."
        )

    logger.info("Initialising LLM: %s", LLM_MODEL)

    return ChatGroq(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=GROQ_API_KEY,
        max_tokens=1024,
    )