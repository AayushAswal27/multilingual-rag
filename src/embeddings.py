"""Multilingual embedding model wrapper."""

import logging
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return the shared embedding model instance.

    Cached so the model is loaded from disk only once per process.
    The same model must be used for both indexing and querying —
    vectors from different models are not comparable.
    """
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )