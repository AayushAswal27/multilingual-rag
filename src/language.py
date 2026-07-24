"""Detect the language of a query."""

import logging

from langdetect import detect, LangDetectException

logger = logging.getLogger(__name__)

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ur": "Urdu",
    "or": "Odia",
    "ne": "Nepali",
}

DEFAULT_LANGUAGE = "English"


def detect_language(text: str) -> str:
    """
    Return the human-readable language name of the text.

    Falls back to English on failure. langdetect is unreliable on
    very short inputs, so queries under 10 characters are not
    trusted and default instead.
    """
    if len(text.strip()) < 10:
        return DEFAULT_LANGUAGE

    try:
        code = detect(text)
    except LangDetectException:
        logger.warning("Language detection failed; defaulting to English")
        return DEFAULT_LANGUAGE

    name = LANGUAGE_NAMES.get(code, DEFAULT_LANGUAGE)
    logger.info("Detected language: %s (%s)", name, code)
    return name