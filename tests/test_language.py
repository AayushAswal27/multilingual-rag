"""Tests for language detection."""

from src.language import detect_language


def test_detects_english():
    assert detect_language("Who is eligible for this scheme?") == "English"


def test_detects_hindi():
    assert detect_language("इस योजना के तहत कौन पात्र है और कितना लाभ मिलता है") == "Hindi"


def test_short_text_defaults_to_english():
    """Very short input is unreliable, so it defaults."""
    assert detect_language("hi") == "English"


def test_empty_string_defaults():
    assert detect_language("") == "English"