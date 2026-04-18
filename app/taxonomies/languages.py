"""Language-name normalization.

The source DB stores multiple spellings for the same language, e.g.
"English", "EN", "Inglés". We normalize to ISO-639-1 alpha-2 codes.
"""
from __future__ import annotations

LANGUAGE_ALIASES: dict[str, str] = {
    # English
    "english": "en",
    "en": "en",
    "inglés": "en",
    "ingles": "en",
    # Arabic
    "arabic": "ar",
    "ar": "ar",
    "العربية": "ar",
    # French
    "french": "fr",
    "fr": "fr",
    "français": "fr",
    "francais": "fr",
    # Spanish
    "spanish": "es",
    "es": "es",
    "español": "es",
    "espanol": "es",
    # German
    "german": "de",
    "de": "de",
    "deutsch": "de",
    # Hindi
    "hindi": "hi",
    "hi": "hi",
    # Turkish
    "turkish": "tr",
    "tr": "tr",
    "türkçe": "tr",
    "turkce": "tr",
    # Chinese (Mandarin)
    "chinese": "zh",
    "mandarin": "zh",
    "zh": "zh",
    "中文": "zh",
    # Portuguese
    "portuguese": "pt",
    "pt": "pt",
    "português": "pt",
    "portugues": "pt",
    # Italian
    "italian": "it",
    "it": "it",
    "italiano": "it",
    # Japanese
    "japanese": "ja",
    "ja": "ja",
    "日本語": "ja",
    # Russian
    "russian": "ru",
    "ru": "ru",
    "русский": "ru",
}


def normalize(name: str) -> str:
    """Return ISO-639-1 code, or the lowercased input if unknown."""
    key = (name or "").strip().lower()
    return LANGUAGE_ALIASES.get(key, key)
