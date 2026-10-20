"""Shared, intentionally conservative preprocessing for training and inference."""
import re
import unicodedata

def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", text).strip()

def text_statistics(original: str) -> dict[str, int]:
    normalized = normalize_text(original)
    return {
        "characters": len(original),
        "words": len(normalized.split()) if normalized else 0,
        "normalized_characters": len(normalized),
    }

