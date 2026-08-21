"""Shared upload/URL file-type allowlists and MIME maps."""

from __future__ import annotations

from app.models.pipeline import ParseChannel

IMAGE_PDF_SUFFIXES: set[str] = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

# Gemini-native text MIME types (not Office Open XML — xlsx/docx unsupported).
GEMINI_TEXT_SUFFIXES: set[str] = {
    ".csv",
    ".txt",
    ".md",
    ".html",
    ".htm",
    ".xml",
    ".json",
    ".rtf",
}

_MIME_BY_SUFFIX: dict[str, str] = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".csv": "text/csv",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".html": "text/html",
    ".htm": "text/html",
    ".xml": "text/xml",
    ".json": "application/json",
    ".rtf": "text/rtf",
}


def mime_for_suffix(suffix: str) -> str:
    key = suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
    mime = _MIME_BY_SUFFIX.get(key)
    if not mime:
        raise ValueError(f"Unsupported file type: {key}")
    return mime


def allowed_suffixes_for_channel(channel: ParseChannel | str) -> set[str]:
    if isinstance(channel, ParseChannel):
        value = channel
    else:
        value = ParseChannel(str(channel).strip().lower())
    if value == ParseChannel.gemini:
        return IMAGE_PDF_SUFFIXES | GEMINI_TEXT_SUFFIXES
    return set(IMAGE_PDF_SUFFIXES)
