"""Document models — Stage 1 output only. No product fields."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BBox(BaseModel):
    """Axis-aligned box in page pixel coordinates: x0, y0, x1, y1."""

    x0: float
    y0: float
    x1: float
    y1: float


class TextBlock(BaseModel):
    block_id: str
    text: str
    bbox: BBox | None = None
    confidence: float | None = None
    reading_order: int = 0


class ParsedPage(BaseModel):
    page_number: int
    width: int | None = None
    height: int | None = None
    blocks: list[TextBlock] = Field(default_factory=list)
    full_text: str = ""


class ParsedDocument(BaseModel):
    """Structured document from OCR + layout parsing.

    Intentionally has no product schema fields.
    """

    source_filename: str
    source_path: str
    page_count: int
    pages: list[ParsedPage] = Field(default_factory=list)
    structure_notes: list[str] = Field(default_factory=list)
    raw_docling_markdown: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    def text_for_extraction(self, max_chars: int = 24000) -> str:
        parts: list[str] = []
        for page in self.pages:
            parts.append(f"--- Page {page.page_number} ---")
            for block in sorted(page.blocks, key=lambda b: b.reading_order):
                parts.append(f"[{block.block_id}] {block.text}")
            if not page.blocks and page.full_text:
                parts.append(page.full_text)
        text = "\n".join(parts)
        if len(text) > max_chars:
            return text[:max_chars] + "\n...[truncated]"
        return text
