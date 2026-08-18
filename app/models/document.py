"""Document models — Stage 1 output only. No product fields."""

from __future__ import annotations

from collections import defaultdict
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
    row_id: int | None = None


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
            ordered_blocks = sorted(page.blocks, key=lambda b: b.reading_order)
            grouped: dict[int, list[TextBlock]] = defaultdict(list)
            no_row_blocks: list[TextBlock] = []
            for block in ordered_blocks:
                if block.row_id is None:
                    no_row_blocks.append(block)
                else:
                    grouped[block.row_id].append(block)

            for row_id in sorted(grouped):
                row_blocks = sorted(grouped[row_id], key=lambda b: b.reading_order)
                if row_blocks:
                    lead = f"[{row_blocks[0].block_id}] "
                    row_text = " | ".join(b.text for b in row_blocks)
                    parts.append(lead + row_text)

            for block in no_row_blocks:
                parts.append(f"[{block.block_id}] {block.text}")
            if not page.blocks and page.full_text:
                parts.append(page.full_text)
        text = "\n".join(parts)
        if len(text) > max_chars:
            return text[:max_chars] + "\n...[truncated]"
        return text
