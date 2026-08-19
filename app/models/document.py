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
    line_id: int | None = None
    column_id: int | None = None
    region_id: str | None = None


class LayoutRegion(BaseModel):
    """A candidate visual card/block on a page (product card, heading, or
    variant grid) produced by layout segmentation. Regions are the unit
    within which product fields may be associated with one another."""

    region_id: str
    page: int
    column_id: int | None = None
    bbox: BBox | None = None
    block_ids: list[str] = Field(default_factory=list)
    kind: str = "card"


class ParsedPage(BaseModel):
    page_number: int
    width: int | None = None
    height: int | None = None
    blocks: list[TextBlock] = Field(default_factory=list)
    regions: list[LayoutRegion] = Field(default_factory=list)
    full_text: str = ""


def format_page_text(page: ParsedPage) -> str:
    """Render one page's blocks for LLM extraction.

    Blocks are grouped by `region_id` (one candidate product card/heading per
    region) so fields from different visual regions are never concatenated
    onto the same line. Blocks without a region (no bbox / layout noise /
    markdown-derived documents) fall back to the legacy row-joined format,
    since there is no column risk to guard against in that case.
    """
    parts: list[str] = [f"--- Page {page.page_number} ---"]
    ordered_blocks = sorted(page.blocks, key=lambda b: b.reading_order)

    grouped_by_region: dict[str, list[TextBlock]] = defaultdict(list)
    ungrouped: list[TextBlock] = []
    for block in ordered_blocks:
        if block.region_id:
            grouped_by_region[block.region_id].append(block)
        else:
            ungrouped.append(block)

    if grouped_by_region:
        region_order = sorted(
            grouped_by_region,
            key=lambda rid: min(b.reading_order for b in grouped_by_region[rid]),
        )
        for region_id in region_order:
            parts.append(f"[region {region_id}]")
            for block in sorted(grouped_by_region[region_id], key=lambda b: b.reading_order):
                parts.append(f"  [{block.block_id}] {block.text}")

        row_blocks = [b for b in ungrouped if b.row_id is not None]
        no_row_blocks = [b for b in ungrouped if b.row_id is None]
        grouped_rows: dict[int, list[TextBlock]] = defaultdict(list)
        for block in row_blocks:
            grouped_rows[block.row_id].append(block)
        for row_id in sorted(grouped_rows):
            row = sorted(grouped_rows[row_id], key=lambda b: b.reading_order)
            if row:
                lead = f"[{row[0].block_id}] "
                parts.append(lead + " | ".join(b.text for b in row))
        for block in no_row_blocks:
            parts.append(f"[{block.block_id}] {block.text}")
    else:
        # No layout regions available (markdown-derived or bbox-free document).
        row_blocks = [b for b in ordered_blocks if b.row_id is not None]
        no_row_blocks = [b for b in ordered_blocks if b.row_id is None]
        grouped_rows = defaultdict(list)
        for block in row_blocks:
            grouped_rows[block.row_id].append(block)
        for row_id in sorted(grouped_rows):
            row = sorted(grouped_rows[row_id], key=lambda b: b.reading_order)
            if row:
                lead = f"[{row[0].block_id}] "
                parts.append(lead + " | ".join(b.text for b in row))
        for block in no_row_blocks:
            parts.append(f"[{block.block_id}] {block.text}")

    if not page.blocks and page.full_text:
        parts.append(page.full_text)
    return "\n".join(parts)


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
        text = "\n".join(format_page_text(page) for page in self.pages)
        if len(text) > max_chars:
            return text[:max_chars] + "\n...[truncated]"
        return text
