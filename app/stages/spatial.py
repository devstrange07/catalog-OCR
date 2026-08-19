"""Spatial utilities for OCR block row reconstruction.

Line clustering itself now lives in `app.stages.layout` (shared with column/
region detection); this module keeps the `group_into_rows` entrypoint for
backward compatibility with the parse stages and existing tests.
"""

from __future__ import annotations

from app.models.document import TextBlock
from app.stages.layout import cluster_lines


def group_into_rows(blocks: list[TextBlock]) -> list[TextBlock]:
    """Group OCR blocks into visual rows and assign row-major reading order."""
    if not blocks:
        return []

    with_bbox = [b for b in blocks if b.bbox is not None]
    without_bbox = [b for b in blocks if b.bbox is None]
    if not with_bbox:
        return [
            block.model_copy(update={"reading_order": idx, "row_id": None})
            for idx, block in enumerate(blocks)
        ]

    lines = cluster_lines(with_bbox)

    normalized: list[TextBlock] = []
    reading_order = 0
    for line in lines:
        for block in sorted(line.blocks, key=lambda b: (b.bbox.x0, b.bbox.y0)):  # type: ignore[union-attr]
            normalized.append(
                block.model_copy(update={"reading_order": reading_order, "row_id": line.line_id})
            )
            reading_order += 1

    for block in without_bbox:
        normalized.append(block.model_copy(update={"reading_order": reading_order, "row_id": None}))
        reading_order += 1

    return normalized
