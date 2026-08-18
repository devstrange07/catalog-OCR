"""Spatial utilities for OCR block row reconstruction."""

from __future__ import annotations

from statistics import median

from app.models.document import TextBlock


def _block_height(block: TextBlock) -> float | None:
    if block.bbox is None:
        return None
    return max(0.0, float(block.bbox.y1 - block.bbox.y0))


def _block_center_y(block: TextBlock) -> float | None:
    if block.bbox is None:
        return None
    return float((block.bbox.y0 + block.bbox.y1) / 2.0)


def group_into_rows(blocks: list[TextBlock]) -> list[TextBlock]:
    """Group OCR blocks into visual rows and assign row-major reading order."""
    if not blocks:
        return []

    with_bbox = [b for b in blocks if b.bbox is not None]
    without_bbox = [b for b in blocks if b.bbox is None]
    if not with_bbox:
        out: list[TextBlock] = []
        for idx, block in enumerate(blocks):
            out.append(block.model_copy(update={"reading_order": idx, "row_id": None}))
        return out

    heights = [h for h in (_block_height(b) for b in with_bbox) if h is not None and h > 0]
    median_height = float(median(heights)) if heights else 10.0
    y_tolerance = max(3.0, median_height * 0.6)

    y_sorted = sorted(with_bbox, key=lambda b: (b.bbox.y0, b.bbox.x0))  # type: ignore[union-attr]
    rows: list[list[TextBlock]] = []
    row_minmax: list[tuple[float, float]] = []

    for block in y_sorted:
        center_y = _block_center_y(block)
        if center_y is None:
            continue
        placed = False
        for i, (y_min, y_max) in enumerate(row_minmax):
            if (y_min - y_tolerance) <= center_y <= (y_max + y_tolerance):
                rows[i].append(block)
                h = _block_height(block) or median_height
                new_min = min(y_min, center_y - h / 2.0)
                new_max = max(y_max, center_y + h / 2.0)
                row_minmax[i] = (new_min, new_max)
                placed = True
                break
        if not placed:
            h = _block_height(block) or median_height
            rows.append([block])
            row_minmax.append((center_y - h / 2.0, center_y + h / 2.0))

    normalized: list[TextBlock] = []
    reading_order = 0
    for row_id, row in enumerate(rows):
        for block in sorted(row, key=lambda b: (b.bbox.x0, b.bbox.y0)):  # type: ignore[union-attr]
            normalized.append(
                block.model_copy(
                    update={
                        "reading_order": reading_order,
                        "row_id": row_id,
                    }
                )
            )
            reading_order += 1

    for block in without_bbox:
        normalized.append(
            block.model_copy(
                update={
                    "reading_order": reading_order,
                    "row_id": None,
                }
            )
        )
        reading_order += 1

    return normalized
