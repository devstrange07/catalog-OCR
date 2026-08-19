"""Layout segmentation: OCR blocks -> lines -> columns -> card regions.

This is the fix for cross-column contamination: `app.stages.spatial`
previously grouped OCR blocks into page-wide rows with no notion of columns,
so a product name in the left column and a price in the right column ended
up in the same row and were treated as associable. This module adds a
column-aware layer on top of line clustering so downstream stages can
constrain field association to a single visual card/region.

All thresholds are derived from the page's own median OCR line height and
page width rather than fixed pixel constants, so the same logic works across
OCR engines, page sizes, and DPI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from statistics import median

from app.models.document import BBox, LayoutRegion, ParsedDocument, ParsedPage, TextBlock

_NOISE_RE = re.compile(r"^[.\-_·•\s]{3,}$")


def _height(block: TextBlock) -> float | None:
    if block.bbox is None:
        return None
    return max(0.0, float(block.bbox.y1 - block.bbox.y0))


def _center_y(block: TextBlock) -> float | None:
    if block.bbox is None:
        return None
    return float((block.bbox.y0 + block.bbox.y1) / 2.0)


def normalize_bbox(
    bbox: BBox | None, page_width: float | None, page_height: float | None
) -> tuple[float, float, float, float] | None:
    """Return bbox coordinates as page-relative fractions (0..1).

    Returns None when the bbox or page dimensions are unavailable, so callers
    can fall back to skipping geometry-dependent logic rather than dividing
    by an unknown/zero page size.
    """
    if bbox is None or not page_width or not page_height:
        return None
    return (
        bbox.x0 / page_width,
        bbox.y0 / page_height,
        bbox.x1 / page_width,
        bbox.y1 / page_height,
    )


def is_layout_noise(text: str) -> bool:
    """True for dotted leaders, rules, and other decorative/punctuation-only
    OCR detections that should not anchor or widen a product region."""
    stripped = (text or "").strip()
    if not stripped:
        return True
    if _NOISE_RE.match(stripped):
        return True
    compact = stripped.replace(" ", "")
    if len(compact) >= 3 and len(set(compact)) == 1 and not compact[0].isalnum():
        return True
    return False


@dataclass
class Line:
    """One visual text line: OCR blocks sharing a baseline band."""

    line_id: int
    blocks: list[TextBlock]
    y_min: float
    y_max: float

    @property
    def center_y(self) -> float:
        return (self.y_min + self.y_max) / 2.0

    @property
    def x_min(self) -> float:
        return min(b.bbox.x0 for b in self.blocks if b.bbox is not None)

    @property
    def x_max(self) -> float:
        return max(b.bbox.x1 for b in self.blocks if b.bbox is not None)

    @property
    def center_x(self) -> float:
        return (self.x_min + self.x_max) / 2.0


def cluster_lines(blocks: list[TextBlock]) -> list[Line]:
    """Group OCR blocks into visual lines by y-overlap / baseline proximity.

    Membership is tested against each line's frozen anchor center (the
    center-y of the first block placed in the line), not a min/max window
    that grows as more blocks join. A growing window lets a wide/short OCR
    line "reach" into the row above or below it and merge two real rows into
    one — that was the pre-existing bug in `spatial.group_into_rows`.
    """
    with_bbox = [b for b in blocks if b.bbox is not None]
    if not with_bbox:
        return []

    heights = [h for h in (_height(b) for b in with_bbox) if h is not None and h > 0]
    median_height = float(median(heights)) if heights else 10.0
    y_tolerance = max(2.0, median_height * 0.5)

    y_sorted = sorted(with_bbox, key=lambda b: (b.bbox.y0, b.bbox.x0))  # type: ignore[union-attr]
    lines: list[Line] = []
    anchors: list[float] = []

    for block in y_sorted:
        cy = _center_y(block)
        if cy is None:
            continue
        placed = False
        for i, anchor in enumerate(anchors):
            if abs(cy - anchor) <= y_tolerance:
                lines[i].blocks.append(block)
                h = _height(block) or median_height
                lines[i].y_min = min(lines[i].y_min, cy - h / 2.0)
                lines[i].y_max = max(lines[i].y_max, cy + h / 2.0)
                placed = True
                break
        if not placed:
            h = _height(block) or median_height
            lines.append(Line(line_id=len(lines), blocks=[block], y_min=cy - h / 2.0, y_max=cy + h / 2.0))
            anchors.append(cy)

    lines.sort(key=lambda ln: ln.center_y)
    for idx, line in enumerate(lines):
        line.line_id = idx
    return lines


def _gap_is_leader_filled(
    a: TextBlock, b: TextBlock, line: Line, noise_blocks: list[TextBlock]
) -> bool:
    """True when a dotted leader / decorative separator (already excluded
    from clustering as layout noise) sits inside the gap between `a` and
    `b`. This is what distinguishes a real column gutter from a "name
    ⋯⋯⋯ price" price-leader gap on a single card: both look identical as a
    recurring wide x-gap, but only the leader case has its own OCR
    detection filling the space.
    """
    for noise in noise_blocks:
        if noise.bbox is None:
            continue
        if noise.bbox.y1 < line.y_min or noise.bbox.y0 > line.y_max:
            continue
        if noise.bbox.x0 >= a.bbox.x1 - 1 and noise.bbox.x1 <= b.bbox.x0 + 1:  # type: ignore[union-attr]
            return True
    return False


def detect_columns(
    lines: list[Line],
    page_width: float | None,
    median_line_height: float,
    noise_blocks: list[TextBlock] | None = None,
    *,
    threshold_width: float | None = None,
) -> list[tuple[float, float]]:
    """Detect column boundaries as x-gaps that recur across most of the
    lines that span them — a real printed column gutter shows up at nearly
    the same x position on almost every line that crosses it, whereas an
    incidental in-card gap (e.g. name-to-price spacing) does not recur at a
    consistent x across the page. Returns a list of (x0, x1) column bands
    covering the full page width; a single-column page returns one band.

    The minimum gap width worth considering scales with page width and OCR
    line height rather than a fixed pixel constant or a self-referential
    "median gap" (which degenerates when every line has exactly one gap —
    e.g. a strict two-column grid — leaving nothing smaller to contrast
    against).

    When *threshold_width* is provided it overrides ``page_width`` only for
    the gap-threshold formula, allowing callers to run a region-local
    re-check at a tighter scale while keeping absolute-coordinate band
    boundaries.
    """
    if not lines:
        return [(0.0, page_width or 1.0)]

    width = float(page_width) if page_width else max((ln.x_max for ln in lines), default=1.0)
    if width <= 0:
        width = max((ln.x_max for ln in lines), default=1.0)

    noise_blocks = noise_blocks or []
    gap_records: list[tuple[Line, float, float]] = []
    for line in lines:
        ordered = sorted((b for b in line.blocks if b.bbox is not None), key=lambda b: b.bbox.x0)
        for a, b in zip(ordered, ordered[1:]):
            gap = b.bbox.x0 - a.bbox.x1
            if gap <= 0:
                continue
            if _gap_is_leader_filled(a, b, line, noise_blocks):
                continue
            gap_records.append((line, (a.bbox.x1 + b.bbox.x0) / 2.0, gap))

    if not gap_records:
        return [(0.0, width)]

    tw = float(threshold_width) if threshold_width is not None else width
    local_mode = threshold_width is not None
    mlh_factor = 1.2 if local_mode else 3.0
    gap_threshold = max(tw * 0.04, median_line_height * mlh_factor)
    strong = [(line, xm, g) for line, xm, g in gap_records if g >= gap_threshold]
    if not strong:
        return [(0.0, width)]

    # Tolerance for how much a gutter's x-position may drift line to line —
    # a fraction of page width, not the gap width itself (gap width and
    # positional jitter are different quantities; conflating them made the
    # tolerance balloon for wide gutters).
    x_tolerance = max(width * 0.05, 1.0)
    strong.sort(key=lambda t: t[1])
    clusters: list[list[tuple[Line, float, float]]] = []
    for item in strong:
        if clusters and abs(item[1] - clusters[-1][-1][1]) <= x_tolerance:
            clusters[-1].append(item)
        else:
            clusters.append([item])

    boundaries: list[float] = []
    for cluster in clusters:
        xm_avg = sum(c[1] for c in cluster) / len(cluster)
        supporting_lines = {id(c[0]) for c in cluster}
        eligible = 0
        for line in lines:
            xs = [b.bbox.x0 for b in line.blocks if b.bbox is not None]
            xe = [b.bbox.x1 for b in line.blocks if b.bbox is not None]
            if not xs:
                continue
            if min(xs) < xm_avg < max(xe):
                eligible += 1
        if eligible == 0:
            continue
        support = len(supporting_lines) / eligible
        # A true column gutter recurs across most of the page height, not
        # just a couple of incidentally-aligned in-card gaps.
        if support >= 0.5 and len(supporting_lines) >= 2:
            boundaries.append(xm_avg)

    if not boundaries:
        return [(0.0, width)]

    boundaries.sort()
    bounds = [0.0, *boundaries, width]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def _assign_column(cx: float, bands: list[tuple[float, float]]) -> int:
    if not bands:
        return 0
    for idx, (x0, x1) in enumerate(bands):
        if idx == len(bands) - 1:
            if cx >= x0:
                return idx
        elif x0 <= cx < x1:
            return idx
    return len(bands) - 1


def _group_regions_in_column(
    lines: list[Line],
    median_line_height: float,
    page_number: int,
    column_id: int,
) -> list[tuple[LayoutRegion, list[Line]]]:
    """Split one column's lines into card/heading regions on vertical gaps
    that are large relative to the page's own line spacing."""
    if not lines:
        return []

    gap_threshold = max(median_line_height * 1.6, 4.0)
    groups: list[list[Line]] = [[lines[0]]]
    for prev, curr in zip(lines, lines[1:]):
        gap = curr.y_min - prev.y_max
        if gap > gap_threshold:
            groups.append([curr])
        else:
            groups[-1].append(curr)

    result: list[tuple[LayoutRegion, list[Line]]] = []
    for idx, group in enumerate(groups):
        block_ids = [b.block_id for line in group for b in line.blocks]
        bbox = BBox(
            x0=min(ln.x_min for ln in group),
            y0=min(ln.y_min for ln in group),
            x1=max(ln.x_max for ln in group),
            y1=max(ln.y_max for ln in group),
        )
        # A line with several side-by-side blocks under a shared heading is
        # the horizontal-variant-grid case; keep it as one region so variants
        # stay attached to their collection heading instead of being split.
        kind = "variant_grid" if any(len(ln.blocks) >= 3 for ln in group) else "card"
        region = LayoutRegion(
            region_id=f"p{page_number}-c{column_id}-r{idx}",
            page=page_number,
            column_id=column_id,
            bbox=bbox,
            block_ids=block_ids,
            kind=kind,
        )
        result.append((region, group))
    return result


def _resplit_region(
    region: LayoutRegion,
    region_lines: list[Line],
    median_line_height: float,
    noise_blocks: list[TextBlock],
    page_number: int,
    column_id: int,
    base_region_idx: int,
) -> list[tuple[LayoutRegion, list[Line]]]:
    """Re-run column detection *within* a single region using the region's
    own bbox width for threshold scaling.  If an internal recurring gutter is
    found, split the region; otherwise return it unchanged.

    This catches narrow inter-table gutters that the page-level pass missed
    because its gap_threshold was scaled to the full page width.
    """
    if len(region_lines) < 2:
        return [(region, region_lines)]
    if region.bbox is None:
        return [(region, region_lines)]

    region_width = region.bbox.x1 - region.bbox.x0
    if region_width <= 0:
        return [(region, region_lines)]

    local_noise = [
        n for n in noise_blocks
        if n.bbox is not None
        and n.bbox.y1 >= region.bbox.y0
        and n.bbox.y0 <= region.bbox.y1
    ]

    local_bands = detect_columns(
        region_lines,
        page_width=None,
        median_line_height=median_line_height,
        noise_blocks=local_noise,
        threshold_width=region_width,
    )

    if len(local_bands) <= 1:
        return [(region, region_lines)]

    sub_columns: dict[int, list[TextBlock]] = {}
    for line in region_lines:
        for blk in line.blocks:
            if blk.bbox is None:
                continue
            cx = (blk.bbox.x0 + blk.bbox.x1) / 2.0
            col = _assign_column(cx, local_bands)
            sub_columns.setdefault(col, []).append(blk)

    result: list[tuple[LayoutRegion, list[Line]]] = []
    for sub_idx, sub_col_id in enumerate(sorted(sub_columns)):
        sub_lines = cluster_lines(sub_columns[sub_col_id])
        if not sub_lines:
            continue
        sub_groups = _group_regions_in_column(
            sub_lines, median_line_height, page_number, column_id,
        )
        for grp_idx, (sub_region, grp_lines) in enumerate(sub_groups):
            sub_region.region_id = (
                f"p{page_number}-c{column_id}-r{base_region_idx}s{sub_idx}g{grp_idx}"
            )
            result.append((sub_region, grp_lines))

    return result if result else [(region, region_lines)]


def build_regions(page: ParsedPage) -> tuple[list[TextBlock], list[LayoutRegion]]:
    """Segment a page's OCR blocks into lines, columns, and card regions.

    Returns `(updated_blocks, regions)`. Blocks are returned as copies with
    `line_id`, `column_id`, and `region_id` populated. Layout-noise blocks
    (dotted leaders, rules) and blocks with no bbox are returned unchanged
    with `region_id=None` — downstream spatial checks skip fields that can't
    be resolved to a region rather than guessing.
    """
    blocks = list(page.blocks)
    with_bbox = [b for b in blocks if b.bbox is not None]
    without_bbox = [b for b in blocks if b.bbox is None]
    if not with_bbox:
        return blocks, []

    noise_ids = {b.block_id for b in with_bbox if is_layout_noise(b.text)}
    clusterable = [b for b in with_bbox if b.block_id not in noise_ids]
    noise_blocks = [b for b in with_bbox if b.block_id in noise_ids]
    if not clusterable:
        return blocks, []

    # Whole-page lines are only used to detect column gutters (gap analysis
    # needs a line's full set of same-row blocks, across columns). A single
    # such line can legitimately contain blocks from two different columns
    # (e.g. a name in the left column and a price in the right column at the
    # same y) — line-level column assignment would incorrectly lump both
    # into one band. So columns are assigned per-BLOCK by its own x-center,
    # and lines are then re-clustered independently within each column.
    page_lines = cluster_lines(clusterable)
    heights = [h for h in (_height(b) for b in clusterable) if h is not None and h > 0]
    median_line_height = float(median(heights)) if heights else 10.0
    bands = detect_columns(page_lines, page.width, median_line_height, noise_blocks=noise_blocks)

    blocks_by_column: dict[int, list[TextBlock]] = {}
    for b in clusterable:
        cx = (b.bbox.x0 + b.bbox.x1) / 2.0  # type: ignore[union-attr]
        col = _assign_column(cx, bands)
        blocks_by_column.setdefault(col, []).append(b)

    initial_regions: list[tuple[LayoutRegion, list[Line]]] = []
    for col_id in sorted(blocks_by_column):
        col_lines = cluster_lines(blocks_by_column[col_id])
        initial_regions.extend(
            _group_regions_in_column(col_lines, median_line_height, page.page_number, col_id)
        )

    ordered_regions: list[tuple[LayoutRegion, list[Line]]] = []
    for idx, (region, region_lines) in enumerate(initial_regions):
        ordered_regions.extend(
            _resplit_region(
                region,
                region_lines,
                median_line_height,
                noise_blocks,
                page.page_number,
                region.column_id or 0,
                idx,
            )
        )

    block_layout: dict[str, tuple[str, int | None, int]] = {}
    for region, region_lines in ordered_regions:
        for line in region_lines:
            for block in line.blocks:
                block_layout[block.block_id] = (region.region_id, region.column_id, line.line_id)

    updated: list[TextBlock] = []
    reading_order = 0
    for region, region_lines in ordered_regions:
        for line in sorted(region_lines, key=lambda ln: ln.center_y):
            for block in sorted(line.blocks, key=lambda b: b.bbox.x0):  # type: ignore[union-attr]
                region_id, column_id, line_id = block_layout[block.block_id]
                updated.append(
                    block.model_copy(
                        update={
                            "reading_order": reading_order,
                            "row_id": line_id,
                            "line_id": line_id,
                            "column_id": column_id,
                            "region_id": region_id,
                        }
                    )
                )
                reading_order += 1

    handled_ids = {b.block_id for b in updated}
    leftover = [b for b in with_bbox if b.block_id not in handled_ids] + without_bbox
    for block in leftover:
        updated.append(
            block.model_copy(
                update={
                    "reading_order": reading_order,
                    "row_id": None,
                    "line_id": None,
                    "column_id": None,
                    "region_id": None,
                }
            )
        )
        reading_order += 1

    regions = [r for r, _ in ordered_regions]
    return updated, regions


@dataclass
class PageIndex:
    """Lookup structures shared by Stage 3 validation and semantic checks."""

    blocks_by_page: dict[int, list[TextBlock]] = field(default_factory=dict)
    regions_by_page: dict[int, list[LayoutRegion]] = field(default_factory=dict)
    region_lookup: dict[tuple[int, str], LayoutRegion] = field(default_factory=dict)
    blocks_by_region: dict[str, list[TextBlock]] = field(default_factory=dict)
    page_dims: dict[int, tuple[float, float]] = field(default_factory=dict)


def build_page_index(parsed: ParsedDocument | None) -> PageIndex:
    index = PageIndex()
    if parsed is None:
        return index
    for page in parsed.pages:
        index.blocks_by_page[page.page_number] = list(page.blocks)
        index.regions_by_page[page.page_number] = list(page.regions)
        for region in page.regions:
            index.region_lookup[(page.page_number, region.region_id)] = region
        for block in page.blocks:
            if block.region_id:
                index.blocks_by_region.setdefault(block.region_id, []).append(block)
        if page.width and page.height:
            index.page_dims[page.page_number] = (float(page.width), float(page.height))
    return index


def bbox_to_list(bbox: BBox | None) -> list[float] | None:
    return [bbox.x0, bbox.y0, bbox.x1, bbox.y1] if bbox is not None else None


def bbox_center(bbox: BBox) -> tuple[float, float]:
    return ((bbox.x0 + bbox.x1) / 2.0, (bbox.y0 + bbox.y1) / 2.0)


def page_diagonal(page_dims: dict[int, tuple[float, float]], page: int | None) -> float | None:
    if page is None:
        return None
    dims = page_dims.get(page)
    if not dims:
        return None
    w, h = dims
    return (w**2 + h**2) ** 0.5


def proximity_confidence(a: BBox | None, b: BBox | None, diagonal: float | None) -> float:
    """Rough 0..1 diagnostic: how spatially plausible a cross-region match is,
    based on distance relative to the page diagonal. Purely informational —
    it does not influence whether a field is treated as resolved."""
    if a is None or b is None or not diagonal:
        return 0.2
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    return round(max(0.0, min(1.0, 1.0 - dist / diagonal)), 3)
