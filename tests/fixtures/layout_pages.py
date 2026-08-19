"""Synthetic OCR-block page builders for layout/validation tests.

These build `ParsedPage`/`ParsedDocument` objects the same way Stage 1 does
(raw OCR blocks -> `layout.build_regions`), without needing real images or a
live OCR engine.
"""

from __future__ import annotations

from app.models.document import BBox, ParsedDocument, ParsedPage, TextBlock
from app.stages.layout import build_regions


def block(block_id: str, text: str, x0: float, y0: float, x1: float, y1: float) -> TextBlock:
    return TextBlock(block_id=block_id, text=text, bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1))


def make_page(page_number: int, width: int, height: int, blocks: list[TextBlock]) -> ParsedPage:
    page = ParsedPage(page_number=page_number, width=width, height=height, blocks=blocks)
    updated_blocks, regions = build_regions(page)
    page.blocks = updated_blocks
    page.regions = regions
    return page


def make_document(pages: list[ParsedPage], filename: str = "fixture.png") -> ParsedDocument:
    return ParsedDocument(
        source_filename=filename,
        source_path=filename,
        page_count=len(pages),
        pages=pages,
    )


def two_column_flyer_page() -> ParsedPage:
    """A two-column flyer: left card "Purple Orchid Elegance", right card
    "White Gerbera Basket". Both cards have 3-4 stacked lines at matching y
    positions, which is exactly the layout that used to get merged into
    page-wide rows and serialized as `left | right` before this fix.
    """
    blocks = [
        # Left column card
        block("p1-b0", "Purple Orchid Elegance", 40, 100, 300, 120),
        block("p1-b1", "10 imported stems", 40, 130, 260, 150),
        block("p1-b2", "Rs 1649 was Rs 1849", 40, 160, 280, 180),
        block("p1-b3", "BLOOM100", 40, 190, 150, 210),
        # Right column card
        block("p1-b4", "White Gerbera Basket", 560, 100, 860, 120),
        block("p1-b5", "15 flowers", 560, 130, 720, 150),
        block("p1-b6", "Rs 899 free same-day delivery", 560, 160, 900, 180),
    ]
    return make_page(1, 1000, 400, blocks)


def hero_card_title_price_separate_lines_page() -> ParsedPage:
    """Single-column hero card: title and price are on separate OCR lines,
    but close together (one visual card / one region)."""
    blocks = [
        block("p1-b0", "The Classic Twelve", 40, 100, 300, 122),
        block("p1-b1", "Fresh Cut Today", 40, 128, 220, 148),
        block("p1-b2", "Vase not included", 40, 154, 240, 174),
        block("p1-b3", "Rs 1299", 40, 180, 160, 200),
    ]
    return make_page(1, 600, 300, blocks)


def variant_grid_page() -> ParsedPage:
    """One collection heading followed by a horizontal row of 3 variant
    cells — must stay one region so variants attach to the heading."""
    blocks = [
        block("p1-b0", "Chronos Carousel", 40, 100, 260, 122),
        block("p1-b1", "Toddler: Free", 40, 130, 160, 150),
        block("p1-b2", "Child: 999", 200, 130, 300, 150),
        block("p1-b3", "Adult: 1499", 340, 130, 440, 150),
    ]
    return make_page(1, 600, 300, blocks)


def dotted_leader_menu_page() -> ParsedPage:
    """Two single-column menu entries separated by a dotted price leader —
    the leader must not merge or contaminate the two entries."""
    blocks = [
        block("p1-b0", "Veg Noodles", 40, 100, 200, 120),
        block("p1-b1", "159", 400, 100, 440, 120),
        block("p1-b2", "......................", 210, 105, 390, 115),
        block("p1-b3", "Egg Noodles", 40, 250, 200, 270),
        block("p1-b4", "199", 400, 250, 440, 270),
    ]
    return make_page(1, 600, 400, blocks)


def misassociated_metadata_page() -> ParsedPage:
    """Hero card ("The Classic Twelve" with its own descriptive tags) stacked
    above a second card whose title is split across a size-label block
    ("Large") and the actual title block on the same line — mirrors the
    "Large Pink Rose Twenty-Four" / stray "Fresh Cut Today" example."""
    blocks = [
        # Hero card (region A)
        block("p1-b0", "The Classic Twelve", 40, 100, 300, 120),
        block("p1-b1", "Fresh Cut Today", 40, 130, 220, 150),
        block("p1-b2", "Vase not included", 40, 160, 240, 180),
        block("p1-b3", "Rs 2499", 40, 190, 160, 210),
        # Second card (region B) — size label + title on one line
        block("p1-b4", "Large", 40, 280, 100, 300),
        block("p1-b5", "Pink Rose Twenty-Four", 110, 280, 320, 300),
        block("p1-b6", "Rs 999", 40, 310, 160, 330),
    ]
    return make_page(1, 600, 500, blocks)


def adjacent_tables_narrow_gutter_page() -> ParsedPage:
    """Two side-by-side tables with a narrow (~30 px) gutter between the left
    table's last column ("Save") and the right table's first column ("Item").
    Mirrors the real Hocco pamphlet page 2 layout where the page-level column
    detector's gap_threshold (~65 px for a 1632 px page) is wider than the
    true inter-table gap, causing left-table "save/MRP" cells to leak into
    the right table's region."""
    blocks = [
        # --- Left table: "School & Party Combos" (4 columns) ---
        block("p1-b32", "SCHOOL & PARTY COMBOS", 80, 440, 470, 465),
        # Header row
        block("p1-b33", "Combo", 80, 475, 170, 495),
        block("p1-b34", "Includes", 190, 475, 290, 495),
        block("p1-b35", "Combo Price", 310, 475, 430, 495),
        block("p1-b36", "Save", 450, 475, 530, 495),
        # Row 1: School Squad
        block("p1-b37", "School Squad", 80, 505, 180, 525),
        block("p1-b38", "6 cones any flavour", 190, 505, 290, 525),
        block("p1-b39", "749/-", 360, 505, 440, 525),
        block("p1-b40", "save 145", 480, 505, 550, 520),
        block("p1-b41", "MRP 894", 480, 520, 550, 535),
        # Row 2: Office Coolers
        block("p1-b42", "Office Coolers", 80, 545, 180, 565),
        block("p1-b43", "10 cups 100ml", 190, 545, 290, 565),
        block("p1-b48", "849/-", 360, 555, 440, 575),
        block("p1-b49", "save 141", 480, 550, 550, 565),
        block("p1-b50", "MRP 990", 480, 565, 550, 580),
        # Row 3: Birthday Blast
        block("p1-b51", "Birthday Blast", 80, 590, 180, 610),
        block("p1-b52", "cake tub 1kg", 190, 590, 290, 610),
        block("p1-b53", "1499/-", 360, 595, 440, 615),
        block("p1-b54", "save 396", 480, 590, 550, 605),
        block("p1-b55", "MRP 1895", 480, 605, 550, 620),
        # --- Right table: "Pick Your Kulfi & Frozens" (3 columns) ---
        block("p1-b67", "PICK YOUR KULFI & FROZENS", 600, 440, 920, 465),
        # Header row
        block("p1-b68", "Item", 580, 475, 680, 495),
        block("p1-b69", "Size", 720, 475, 790, 495),
        block("p1-b70", "Price", 830, 475, 910, 495),
        # Row 1: Malai Kulfi Stick
        block("p1-b71", "Malai Kulfi Stick", 580, 505, 710, 525),
        block("p1-b72", "1 pc", 720, 505, 770, 525),
        block("p1-b73", "59/-", 840, 505, 910, 525),
        # Row 2: Kesar Pista Kulfi
        block("p1-b74", "Kesar Pista Kulfi", 580, 540, 730, 560),
        block("p1-b75", "1 pc", 720, 540, 770, 560),
        block("p1-b76", "69/-", 840, 540, 910, 560),
        # Row 3: Choco Almond Bar
        block("p1-b77", "Choco Almond Bar", 580, 575, 730, 595),
        block("p1-b78", "1 pc", 720, 575, 770, 595),
        block("p1-b79", "79/-", 840, 575, 910, 595),
    ]
    return make_page(1, 1632, 2176, blocks)


def duplicate_name_ambiguous_page() -> ParsedPage:
    """Same product name printed in two unrelated regions — the anchor
    cannot be resolved unambiguously, so spatial checks are skipped rather
    than guessing."""
    blocks = [
        block("p1-b0", "Combo Deal", 40, 100, 200, 120),
        block("p1-b1", "499", 40, 130, 100, 150),
        block("p1-b2", "Combo Deal", 40, 400, 200, 420),
        block("p1-b3", "599", 40, 430, 100, 450),
    ]
    return make_page(1, 600, 600, blocks)
