"""Tests for app.stages.layout: line clustering, column detection, and
card/region grouping — the fix for cross-column contamination."""

from __future__ import annotations

from app.models.document import format_page_text
from app.stages.layout import cluster_lines, detect_columns, is_layout_noise
from tests.fixtures.layout_pages import (
    adjacent_tables_narrow_gutter_page,
    block,
    dotted_leader_menu_page,
    duplicate_name_ambiguous_page,
    hero_card_title_price_separate_lines_page,
    make_page,
    two_column_flyer_page,
    variant_grid_page,
)


def test_cluster_lines_frozen_anchor_prevents_row_merge():
    """A wide short line must not "reach" into the row above/below it and
    merge two real rows into one (the pre-existing spatial.py bug)."""
    blocks = [
        block("b0", "Veg noodles", 140, 100, 260, 112),
        block("b1", "139/-", 40, 102, 70, 112),
        block("b2", "109/-", 80, 102, 110, 112),
        block("b3", "159/-", 40, 114, 70, 124),
        block("b4", "129/-", 80, 114, 110, 124),
        block("b5", "Egg noodles", 140, 112, 260, 124),
    ]
    lines = cluster_lines(blocks)
    assert len(lines) == 2
    first = sorted(lines[0].blocks, key=lambda b: b.bbox.x0)
    second = sorted(lines[1].blocks, key=lambda b: b.bbox.x0)
    assert [b.text for b in first] == ["139/-", "109/-", "Veg noodles"]
    assert [b.text for b in second] == ["159/-", "129/-", "Egg noodles"]


def test_detect_columns_finds_recurring_gutter():
    blocks = [
        block("a0", "Left A", 40, 100, 300, 120),
        block("b0", "Right A", 560, 100, 860, 120),
        block("a1", "Left B", 40, 130, 260, 150),
        block("b1", "Right B", 560, 130, 720, 150),
        block("a2", "Left C", 40, 160, 280, 180),
        block("b2", "Right C", 560, 160, 900, 180),
    ]
    lines = cluster_lines(blocks)
    bands = detect_columns(lines, page_width=1000, median_line_height=20)
    assert len(bands) == 2
    assert bands[0][0] == 0.0
    assert bands[1][1] == 1000.0
    # boundary sits between the columns, not inside either one
    boundary = bands[0][1]
    assert 300 < boundary < 560


def test_detect_columns_single_column_default():
    blocks = [
        block("a0", "Name", 40, 100, 200, 120),
        block("a1", "Price", 40, 130, 100, 150),
    ]
    lines = cluster_lines(blocks)
    bands = detect_columns(lines, page_width=600, median_line_height=20)
    assert bands == [(0.0, 600.0)]


def test_detect_columns_ignores_dotted_leader_gap():
    """A recurring wide gap must NOT be treated as a column gutter when a
    dotted leader (layout noise) fills it — that's a price-leader within one
    card, not a structural column boundary."""
    page = dotted_leader_menu_page()
    assert len(page.regions) == 2
    for region in page.regions:
        assert region.column_id == 0
    name_block = next(b for b in page.blocks if b.text == "Veg Noodles")
    price_block = next(b for b in page.blocks if b.text == "159")
    assert name_block.region_id == price_block.region_id
    leader_block = next(b for b in page.blocks if "..." in b.text)
    assert leader_block.region_id is None


def test_is_layout_noise_detects_dotted_leaders_and_rules():
    assert is_layout_noise("....................")
    assert is_layout_noise("----------")
    assert is_layout_noise("   ")
    assert not is_layout_noise("Veg Noodles")
    assert not is_layout_noise("159")


def test_build_regions_splits_two_column_flyer_without_cross_contamination():
    page = two_column_flyer_page()
    assert len(page.regions) == 2
    by_col = {r.column_id: r for r in page.regions}
    left = by_col[0]
    right = by_col[1]

    def region_text(region):
        return " ".join(b.text for b in page.blocks if b.region_id == region.region_id)

    left_text = region_text(left)
    right_text = region_text(right)
    assert "Purple Orchid Elegance" in left_text
    assert "1649" in left_text
    assert "White Gerbera Basket" not in left_text
    assert "899" not in left_text

    assert "White Gerbera Basket" in right_text
    assert "899" in right_text
    assert "Purple Orchid Elegance" not in right_text
    assert "1649" not in right_text


def test_build_regions_keeps_hero_title_and_price_in_one_region_despite_separate_lines():
    page = hero_card_title_price_separate_lines_page()
    assert len(page.regions) == 1
    region_id = page.regions[0].region_id
    assert all(b.region_id == region_id for b in page.blocks)


def test_build_regions_keeps_variant_grid_with_its_heading():
    page = variant_grid_page()
    assert len(page.regions) == 1
    assert page.regions[0].kind == "variant_grid"
    assert set(page.regions[0].block_ids) == {"p1-b0", "p1-b1", "p1-b2", "p1-b3"}


def test_build_regions_leaves_duplicate_named_cards_as_separate_regions():
    page = duplicate_name_ambiguous_page()
    assert len(page.regions) == 2
    assert page.regions[0].region_id != page.regions[1].region_id


def test_format_page_text_does_not_join_columns_with_pipe():
    """Direct regression test for the contamination source: the old
    row-based serializer joined cross-column blocks with " | " onto one
    line, which is exactly what let the LLM combine unrelated fields."""
    page = two_column_flyer_page()
    text = format_page_text(page)
    assert " | " not in text
    assert "[region p1-c0-r0]" in text
    assert "[region p1-c1-r0]" in text
    # Each region's block lines appear together, under their own header.
    left_idx = text.index("[region p1-c0-r0]")
    right_idx = text.index("[region p1-c1-r0]")
    orchid_idx = text.index("Purple Orchid Elegance")
    gerbera_idx = text.index("White Gerbera Basket")
    assert left_idx < orchid_idx < right_idx
    assert right_idx < gerbera_idx


def test_adjacent_tables_narrow_gutter_splits_into_separate_regions():
    """Two side-by-side tables with a narrow gutter must NOT merge into a
    single region — the left table's 'save/MRP' column must stay separate
    from the right table's product items."""
    page = adjacent_tables_narrow_gutter_page()

    def block_region(text: str) -> str | None:
        return next((b.region_id for b in page.blocks if b.text == text), None)

    save_region = block_region("save 141")
    mrp_region = block_region("MRP 990")
    kesar_region = block_region("Kesar Pista Kulfi")
    malai_region = block_region("Malai Kulfi Stick")

    assert save_region is not None, "save 141 block should have a region"
    assert kesar_region is not None, "Kesar Pista Kulfi block should have a region"

    # Core assertion: left-table Save cells must NOT share a region with
    # right-table product items.
    assert save_region != kesar_region, (
        f"'save 141' (region {save_region}) and 'Kesar Pista Kulfi' "
        f"(region {kesar_region}) must be in different regions"
    )
    assert mrp_region != kesar_region

    # Right-table items should share a region with each other.
    assert kesar_region == malai_region or kesar_region is not None


def test_make_page_helper_matches_build_regions():
    page = make_page(1, 600, 300, [block("x0", "Solo", 40, 40, 100, 60)])
    assert page.regions
    assert page.blocks[0].region_id == page.regions[0].region_id
