"""Stage 2 candidate dedupe/merge behavior."""

from __future__ import annotations

import app.stages.extract as extract_stage
from app.models.product import CandidateProduct, PriceVariant


def test_dedupe_candidates_merges_variants_from_duplicate_chunks():
    candidates = [
        CandidateProduct(
            name="Lazy River",
            page=1,
            price=250,
            price_variants=[PriceVariant(label="Child", price=250)],
        ),
        CandidateProduct(
            name="Lazy River",
            page=1,
            price=250,
            price_variants=[PriceVariant(label="Adult", price=150)],
        ),
    ]

    merged = extract_stage._dedupe_candidates(candidates)

    assert len(merged) == 1
    labels = {v.label for v in merged[0].price_variants}
    assert labels == {"Child", "Adult"}


def test_dedupe_candidates_fills_missing_price_on_matching_variant():
    candidates = [
        CandidateProduct(
            name="Chronos Carousel",
            page=1,
            price_variants=[PriceVariant(label="Toddler", price=None)],
        ),
        CandidateProduct(
            name="Chronos Carousel",
            page=1,
            price_variants=[PriceVariant(label="Toddler", price=99)],
        ),
    ]

    merged = extract_stage._dedupe_candidates(candidates)

    assert len(merged) == 1
    assert len(merged[0].price_variants) == 1
    assert merged[0].price_variants[0].price == 99


def test_dedupe_candidates_keeps_distinct_products_separate():
    candidates = [
        CandidateProduct(name="Giant Wheel", page=1, price=350),
        CandidateProduct(name="Roller Coaster", page=1, price=350),
    ]

    merged = extract_stage._dedupe_candidates(candidates)

    assert len(merged) == 2
