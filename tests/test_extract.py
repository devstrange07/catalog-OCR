"""Stage 2 candidate dedupe/merge behavior."""

from __future__ import annotations

import app.stages.extract as extract_stage
from app.models.product import CandidateProduct, PriceTier, Variant


def test_dedupe_candidates_merges_variants_from_duplicate_chunks():
    candidates = [
        CandidateProduct(
            name="Lazy River",
            page=1,
            price=250,
            variants=[Variant(attribute="Age", label="Child", price=250)],
        ),
        CandidateProduct(
            name="Lazy River",
            page=1,
            price=250,
            variants=[Variant(attribute="Age", label="Adult", price=150)],
        ),
    ]

    merged = extract_stage._dedupe_candidates(candidates)

    assert len(merged) == 1
    labels = {v.label for v in merged[0].variants}
    assert labels == {"Child", "Adult"}


def test_dedupe_candidates_fills_missing_price_on_matching_variant():
    candidates = [
        CandidateProduct(
            name="Chronos Carousel",
            page=1,
            variants=[Variant(attribute="Age", label="Toddler", price=None)],
        ),
        CandidateProduct(
            name="Chronos Carousel",
            page=1,
            variants=[Variant(attribute="Age", label="Toddler", price=99)],
        ),
    ]

    merged = extract_stage._dedupe_candidates(candidates)

    assert len(merged) == 1
    assert len(merged[0].variants) == 1
    assert merged[0].variants[0].price == 99


def test_dedupe_candidates_merges_price_tiers():
    candidates = [
        CandidateProduct(
            name="Pro Plan",
            page=1,
            price=29,
            price_tiers=[PriceTier(label="Monthly", price=29)],
        ),
        CandidateProduct(
            name="Pro Plan",
            page=1,
            price=29,
            price_tiers=[PriceTier(label="Annual", price=23.2, billing_cycle="annual")],
        ),
    ]

    merged = extract_stage._dedupe_candidates(candidates)

    assert len(merged) == 1
    labels = {t.label for t in merged[0].price_tiers}
    assert labels == {"Monthly", "Annual"}


def test_dedupe_candidates_keeps_distinct_products_separate():
    candidates = [
        CandidateProduct(name="Giant Wheel", page=1, price=350),
        CandidateProduct(name="Roller Coaster", page=1, price=350),
    ]

    merged = extract_stage._dedupe_candidates(candidates)

    assert len(merged) == 2
