from app.models.product import CandidateProduct, IssueSeverity, PriceVariant, ProductStatus
from app.stages.validate import validate_products
from tests.fixtures.layout_pages import (
    duplicate_name_ambiguous_page,
    hero_card_title_price_separate_lines_page,
    make_document,
    misassociated_metadata_page,
    two_column_flyer_page,
    variant_grid_page,
)


def test_validate_required_name_and_price():
    records = validate_products(
        [
            CandidateProduct(
                name="Tea",
                price="$12.50",
                currency="usd",
                price_variants=[
                    PriceVariant(
                        label="Regular",
                        price="12.50",
                        details="Stay, Breakfast, Sightseeing",
                    )
                ],
            ),
            CandidateProduct(name="", sku="X"),
            CandidateProduct(name="Tea", price="$12.50", currency="usd"),
        ]
    )
    assert records[0].status == ProductStatus.validated
    assert records[0].schema_valid is True
    assert records[0].extraction_valid is True
    assert records[0].price == 12.5
    assert records[0].currency == "USD"
    assert records[0].price_variants is not None
    assert records[0].price_variants[0].price == 12.5
    assert records[0].price_variants[0].details == "Stay, Breakfast, Sightseeing"
    assert records[1].status == ProductStatus.invalid
    assert records[1].schema_valid is False
    assert any("duplicate" in w.lower() for w in records[0].warnings)


def test_validate_sentinel_variant_price_does_not_reject():
    records = validate_products(
        [
            CandidateProduct(
                name="Chronos Carousel",
                price=999,
                price_variants=[
                    PriceVariant(label="Toddler", price="FREE", details="Below 3 Years"),
                    PriceVariant(label="Child", price=999, details="3 - 12 Years"),
                ],
            )
        ]
    )
    record = records[0]
    assert record.status == ProductStatus.validated
    assert record.validation_errors == []
    assert record.price_variants[0].price is None
    assert record.price_variants[0].available == "free"
    assert record.price_variants[1].price == 999.0


def test_validate_unparseable_variant_price_warns_not_rejects():
    records = validate_products(
        [
            CandidateProduct(
                name="Lockers",
                price=90,
                price_variants=[
                    PriceVariant(label="Large", price="???"),
                    PriceVariant(label="Small", price=90),
                ],
            )
        ]
    )
    record = records[0]
    assert record.status == ProductStatus.validated
    assert record.validation_errors == []
    assert any("Unparseable variant price" in w for w in record.warnings)
    assert any("Unparseable variant price" in w for w in record.price_variants[0].warnings)


def test_validate_empty_shell_with_only_price_variants_is_not_rejected():
    records = validate_products(
        [
            CandidateProduct(
                name="",
                price_variants=[PriceVariant(label="Adult", price=200)],
            )
        ]
    )
    assert records[0].status == ProductStatus.invalid
    assert any(i.code == "MISSING_REQUIRED_FIELD" for i in records[0].validation_errors)
    assert not any(i.code == "EMPTY_PRODUCT_SHELL" for i in records[0].validation_errors)


def test_validate_two_column_flyer_keeps_each_product_on_its_own_price():
    """Field Notes-style cross-column contamination case: two products side
    by side must each resolve to their own region's price/description with
    no swapping, and remain `validated` since there's no ambiguity."""
    page = two_column_flyer_page()
    parsed = make_document([page])

    candidates = [
        CandidateProduct(
            name="Purple Orchid Elegance",
            page=1,
            price=1649,
            description="10 imported stems, BLOOM100",
            region_id="p1-c0-r0",
        ),
        CandidateProduct(
            name="White Gerbera Basket",
            page=1,
            price=899,
            description="15 flowers, free same-day delivery",
            region_id="p1-c1-r0",
        ),
    ]
    records = validate_products(candidates, parsed=parsed)

    orchid, gerbera = records
    assert orchid.status == ProductStatus.validated
    assert orchid.price == 1649
    assert orchid.region_id == "p1-c0-r0"
    assert orchid.validation_errors == []

    assert gerbera.status == ProductStatus.validated
    assert gerbera.price == 899
    assert gerbera.region_id == "p1-c1-r0"
    assert gerbera.validation_errors == []

    # Provenance recorded and scoped to each product's own region/bbox.
    orchid_price_prov = next(p for p in orchid.provenance if p.field == "price")
    gerbera_price_prov = next(p for p in gerbera.provenance if p.field == "price")
    assert orchid_price_prov.region_id == "p1-c0-r0"
    assert gerbera_price_prov.region_id == "p1-c1-r0"
    assert orchid_price_prov.bbox != gerbera_price_prov.bbox


def test_validate_flags_cross_region_price_as_needs_review_and_unresolved():
    """A naive/contaminated extraction — Purple Orchid's candidate reports
    the *other* column's price — must not be trusted across the region
    boundary: the field is nulled, flagged, and the record is needs_review."""
    page = two_column_flyer_page()
    parsed = make_document([page])

    candidates = [
        CandidateProduct(
            name="Purple Orchid Elegance",
            page=1,
            price=899,  # wrong: this is White Gerbera Basket's price
            region_id="p1-c0-r0",
        ),
    ]
    records = validate_products(candidates, parsed=parsed)
    record = records[0]

    assert record.status == ProductStatus.needs_review
    assert record.schema_valid is True
    assert record.extraction_valid is False
    assert record.price is None  # preserved as unresolved, not assigned cross-region
    issue = next(i for i in record.validation_errors if i.code == "SPATIAL_ASSOCIATION_AMBIGUOUS")
    assert issue.severity == IssueSeverity.review
    assert issue.field == "price"
    assert issue.region_id == "p1-c0-r0"
    assert issue.field_bbox is not None
    assert issue.product_bbox is not None


def test_validate_promo_code_out_of_region_flags_contamination():
    """A promo code that only appears in a *different* product's region is
    cross-record contamination, not this product's own attribute."""
    page = two_column_flyer_page()
    parsed = make_document([page])

    candidates = [
        CandidateProduct(name="Purple Orchid Elegance", page=1, price=1649, region_id="p1-c0-r0"),
        CandidateProduct(
            name="White Gerbera Basket",
            page=1,
            price=899,
            description="BLOOM100",  # actually belongs to Purple Orchid's region
            region_id="p1-c1-r0",
        ),
    ]
    records = validate_products(candidates, parsed=parsed)
    gerbera = records[1]

    assert gerbera.status == ProductStatus.needs_review
    assert any(i.code == "PROMO_CODE_OUT_OF_REGION" for i in gerbera.validation_errors)


def test_validate_dedupes_equivalent_product_and_variant_level_issues():
    """A default variant that mirrors an already-contaminated top-level
    price must not produce a second, duplicate issue."""
    page = two_column_flyer_page()
    parsed = make_document([page])

    candidates = [
        CandidateProduct(
            name="Purple Orchid Elegance",
            page=1,
            price=899,
            region_id="p1-c0-r0",
            price_variants=[PriceVariant(label="Standard", price=899, is_default=True)],
        ),
    ]
    records = validate_products(candidates, parsed=parsed)
    record = records[0]

    spatial_issues = [i for i in record.validation_errors if i.code == "SPATIAL_ASSOCIATION_AMBIGUOUS"]
    assert len(spatial_issues) == 1
    assert record.status == ProductStatus.needs_review


def test_validate_hero_card_title_and_price_on_separate_lines_stays_validated():
    """Product title and price intentionally on separate OCR lines within
    one visual card must resolve cleanly with no spatial warning."""
    page = hero_card_title_price_separate_lines_page()
    parsed = make_document([page])

    candidates = [
        CandidateProduct(
            name="The Classic Twelve",
            page=1,
            price=1299,
            description="Fresh Cut Today, Vase not included",
            region_id="p1-c0-r0",
        )
    ]
    records = validate_products(candidates, parsed=parsed)
    record = records[0]

    assert record.status == ProductStatus.validated
    assert record.validation_errors == []
    assert record.price == 1299
    assert not any("row" in w.lower() for w in record.warnings)


def test_validate_horizontal_variant_grid_resolves_all_variants_in_one_region():
    page = variant_grid_page()
    parsed = make_document([page])

    candidates = [
        CandidateProduct(
            name="Chronos Carousel",
            page=1,
            price=999,
            region_id="p1-c0-r0",
            price_variants=[
                PriceVariant(label="Toddler", price="Free"),
                PriceVariant(label="Child", price=999, is_default=True),
                PriceVariant(label="Adult", price=1499),
            ],
        )
    ]
    records = validate_products(candidates, parsed=parsed)
    record = records[0]

    assert record.status == ProductStatus.validated
    assert record.validation_errors == []
    assert record.price_variants[1].price == 999
    assert record.price_variants[2].price == 1499


def test_validate_ambiguous_duplicate_name_anchor_is_skipped_not_guessed():
    """When a name matches two unrelated regions and no region_id is given,
    the anchor is unresolved — values pass through untouched rather than
    being guessed against the wrong card."""
    page = duplicate_name_ambiguous_page()
    parsed = make_document([page])

    candidates = [CandidateProduct(name="Combo Deal", page=1, price=499)]
    records = validate_products(candidates, parsed=parsed)
    record = records[0]

    assert record.region_id is None
    assert record.price == 499
    assert record.status == ProductStatus.validated


def test_validate_geometry_free_document_never_flags_spatial_review():
    """A markdown/HTML-crawled document has no bbox data at all; spatial
    checks must be skipped cleanly rather than producing false positives."""
    from app.models.document import ParsedPage, TextBlock

    parsed = make_document(
        [
            ParsedPage(
                page_number=1,
                blocks=[
                    TextBlock(block_id="p1-b0", text="Widget Pro", reading_order=0),
                    TextBlock(block_id="p1-b1", text="SKU: W-1", reading_order=1),
                ],
                full_text="Widget Pro\nSKU: W-1",
            )
        ]
    )
    candidates = [CandidateProduct(name="Widget Pro", page=1, price=19.99, sku="W-1")]
    records = validate_products(candidates, parsed=parsed)
    record = records[0]

    assert record.region_id is None
    assert record.status == ProductStatus.validated
    assert record.validation_errors == []


def test_validate_name_size_concatenation_and_borrowed_description_are_flagged():
    """Integration case mirroring the "Large Pink Rose Twenty-Four" example:
    a size label folded into the name, plus descriptive tags borrowed from a
    different card's region."""
    page = misassociated_metadata_page()
    parsed = make_document([page])

    candidates = [
        CandidateProduct(
            name="The Classic Twelve",
            page=1,
            price=2499,
            description="Fresh Cut Today, Vase not included",
            region_id="p1-c0-r0",
        ),
        CandidateProduct(
            name="Large Pink Rose Twenty-Four",
            page=1,
            price=999,
            description="Fresh Cut Today, Vase not included",  # wrongly borrowed
            region_id="p1-c0-r1",
        ),
    ]
    records = validate_products(candidates, parsed=parsed)
    hero, rose = records

    assert hero.status == ProductStatus.validated

    assert rose.status == ProductStatus.needs_review
    codes = {i.code for i in rose.validation_errors}
    assert "NAME_INCLUDES_SEPARATE_FIELD" in codes
    assert "ATTRIBUTE_OUT_OF_REGION" in codes
    name_issue = next(i for i in rose.validation_errors if i.code == "NAME_INCLUDES_SEPARATE_FIELD")
    assert "Large" in name_issue.message


def test_validate_warns_on_spatial_row_mismatch():
    """Renamed/updated for the region-based model: kept as a regression test
    using an explicit ParsedDocument with pre-set row_id/region_id (as if
    layout segmentation had already run)."""
    from app.models.document import BBox, LayoutRegion, ParsedPage, TextBlock

    name_bbox = BBox(x0=10, y0=10, x1=60, y1=20)
    price_bbox = BBox(x0=90, y0=30, x1=130, y1=40)
    parsed = make_document(
        [
            ParsedPage(
                page_number=1,
                width=400,
                height=200,
                blocks=[
                    TextBlock(
                        block_id="p1-b0",
                        text="Veg noodles",
                        bbox=name_bbox,
                        region_id="p1-c0-r0",
                    ),
                    TextBlock(
                        block_id="p1-b1",
                        text="159/-",
                        bbox=price_bbox,
                        region_id="p1-c0-r1",
                    ),
                ],
                regions=[
                    LayoutRegion(
                        region_id="p1-c0-r0", page=1, column_id=0, bbox=name_bbox, block_ids=["p1-b0"]
                    ),
                    LayoutRegion(
                        region_id="p1-c0-r1", page=1, column_id=0, bbox=price_bbox, block_ids=["p1-b1"]
                    ),
                ],
            )
        ]
    )
    records = validate_products(
        [
            CandidateProduct(
                name="Veg noodles",
                page=1,
                price=159,
                region_id="p1-c0-r0",
            )
        ],
        parsed=parsed,
    )
    record = records[0]
    assert record.status == ProductStatus.needs_review
    assert any(i.code == "SPATIAL_ASSOCIATION_AMBIGUOUS" for i in record.validation_errors)
