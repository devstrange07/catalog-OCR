from app.models.document import BBox, ParsedDocument, ParsedPage, TextBlock
from app.models.product import (
    Availability,
    CandidateProduct,
    Discount,
    ItemType,
    PriceTier,
    ProductStatus,
    Promotion,
    Variant,
)
from app.stages.validate import validate_products


def test_validate_required_name_and_price():
    records = validate_products(
        [
            CandidateProduct(
                name="Tea",
                price="$12.50",
                currency="usd",
                variants=[
                    Variant(
                        attribute="Package",
                        label="Regular",
                        price="12.50",
                        description="Stay, Breakfast, Sightseeing",
                    )
                ],
            ),
            CandidateProduct(name="", sku="X"),
            CandidateProduct(name="Tea", price="$12.50", currency="usd"),
        ]
    )
    assert records[0].status == ProductStatus.validated
    assert records[0].price == 12.5
    assert records[0].currency == "USD"
    assert records[0].variants is not None
    assert records[0].variants[0].price == 12.5
    assert records[0].variants[0].description == "Stay, Breakfast, Sightseeing"
    assert records[1].status == ProductStatus.rejected
    assert any("duplicate" in w.lower() for w in records[0].warnings)


def test_validate_sentinel_variant_price_does_not_reject():
    records = validate_products(
        [
            CandidateProduct(
                name="Chronos Carousel",
                price=999,
                variants=[
                    Variant(attribute="Age", label="Toddler", price="FREE", description="Below 3 Years"),
                    Variant(attribute="Age", label="Child", price=999, description="3 - 12 Years"),
                ],
            )
        ]
    )
    record = records[0]
    assert record.status == ProductStatus.validated
    assert record.validation_errors == []
    assert record.variants[0].price is None
    assert record.variants[0].available == "free"
    assert record.variants[1].price == 999.0


def test_validate_unparseable_variant_price_warns_not_rejects():
    records = validate_products(
        [
            CandidateProduct(
                name="Lockers",
                price=90,
                variants=[
                    Variant(attribute="Size", label="Large", price="???"),
                    Variant(attribute="Size", label="Small", price=90),
                ],
            )
        ]
    )
    record = records[0]
    assert record.status == ProductStatus.validated
    assert record.validation_errors == []
    assert any("Unparseable variant price" in w for w in record.warnings)
    assert any("Unparseable variant price" in w for w in record.variants[0].warnings)


def test_validate_empty_shell_with_only_variants_is_not_empty_shell():
    records = validate_products(
        [
            CandidateProduct(
                name="",
                variants=[Variant(attribute="Age", label="Adult", price=200)],
            )
        ]
    )
    assert records[0].status == ProductStatus.rejected
    assert "Missing required field: name" in records[0].validation_errors
    assert "Empty product shell" not in records[0].validation_errors


def test_validate_warns_on_spatial_row_mismatch():
    parsed = ParsedDocument(
        source_filename="menu.png",
        source_path="menu.png",
        page_count=1,
        pages=[
            ParsedPage(
                page_number=1,
                blocks=[
                    TextBlock(
                        block_id="p1-b0",
                        text="Veg noodles",
                        bbox=BBox(x0=10, y0=10, x1=60, y1=20),
                        row_id=0,
                    ),
                    TextBlock(
                        block_id="p1-b1",
                        text="159/-",
                        bbox=BBox(x0=90, y0=30, x1=130, y1=40),
                        row_id=1,
                    ),
                ],
            )
        ],
    )
    records = validate_products(
        [
            CandidateProduct(
                name="Veg noodles",
                page=1,
                price=159,
                source_snippet="Veg noodles 159/-",
            )
        ],
        parsed=parsed,
    )
    assert any("not on the same OCR row" in w for w in records[0].warnings)


def test_validate_warns_on_variant_spatial_row_mismatch():
    parsed = ParsedDocument(
        source_filename="menu.png",
        source_path="menu.png",
        page_count=1,
        pages=[
            ParsedPage(
                page_number=1,
                blocks=[
                    TextBlock(
                        block_id="p1-b0",
                        text="Veg noodles",
                        bbox=BBox(x0=10, y0=10, x1=60, y1=20),
                        row_id=0,
                    ),
                    TextBlock(
                        block_id="p1-b1",
                        text="159/-",
                        bbox=BBox(x0=90, y0=10, x1=130, y1=20),
                        row_id=0,
                    ),
                    TextBlock(
                        block_id="p1-b2",
                        text="199/-",
                        bbox=BBox(x0=90, y0=30, x1=130, y1=40),
                        row_id=1,
                    ),
                ],
            )
        ],
    )
    records = validate_products(
        [
            CandidateProduct(
                name="Veg noodles",
                page=1,
                price=159,
                source_snippet="Veg noodles 159/- 199/-",
                variants=[
                    Variant(attribute="Portion", label="Half", price=159),
                    Variant(attribute="Portion", label="Full", price=199),
                ],
            )
        ],
        parsed=parsed,
    )
    assert any("variant 'Full'" in w for w in records[0].warnings)


def test_validate_currency_symbol_maps_to_iso():
    records = validate_products(
        [
            CandidateProduct(name="Kulfi", price=59, currency="₹"),
            CandidateProduct(name="Widget", price=9.99, currency="$"),
            CandidateProduct(name="Pastry", price=3.5, currency="€"),
        ]
    )
    assert records[0].currency == "INR"
    assert records[1].currency == "USD"
    assert records[2].currency == "EUR"
    assert not any("Currency not ISO-like" in w for w in records[0].warnings)
    assert not any("Currency not ISO-like" in w for w in records[1].warnings)


def test_validate_list_price_discount_promotion_and_tiers():
    records = validate_products(
        [
            CandidateProduct(
                name="Birthday Blast",
                brand="HOCCO",
                item_type="bundle",
                category="School & Party Combos",
                price=1499,
                list_price=1895,
                currency="₹",
                discount=Discount(amount=396, description="save ₹396"),
                includes=["cake tub 1kg", "12 cones", "2 shakes"],
                promotion=Promotion(
                    type="bundle",
                    label="Party Combo",
                    conditions="Not clubbable with coupon",
                ),
                validity="1-30 Sep 2026",
                conditions=["Inclusive of 5% GST"],
                tags=["bestseller"],
                price_tiers=[
                    PriceTier(label="Combo Price", price=1499, is_current=True),
                    PriceTier(label="MRP", price=1895, is_current=False),
                ],
                confidence=0.95,
            )
        ]
    )
    rec = records[0]
    assert rec.status == ProductStatus.validated
    assert rec.item_type == ItemType.bundle
    assert rec.currency == "INR"
    assert rec.list_price == 1895.0
    assert rec.discount is not None and rec.discount.amount == 396.0
    assert rec.includes == ["cake tub 1kg", "12 cones", "2 shakes"]
    assert rec.promotion is not None and rec.promotion.label == "Party Combo"
    assert rec.validity == "1-30 Sep 2026"
    assert rec.conditions == ["Inclusive of 5% GST"]
    assert rec.tags == ["bestseller"]
    assert rec.price_tiers is not None
    assert len(rec.price_tiers) == 2
    assert rec.price_tiers[1].is_current is False
    assert rec.confidence == 0.95


def test_validate_item_type_and_availability_aliases():
    records = validate_products(
        [
            CandidateProduct(
                name="Gelato Bar",
                item_type="combo",
                availability="coming soon",
                price=None,
            ),
            CandidateProduct(
                name="Enterprise Plan",
                item_type="subscription",
                availability="contact sales",
                price_qualifier="on_request",
            ),
        ]
    )
    assert records[0].item_type == ItemType.bundle
    assert records[0].availability == Availability.coming_soon
    assert records[1].item_type == ItemType.plan
    assert records[1].availability == Availability.on_request
    assert records[1].price_qualifier == "on_request"


def test_validate_passes_project_name_and_location():
    records = validate_products(
        [
            CandidateProduct(
                name="2BHK Flat",
                sku="RY-0001",
                project_name="Skyline Vistas",
                location="Whitefield, Bangalore",
                category="RESIDENTIAL: APARTMENTS",
                price=10300000,
                currency="INR",
            )
        ]
    )
    assert records[0].status == ProductStatus.validated
    assert records[0].project_name == "Skyline Vistas"
    assert records[0].location == "Whitefield, Bangalore"


def test_flag_scope_anomalies_promotion_bleed():
    """Same promo across distinct projects on >50% of items → warning + confidence drop."""
    promo = Promotion(type="other", label="Free club membership", details="on 3BHK+")
    candidates = [
        CandidateProduct(
            name="3BHK Flat",
            sku=f"RY-{i:04d}",
            project_name="Skyline Vistas" if i <= 4 else "Greenpark Enclave",
            category="RESIDENTIAL: APARTMENTS",
            price=10000000 + i,
            currency="INR",
            promotion=promo,
            confidence=1.0,
        )
        for i in range(1, 7)
    ]
    records = validate_products(candidates)
    assert all(
        "Promotion may be over-applied across sections" in r.warnings for r in records
    )
    assert all(r.confidence == 0.8 for r in records)


def test_flag_scope_anomalies_conditions_bleed():
    shared_conditions = [
        "Possession Dec 2028.",
        "10% booking, 80% link-plan.",
        "Bank tie-ups: SBI, HDFC, ICICI — EMI from 8.35% p.a.",
    ]
    candidates = [
        CandidateProduct(
            name="2BHK Flat",
            sku="RY-0001",
            project_name="Skyline Vistas",
            category="RESIDENTIAL: APARTMENTS",
            price=10300000,
            conditions=shared_conditions,
            confidence=1.0,
        ),
        CandidateProduct(
            name="2BHK Flat",
            sku="RY-0011",
            project_name="Greenpark Enclave",
            category="RESIDENTIAL: APARTMENTS",
            price=9400000,
            conditions=shared_conditions,
            confidence=1.0,
        ),
        CandidateProduct(
            name="Metro Heights",
            project_name="Metro Heights",
            category="MORE APARTMENT PROJECTS",
            price=8200000,
            conditions=shared_conditions,
            confidence=1.0,
        ),
    ]
    records = validate_products(candidates)
    assert all(
        "Conditions may be over-applied from a single section" in r.warnings
        for r in records
    )
    assert all(r.confidence == 0.8 for r in records)


def test_realy_scope_expectations_on_correct_extraction():
    """Correct RealY-like scoping should not trigger bleed warnings.

    Mirrors the pamphlet layout:
    - Skyline: no club promo; has possession/booking terms
    - Greenpark 3BHK+: club promo; no Skyline possession terms
    - Snapshot projects: page-level GST only
    """
    skyline_conditions = [
        "All prices incl. applicable GST",
        "RERA registered",
        "Stamp duty & registration extra.",
        "Possession Dec 2028.",
        "10% booking, 80% link-plan.",
    ]
    greenpark_promo = Promotion(
        type="other", label="Free club membership", details="on 3BHK+"
    )
    page_legal = [
        "All prices incl. applicable GST",
        "RERA registered",
        "Stamp duty & registration extra.",
    ]

    candidates = [
        CandidateProduct(
            name="2BHK Flat",
            sku="RY-0001",
            project_name="Skyline Vistas",
            location="Whitefield, Bangalore",
            category="RESIDENTIAL: APARTMENTS",
            price=10300000,
            currency="INR",
            unit="sqft",
            pack_size="1080 sqft",
            conditions=skyline_conditions,
            page=2,
            confidence=0.95,
            price_tiers=[PriceTier(label="EMI", price=59999, billing_cycle="monthly")],
        ),
        CandidateProduct(
            name="3BHK Flat",
            sku="RY-0003",
            project_name="Skyline Vistas",
            location="Whitefield, Bangalore",
            category="RESIDENTIAL: APARTMENTS",
            price=13800000,
            currency="INR",
            unit="sqft",
            conditions=skyline_conditions,
            page=2,
            confidence=0.95,
            price_tiers=[PriceTier(label="EMI", price=80499, billing_cycle="monthly")],
        ),
        CandidateProduct(
            name="3BHK Flat",
            sku="RY-0012",
            project_name="Greenpark Enclave",
            location="Sarjapur, Bangalore",
            category="RESIDENTIAL: APARTMENTS",
            price=11900000,
            currency="INR",
            unit="sqft",
            promotion=greenpark_promo,
            conditions=page_legal,
            page=2,
            confidence=0.95,
        ),
        CandidateProduct(
            name="Metro Heights",
            project_name="Metro Heights",
            location="Yelahanka, Bangalore",
            category="MORE APARTMENT PROJECTS",
            price=8200000,
            currency="INR",
            price_qualifier="from",
            conditions=page_legal,
            page=2,
            confidence=0.9,
            price_tiers=[PriceTier(label="per sqft", price=7600)],
        ),
    ]
    records = validate_products(candidates)
    assert all(r.status == ProductStatus.validated for r in records)
    assert all(
        "Promotion may be over-applied across sections" not in r.warnings
        for r in records
    )
    # page_legal on 2/4 items is not a majority bleed; skyline terms stay local
    assert all(
        "Conditions may be over-applied from a single section" not in r.warnings
        for r in records
    )
    assert records[0].page == 2
    assert records[0].promotion is None
    assert records[2].promotion is not None
    assert "Free club membership" in (records[2].promotion.label or "")
    assert any(t.label == "EMI" for t in (records[0].price_tiers or []))


def test_realy_bleed_pattern_is_flagged():
    """The buggy RealY extraction pattern (club promo on Skyline + shared A terms) is flagged."""
    club = Promotion(type="other", label="Free club membership", details="on 3BHK+")
    bloated = [
        "All prices incl. applicable GST",
        "RERA registered",
        "Stamp duty & registration extra.",
        "10% booking, 80% link-plan.",
        "Possession Dec 2028.",
        "Bank tie-ups: SBI, HDFC, ICICI — EMI from 8.35% p.a.",
    ]
    candidates = []
    for i in range(1, 5):
        candidates.append(
            CandidateProduct(
                name="3BHK Flat",
                sku=f"RY-{i:04d}",
                project_name="Skyline Vistas",
                category="RESIDENTIAL: APARTMENTS",
                price=14000000,
                promotion=club,
                conditions=bloated,
                confidence=1.0,
                page=1,
            )
        )
    for i in range(11, 14):
        candidates.append(
            CandidateProduct(
                name="3BHK Flat",
                sku=f"RY-{i:04d}",
                project_name="Greenpark Enclave",
                category="RESIDENTIAL: APARTMENTS",
                price=12000000,
                promotion=club,
                conditions=bloated,
                confidence=1.0,
                page=1,
            )
        )
    candidates.append(
        CandidateProduct(
            name="Metro Heights",
            project_name="Metro Heights",
            category="MORE APARTMENT PROJECTS",
            price=8200000,
            conditions=bloated,
            confidence=1.0,
            page=1,
        )
    )
    records = validate_products(candidates)
    promo_warned = [
        r for r in records if "Promotion may be over-applied across sections" in r.warnings
    ]
    cond_warned = [
        r
        for r in records
        if "Conditions may be over-applied from a single section" in r.warnings
    ]
    assert len(promo_warned) >= 6  # club on 7 of 8 items across projects
    assert len(cond_warned) == len(records)
    assert all(r.confidence is not None and r.confidence <= 0.8 for r in cond_warned)


def test_normalize_price_applies_scale_suffixes():
    records = validate_products(
        [
            CandidateProduct(name="A", price="10.5 Lakh", currency="INR"),
            CandidateProduct(name="B", price="2 Cr", currency="INR"),
            CandidateProduct(name="C", price="99K", currency="USD"),
            CandidateProduct(name="D", price="$12.50", currency="USD"),
            CandidateProduct(
                name="E",
                price=100,
                price_tiers=[PriceTier(label="EMI", price="0.6 L")],
            ),
        ]
    )
    assert records[0].price == 1_050_000.0
    assert any("scale suffix" in w.lower() for w in records[0].warnings)
    assert records[1].price == 20_000_000.0
    assert records[2].price == 99_000.0
    assert records[3].price == 12.5
    assert records[4].price_tiers is not None
    assert records[4].price_tiers[0].price == 60_000.0


def test_flag_asymmetric_table_conditions():
    shared_shape = {"category": "Rate Card", "page": 1, "currency": "INR"}
    candidates = [
        CandidateProduct(
            name=f"Left-{i}",
            project_name="Tower A",
            price=100 + i,
            conditions=["* Rates excl. tax"],
            **shared_shape,
        )
        for i in range(3)
    ] + [
        CandidateProduct(
            name=f"Right-{i}",
            project_name="Tower B",
            price=200 + i,
            **shared_shape,
        )
        for i in range(3)
    ]
    records = validate_products(candidates)
    right = [r for r in records if r.project_name == "Tower B"]
    assert all(
        any("possible missed footnote" in w for w in r.warnings) for r in right
    )


def test_flag_snippet_coverage_gap():
    records = validate_products(
        [
            CandidateProduct(
                name="3BHK Flat",
                price=1_000_000,
                source_snippet="3BHK with free club membership and 10% booking",
                includes=None,
                conditions=None,
            )
        ]
    )
    warnings = records[0].warnings
    assert any("free" in w.lower() and "structured field" in w for w in warnings)
    assert any("10%" in w and "structured field" in w for w in warnings)


def test_flag_snippet_skips_already_scaled_price():
    records = validate_products(
        [
            CandidateProduct(
                name="Apartments",
                price="1.05 Cr",
                currency="INR",
                source_snippet="Apartments from 1.05 Cr",
            )
        ]
    )
    assert records[0].price == 10_500_000.0
    assert any("scale suffix" in w.lower() for w in records[0].warnings)
    assert not any("1.05 Cr" in w and "structured field" in w for w in records[0].warnings)


def test_flag_magnitude_inconsistency_on_emi_scale():
    candidates = [
        CandidateProduct(
            name=f"Unit-{i}",
            project_name="Skyline",
            category="Apartments",
            price=10_000_000,
            currency="INR",
            price_tiers=[
                PriceTier(label="EMI", price=60_000 if i < 2 else 0.6),
            ],
        )
        for i in range(3)
    ]
    records = validate_products(candidates)
    assert all(
        any("implausible range" in w for w in r.warnings) for r in records
    )


def test_flag_currency_inconsistency_same_section():
    records = validate_products(
        [
            CandidateProduct(
                name="A",
                project_name="Skyline",
                category="Apartments",
                price=100,
                currency="INR",
            ),
            CandidateProduct(
                name="B",
                project_name="Skyline",
                category="Apartments",
                price=200,
                currency="USD",
            ),
        ]
    )
    assert all(any("Mixed currencies" in w for w in r.warnings) for r in records)
    assert all(r.status == ProductStatus.validated for r in records)


def test_flag_category_shape_drift():
    candidates = [
        CandidateProduct(
            name=f"Item-{i}",
            category="Apartments",
            project_name="P1",
            price=100 + i,
        )
        for i in range(3)
    ] + [
        CandidateProduct(
            name="Odd",
            category="Price Comparison Table Overview",
            project_name="P2",
            price=999,
        )
    ]
    records = validate_products(candidates)
    odd = next(r for r in records if r.name == "Odd")
    assert any("taxonomy label" in w for w in odd.warnings)


def test_flag_missing_derivable_rate():
    candidates = [
        CandidateProduct(
            name="A",
            project_name="Tower",
            category="Apartments",
            price=10_800_000,
            pack_size="1080 sqft",
            price_tiers=[PriceTier(label="per sqft", price=10_000)],
        ),
        CandidateProduct(
            name="B",
            project_name="Tower",
            category="Apartments",
            price=12_000_000,
            pack_size="1200 sqft",
        ),
    ]
    records = validate_products(candidates)
    assert any(
        "computable but missing" in w for w in records[1].warnings
    )
    assert not any(
        "computable but missing" in w for w in records[0].warnings
    )


def test_flag_unclaimed_footnote_block():
    parsed = ParsedDocument(
        source_filename="rates.pdf",
        source_path="rates.pdf",
        page_count=1,
        pages=[
            ParsedPage(
                page_number=1,
                blocks=[
                    TextBlock(
                        block_id="p1-b0",
                        text="Widget A",
                        bbox=BBox(x0=10, y0=10, x1=60, y1=20),
                        row_id=0,
                    ),
                    TextBlock(
                        block_id="p1-b1",
                        text="* Prices exclude GST",
                        bbox=BBox(x0=10, y0=40, x1=120, y1=50),
                        row_id=2,
                    ),
                ],
            )
        ],
    )
    records = validate_products(
        [
            CandidateProduct(
                name="Widget A",
                page=1,
                price=100,
                source_snippet="Widget A 100",
            )
        ],
        parsed=parsed,
    )
    assert any("unattached table footnote" in w for w in records[0].warnings)


def test_coverage_repair_fills_empty_includes_only():
    from app.stages.coverage_repair import apply_coverage_repair_payload

    records = validate_products(
        [
            CandidateProduct(
                name="Plan",
                price=99,
                source_snippet="Plan includes free onboarding",
                includes=["already kept"],
            )
        ]
    )
    apply_coverage_repair_payload(
        records[0],
        {
            "includes": ["should not overwrite"],
            "conditions": ["from snippet"],
            "description": "filled",
        },
    )
    assert records[0].includes == ["already kept"]
    assert records[0].conditions == ["from snippet"]
    assert records[0].description == "filled"
    assert any("Coverage repair" in w for w in records[0].warnings)
