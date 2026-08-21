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
