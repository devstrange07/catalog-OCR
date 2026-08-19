from app.models.document import BBox, ParsedDocument, ParsedPage, TextBlock
from app.models.product import CandidateProduct, PriceVariant, ProductStatus
from app.stages.validate import validate_products


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
    assert records[0].price == 12.5
    assert records[0].currency == "USD"
    assert records[0].price_variants is not None
    assert records[0].price_variants[0].price == 12.5
    assert records[0].price_variants[0].details == "Stay, Breakfast, Sightseeing"
    assert records[1].status == ProductStatus.rejected
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
                price_variants=[
                    PriceVariant(label="Half", price=159),
                    PriceVariant(label="Full", price=199),
                ],
            )
        ],
        parsed=parsed,
    )
    assert any("variant 'Full'" in w for w in records[0].warnings)
