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
