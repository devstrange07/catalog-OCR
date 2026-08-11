from app.models.product import CandidateProduct, ProductStatus
from app.stages.validate import validate_products


def test_validate_required_name_and_price():
    records = validate_products(
        [
            CandidateProduct(name="Tea", price="$12.50", currency="usd"),
            CandidateProduct(name="", sku="X"),
            CandidateProduct(name="Tea", price="$12.50", currency="usd"),
        ]
    )
    assert records[0].status == ProductStatus.validated
    assert records[0].price == 12.5
    assert records[0].currency == "USD"
    assert records[1].status == ProductStatus.rejected
    assert any("duplicate" in w.lower() for w in records[0].warnings)
