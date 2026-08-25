"""Tests for post-validation Prism export stage."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.config import settings
from app.models.prism import PrismIssueSeverity
from app.models.product import ProductRecord, ProductStatus


def _validated_product(
    name: str = "Widget",
    price: float = 9.99,
    currency: str = "USD",
) -> ProductRecord:
    return ProductRecord(
        name=name,
        price=price,
        currency=currency,
        description="A test widget",
        status=ProductStatus.validated,
    )


def _gemini_prism_response(products: list[dict]) -> dict:
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": json.dumps({"products": products})}]
                }
            }
        ]
    }


def _raw_product_payload(
    name: str = "Widget",
    amount: float = 9.99,
    currency: str = "USD",
    discount: float | None = None,
) -> dict:
    price_list: dict = {
        "name": "default",
        "country": "US",
        "currency": currency,
        "amount": amount,
    }
    if discount is not None:
        price_list["discount"] = discount
    return {
        "id": "placeholder",
        "name": name,
        "description": "A test widget",
        "kind": "product",
        "plans": [
            {
                "id": "plan-placeholder",
                "name": "Default",
                "price_lists": [price_list],
            }
        ],
        "updated_at": "placeholder",
    }


def _mock_httpx(fake_resp: MagicMock):
    mock_client = MagicMock()
    mock_client.__enter__ = lambda s: mock_client
    mock_client.__exit__ = lambda s, *a: None
    mock_client.post.return_value = fake_resp
    return mock_client


def test_export_products_prism_happy_path(monkeypatch):
    from app.stages import export_prism as mod

    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(settings, "gemini_model", "gemini-2.5-flash")

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = _gemini_prism_response(
        [_raw_product_payload(amount=9.99)]
    )

    with patch("httpx.Client") as mock_client_cls:
        mock_client_cls.return_value = _mock_httpx(fake_resp)
        result = mod.export_products_prism(
            [_validated_product()], job_id="abc123"
        )

    assert result.valid is True
    assert result.job_id == "abc123"
    assert result.source_product_count == 1
    assert result.exported_product_count == 1
    assert len(result.products) == 1
    product = result.products[0]
    assert product.name == "Widget"
    assert product.id == "abc123-0000"
    assert product.plans[0].id == "abc123-0000-plan-00"
    assert product.plans[0].price_lists[0].amount == 9.99
    assert product.updated_at  # stamped
    assert not any(i.severity == PrismIssueSeverity.error for i in result.issues)


def test_export_numeric_rejection_amount_and_discount(monkeypatch):
    from app.stages import export_prism as mod

    monkeypatch.setattr(settings, "gemini_api_key", "test-key")

    bad = _raw_product_payload(amount=-5.0, discount=100.0)
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = _gemini_prism_response([bad])

    with patch("httpx.Client") as mock_client_cls:
        mock_client_cls.return_value = _mock_httpx(fake_resp)
        result = mod.export_products_prism(
            [_validated_product()], job_id="job1"
        )

    assert result.valid is False
    error_fields = {
        i.field for i in result.issues if i.severity == PrismIssueSeverity.error
    }
    assert "price_lists.amount" in error_fields
    assert "price_lists.discount" in error_fields


def test_export_price_mismatch_cross_check(monkeypatch):
    from app.stages import export_prism as mod

    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(settings, "prism_price_tolerance_pct", 0.15)

    # Source price 10.0; exported amount 100.0 — far outside 15%
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = _gemini_prism_response(
        [_raw_product_payload(amount=100.0, currency="USD")]
    )

    with patch("httpx.Client") as mock_client_cls:
        mock_client_cls.return_value = _mock_httpx(fake_resp)
        result = mod.export_products_prism(
            [_validated_product(price=10.0, currency="USD")],
            job_id="job2",
        )

    assert result.valid is False
    assert any(
        i.severity == PrismIssueSeverity.error and i.field == "price_lists.amount"
        for i in result.issues
    )


def test_export_raises_without_api_key(monkeypatch):
    from app.stages import export_prism as mod

    monkeypatch.setattr(settings, "gemini_api_key", "")

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is missing"):
        mod.export_products_prism([_validated_product()], job_id="job3")


def test_export_empty_validated_short_circuits(monkeypatch):
    from app.stages import export_prism as mod

    monkeypatch.setattr(settings, "gemini_api_key", "test-key")

    rejected = ProductRecord(
        name="Bad",
        status=ProductStatus.rejected,
        validation_errors=["Missing required field: name"],
    )

    with patch("httpx.Client") as mock_client_cls:
        result = mod.export_products_prism([rejected], job_id="job4")
        mock_client_cls.assert_not_called()

    assert result.valid is True
    assert result.exported_product_count == 0
    assert result.source_product_count == 0
    assert result.products == []
