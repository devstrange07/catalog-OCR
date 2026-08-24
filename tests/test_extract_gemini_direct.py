"""Gemini direct extraction — mocked HTTP, no live API."""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.config import settings
from app.models.product import CandidateProduct, PriceTier, Promotion
from app.stages.extract_gemini import PRODUCTS_RESPONSE_SCHEMA
from app.stages.extract_gemini_direct import (
    DIRECT_SYSTEM_PROMPT,
    PROMPT_VERSION,
    QUALITY_WARNING_HOLLOW,
    REPAIR_SYSTEM_PROMPT,
    assess_direct_quality,
    extract_products_gemini_direct,
    extract_products_gemini_direct_from_text,
    merge_direct_candidates,
)
from app.stages.parse_gemini import parse_document_gemini


def _gemini_body(products: list[dict] | str, *, finish_reason: str | None = "STOP") -> dict:
    if isinstance(products, str):
        text = products
    else:
        text = json.dumps({"products": products})
    candidate: dict = {"content": {"parts": [{"text": text}]}}
    if finish_reason is not None:
        candidate["finishReason"] = finish_reason
    return {"candidates": [candidate]}


def _fake_client(post_side_effect) -> MagicMock:
    fake_client = MagicMock()
    fake_client.__enter__.return_value = fake_client
    fake_client.__exit__.return_value = False
    if callable(post_side_effect) and not isinstance(post_side_effect, MagicMock):
        fake_client.post.side_effect = post_side_effect
    else:
        fake_client.post.return_value = post_side_effect
    return fake_client


def _ok_resp(products: list[dict] | str) -> MagicMock:
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = _gemini_body(products)
    return fake_resp


def test_direct_extract_requires_api_key(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "")
    img_path = tmp_path / "page.png"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(img_path)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        extract_products_gemini_direct(img_path)


def test_direct_extract_file_uses_product_schema_and_file_part(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(settings, "gemini_model", "gemini-2.5-flash")
    img_path = tmp_path / "page.png"
    Image.new("RGB", (12, 10), color=(255, 255, 255)).save(img_path)

    fake_client = _fake_client(
        _ok_resp([{"name": "Widget", "price": "9.99", "currency": "USD", "page": 1}])
    )

    with patch("app.stages.extract_gemini_direct.httpx.Client", return_value=fake_client):
        result = extract_products_gemini_direct(img_path)

    assert len(result.candidates) == 1
    assert isinstance(result.candidates[0], CandidateProduct)
    assert result.candidates[0].name == "Widget"
    assert result.candidates[0].price == "9.99"
    assert result.call_count == 1
    assert result.quality.suspicious is False

    body = fake_client.post.call_args.kwargs["json"]
    gen = body["generationConfig"]
    assert gen["responseMimeType"] == "application/json"
    assert gen["responseSchema"] is PRODUCTS_RESPONSE_SCHEMA
    assert gen["maxOutputTokens"] == settings.extract_max_tokens
    assert gen["thinkingConfig"]["thinkingBudget"] == -1
    parts = body["contents"][0]["parts"]
    assert "inline_data" in parts[1]
    assert parts[1]["inline_data"]["mime_type"] == "image/png"
    assert PROMPT_VERSION == "gemini_direct_v2"
    system_text = body["systemInstruction"]["parts"][0]["text"]
    assert "Capacity priority" in system_text
    assert "City × unit-type rental matrices" in system_text
    assert "section/table" in system_text


def test_direct_extract_empty_products_is_success(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    img_path = tmp_path / "page.png"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(img_path)

    fake_client = _fake_client(_ok_resp([]))

    with patch("app.stages.extract_gemini_direct.httpx.Client", return_value=fake_client):
        result = extract_products_gemini_direct(img_path)
    assert result.candidates == []
    assert result.call_count == 1
    assert result.quality.suspicious is False


def test_direct_extract_malformed_json_retries_then_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    img_path = tmp_path / "page.png"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(img_path)

    fake_client = _fake_client(_ok_resp("{not json"))

    with patch("app.stages.extract_gemini_direct.httpx.Client", return_value=fake_client):
        with pytest.raises(RuntimeError, match="invalid JSON"):
            extract_products_gemini_direct(img_path)
    assert fake_client.post.call_count == 2


def test_direct_extract_http_error(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    img_path = tmp_path / "page.png"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(img_path)

    fake_resp = MagicMock()
    fake_resp.status_code = 500
    fake_resp.text = "boom"
    fake_client = _fake_client(fake_resp)

    with patch("app.stages.extract_gemini_direct.httpx.Client", return_value=fake_client):
        with pytest.raises(RuntimeError, match="failed \\(500\\)"):
            extract_products_gemini_direct(img_path)
    assert fake_client.post.call_count == 1


def test_direct_extract_max_tokens_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    img_path = tmp_path / "page.png"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(img_path)

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = _gemini_body(
        [{"name": "Partial"}], finish_reason="MAX_TOKENS"
    )
    fake_client = _fake_client(fake_resp)

    with patch("app.stages.extract_gemini_direct.httpx.Client", return_value=fake_client):
        with pytest.raises(RuntimeError, match="MAX_TOKENS"):
            extract_products_gemini_direct(img_path)


def test_direct_extract_safety_block(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    img_path = tmp_path / "page.png"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(img_path)

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "promptFeedback": {"blockReason": "SAFETY"},
        "candidates": [],
    }
    fake_client = _fake_client(fake_resp)

    with patch("app.stages.extract_gemini_direct.httpx.Client", return_value=fake_client):
        with pytest.raises(RuntimeError, match="blocked"):
            extract_products_gemini_direct(img_path)


def test_direct_from_text_no_file_part(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    fake_client = _fake_client(_ok_resp([{"name": "From Markdown", "page": 1}]))

    with patch("app.stages.extract_gemini_direct.httpx.Client", return_value=fake_client):
        result = extract_products_gemini_direct_from_text("# Widget\n$5")

    assert result.candidates[0].name == "From Markdown"
    body = fake_client.post.call_args.kwargs["json"]
    parts = body["contents"][0]["parts"]
    assert len(parts) == 1
    assert "text" in parts[0]
    assert "inline_data" not in parts[0]


def test_direct_does_not_import_parse_document_gemini_at_call(tmp_path, monkeypatch):
    """Direct extract must not invoke Stage-1 Gemini parse."""
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    img_path = tmp_path / "page.png"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(img_path)

    fake_client = _fake_client(_ok_resp([{"name": "X"}]))

    with patch("app.stages.parse_gemini.parse_document_gemini", side_effect=AssertionError("parse called")):
        with patch("app.stages.extract_gemini_direct.httpx.Client", return_value=fake_client):
            extract_products_gemini_direct(img_path)

    assert callable(parse_document_gemini)


def test_direct_schema_validation_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    img_path = tmp_path / "page.png"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(img_path)

    fake_client = _fake_client(_ok_resp([{"name": "X", "confidence": 2.5}]))

    with patch("app.stages.extract_gemini_direct.httpx.Client", return_value=fake_client):
        with pytest.raises(RuntimeError, match="schema validation"):
            extract_products_gemini_direct(img_path)


def test_direct_extract_observability_retry_spans(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    img_path = tmp_path / "page.png"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(img_path)

    fake_client = _fake_client(_ok_resp("{not json"))

    span_names: list[str] = []

    @contextmanager
    def fake_span(name, **kwargs):
        span_names.append(name)
        yield None

    monkeypatch.setattr("app.stages.extract_gemini_direct.trace_span", fake_span)

    with patch("app.stages.extract_gemini_direct.httpx.Client", return_value=fake_client):
        with pytest.raises(RuntimeError, match="invalid JSON"):
            extract_products_gemini_direct(img_path)

    assert span_names.count("gemini_direct.attempt") == 2
    assert span_names.count("gemini_direct.generate") == 2


def test_assess_direct_quality_flags_hollow_skus():
    hollow = [
        CandidateProduct(name="2BHK Flat", sku="RY-0001"),
        CandidateProduct(name="3BHK Flat", sku="RY-0002"),
        CandidateProduct(name="4BHK Flat", sku="RY-0003"),
    ]
    report = assess_direct_quality(hollow)
    assert report.suspicious is True
    assert "hollow_skus" in report.reasons
    assert report.hollow_sku_ratio == 1.0
    assert QUALITY_WARNING_HOLLOW in report.warning_messages()


def test_assess_direct_quality_rich_skus_not_suspicious():
    rich = [
        CandidateProduct(
            name="2BHK Flat",
            sku="RY-0001",
            project_name="Skyline Vistas",
            price="1.03",
            pack_size="1080 sqft",
            page=2,
            source_snippet="RY-0001 2BHK 1080 ₹1.03 Cr",
            price_tiers=[PriceTier(label="EMI", price="59999")],
        ),
        CandidateProduct(
            name="3BHK Flat",
            sku="RY-0002",
            project_name="Skyline Vistas",
            price="1.38",
            pack_size="1450 sqft",
            page=2,
            source_snippet="RY-0002 3BHK",
        ),
    ]
    report = assess_direct_quality(rich)
    assert report.suspicious is False
    assert report.warning_messages() == []


def test_merge_direct_candidates_fills_nulls_without_overwrite():
    base = [
        CandidateProduct(name="2BHK Flat", sku="RY-0001", price=None, project_name=None),
        CandidateProduct(
            name="Keep Promo",
            sku="RY-0099",
            project_name="Skyline",
            promotion=Promotion(label="Skyline only"),
        ),
    ]
    incoming = [
        CandidateProduct(
            name="2BHK Flat",
            sku="RY-0001",
            price="1.03",
            project_name="Skyline Vistas",
            pack_size="1080",
            price_tiers=[PriceTier(label="EMI", price="59999")],
            promotion=Promotion(label="Wrong section promo"),
        ),
        CandidateProduct(
            name="Keep Promo",
            sku="RY-0099",
            project_name="Skyline",
            promotion=Promotion(label="Should not replace"),
        ),
    ]
    merged = merge_direct_candidates(base, incoming)
    by_sku = {c.sku: c for c in merged}
    assert by_sku["RY-0001"].price == "1.03"
    assert by_sku["RY-0001"].project_name == "Skyline Vistas"
    assert by_sku["RY-0001"].pack_size == "1080"
    assert by_sku["RY-0001"].price_tiers[0].label == "EMI"
    # Promotion already present on base must not be overwritten by repair.
    assert by_sku["RY-0099"].promotion is not None
    assert by_sku["RY-0099"].promotion.label == "Skyline only"
    # Repair must not attach an unscoped promotion onto a previously promo-less SKU.
    assert by_sku["RY-0001"].promotion is None


def test_merge_does_not_overwrite_populated_price():
    base = [CandidateProduct(name="Villa", sku="RY-0017", price="2.50", project_name="A")]
    incoming = [CandidateProduct(name="Villa", sku="RY-0017", price="9.99", project_name="B")]
    merged = merge_direct_candidates(base, incoming)
    assert merged[0].price == "2.50"
    assert merged[0].project_name == "A"


def test_hollow_sku_response_triggers_one_repair_call(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    img_path = tmp_path / "page.png"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(img_path)

    hollow = [
        {"name": "2BHK Flat", "sku": "RY-0001"},
        {"name": "3BHK Flat", "sku": "RY-0002"},
        {"name": "4BHK Flat", "sku": "RY-0003"},
    ]
    repaired = [
        {
            "name": "2BHK Flat",
            "sku": "RY-0001",
            "price": "1.03",
            "project_name": "Skyline Vistas",
            "pack_size": "1080 sqft",
            "page": 2,
            "source_snippet": "RY-0001 2BHK 1080 ₹1.03 Cr",
            "price_tiers": [{"label": "EMI", "price": "59999"}],
        },
        {
            "name": "3BHK Flat",
            "sku": "RY-0002",
            "price": "1.38",
            "project_name": "Skyline Vistas",
            "pack_size": "1450 sqft",
            "page": 2,
            "source_snippet": "RY-0002",
        },
        {
            "name": "4BHK Flat",
            "sku": "RY-0003",
            "price": "2.00",
            "project_name": "Skyline Vistas",
            "pack_size": "2100 sqft",
            "page": 2,
            "source_snippet": "RY-0003",
        },
    ]

    responses = [_ok_resp(hollow), _ok_resp(repaired)]

    def post_side_effect(*args, **kwargs):
        return responses.pop(0)

    fake_client = _fake_client(post_side_effect)

    with patch("app.stages.extract_gemini_direct.httpx.Client", return_value=fake_client):
        result = extract_products_gemini_direct(img_path)

    assert fake_client.post.call_count == 2
    assert result.call_count == 2
    assert result.quality.repaired is True
    assert result.quality.repair_improved is True
    by_sku = {c.sku: c for c in result.candidates}
    assert by_sku["RY-0001"].price == "1.03"
    assert by_sku["RY-0001"].project_name == "Skyline Vistas"
    assert by_sku["RY-0001"].price_tiers[0].label == "EMI"

    first_body = fake_client.post.call_args_list[0].kwargs["json"]
    second_body = fake_client.post.call_args_list[1].kwargs["json"]
    assert first_body["systemInstruction"]["parts"][0]["text"] == DIRECT_SYSTEM_PROMPT
    assert second_body["systemInstruction"]["parts"][0]["text"] == REPAIR_SYSTEM_PROMPT
    # Same file part reused (no second upload path) — both calls include inline_data.
    assert "inline_data" in first_body["contents"][0]["parts"][1]
    assert "inline_data" in second_body["contents"][0]["parts"][1]


def test_rich_response_does_not_trigger_repair(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    img_path = tmp_path / "page.png"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(img_path)

    rich = [
        {
            "name": "2BHK Flat",
            "sku": "RY-0001",
            "price": "1.03",
            "project_name": "Skyline",
            "page": 2,
            "source_snippet": "row",
        },
        {
            "name": "3BHK Flat",
            "sku": "RY-0002",
            "price": "1.38",
            "project_name": "Skyline",
            "page": 2,
            "source_snippet": "row",
        },
    ]
    fake_client = _fake_client(_ok_resp(rich))

    with patch("app.stages.extract_gemini_direct.httpx.Client", return_value=fake_client):
        result = extract_products_gemini_direct(img_path)

    assert fake_client.post.call_count == 1
    assert result.call_count == 1
    assert result.quality.repaired is False


def test_prompt_documents_matrix_city_tier_convention():
    assert "label = city name" in DIRECT_SYSTEM_PROMPT
    assert "billing_cycle = monthly" in DIRECT_SYSTEM_PROMPT
    assert "Never broadcast a promotion" in DIRECT_SYSTEM_PROMPT


def test_city_rental_matrix_shape_is_not_hollow():
    """UnitType → City → monthly price via labeled tiers is a complete representation."""
    products = [
        CandidateProduct(
            name="1RK Flat",
            item_type="listing",
            billing_cycle="monthly",
            page=5,
            source_snippet="1RK Flat Bangalore 10800 Pune 9500",
            price_tiers=[
                PriceTier(label="Bangalore", price="10800", billing_cycle="monthly", currency="INR"),
                PriceTier(label="Pune", price="9500", billing_cycle="monthly", currency="INR"),
                PriceTier(label="Hyderabad", price="9000", billing_cycle="monthly", currency="INR"),
            ],
        ),
        CandidateProduct(
            name="2BHK Flat",
            item_type="listing",
            billing_cycle="monthly",
            page=5,
            source_snippet="2BHK Flat Bangalore 29400",
            price_tiers=[
                PriceTier(label="Bangalore", price="29400", billing_cycle="monthly", currency="INR"),
                PriceTier(label="Mumbai", price="44800", billing_cycle="monthly", currency="INR"),
            ],
        ),
    ]
    report = assess_direct_quality(products)
    assert report.suspicious is False
    assert report.with_price == 2
    assert all(t.label not in {"price", "tier"} for p in products for t in (p.price_tiers or []))
