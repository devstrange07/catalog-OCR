"""Tests for Gemini structured extraction stage."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.config import settings
from app.models.document import ParsedDocument, ParsedPage
from app.models.product import CandidateProduct


def _make_parsed(full_text: str = "Product X | $5.99") -> ParsedDocument:
    return ParsedDocument(
        source_filename="test.png",
        source_path="/tmp/test.png",
        page_count=1,
        pages=[
            ParsedPage(page_number=1, width=100, height=80, blocks=[], full_text=full_text)
        ],
        meta={"ocr_engine": "unlimited_ocr"},
    )


def _gemini_response(products: list[dict]) -> dict:
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": json.dumps({"products": products})}]
                }
            }
        ]
    }


def test_extract_products_gemini_basic(monkeypatch):
    from app.stages import extract_gemini as mod

    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(settings, "gemini_model", "gemini-2.5-flash")

    products_payload = [
        {"name": "Widget", "price": "9.99", "currency": "USD", "page": 1}
    ]
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = _gemini_response(products_payload)

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = lambda s, *a: None
        mock_client.post.return_value = fake_resp
        mock_client_cls.return_value = mock_client

        parsed = _make_parsed("Widget $9.99")
        result = mod.extract_products_gemini(parsed)

    assert len(result) == 1
    assert isinstance(result[0], CandidateProduct)
    assert result[0].name == "Widget"
    assert result[0].price == "9.99"


def test_extract_products_gemini_raises_without_key(monkeypatch):
    from app.stages import extract_gemini as mod

    monkeypatch.setattr(settings, "gemini_api_key", "")

    parsed = _make_parsed()
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is missing"):
        mod.extract_products_gemini(parsed)


def test_extract_deduplicates_candidates(monkeypatch):
    from app.stages import extract_gemini as mod

    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(settings, "gemini_model", "gemini-2.5-flash")
    monkeypatch.setattr(settings, "extract_pages_per_chunk", 1)

    products_payload = [
        {"name": "Widget", "price": "9.99", "page": 1}
    ]
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = _gemini_response(products_payload)

    parsed = ParsedDocument(
        source_filename="test.pdf",
        source_path="/tmp/test.pdf",
        page_count=2,
        pages=[
            ParsedPage(page_number=1, blocks=[], full_text="Widget $9.99"),
            ParsedPage(page_number=2, blocks=[], full_text="Widget $9.99"),
        ],
        meta={},
    )

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = lambda s, *a: None
        mock_client.post.return_value = fake_resp
        mock_client_cls.return_value = mock_client

        result = mod.extract_products_gemini(parsed)

    # Both chunks return same product on page 1 — dedupe merges them
    assert len(result) == 1
