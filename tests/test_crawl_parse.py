"""Tests for URL crawl helpers (no live network / crawl4ai)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.stages.crawl import (
    CrawlError,
    detect_fetch_mode,
    is_binary_content_type,
    is_binary_suffix,
    normalize_url,
    parsed_document_from_markdown,
)


def test_normalize_url_ok():
    assert normalize_url("  https://Example.com/a  ") == "https://Example.com/a"


@pytest.mark.parametrize(
    "bad",
    ["", "ftp://x.com", "not-a-url", "http://"],
)
def test_normalize_url_rejects(bad: str):
    with pytest.raises(CrawlError):
        normalize_url(bad)


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://cdn.example.com/cat.pdf", True),
        ("https://cdn.example.com/img.PNG", True),
        ("https://cdn.example.com/page", False),
        ("https://cdn.example.com/a.html", False),
    ],
)
def test_is_binary_suffix(url: str, expected: bool):
    assert is_binary_suffix(url) is expected


@pytest.mark.parametrize(
    "ct,expected",
    [
        ("application/pdf", True),
        ("application/pdf; charset=binary", True),
        ("image/png", True),
        ("image/jpeg", True),
        ("text/html", False),
        ("application/json", False),
        (None, False),
    ],
)
def test_is_binary_content_type(ct: str | None, expected: bool):
    assert is_binary_content_type(ct) is expected


def test_detect_fetch_mode_by_suffix():
    assert detect_fetch_mode("https://files.example.com/catalog.pdf") == "binary"


def test_detect_fetch_mode_head_html():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "text/html; charset=utf-8"}

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = False
    mock_client.head.return_value = mock_resp

    with patch("app.stages.crawl.httpx.Client", return_value=mock_client):
        assert detect_fetch_mode("https://shop.example.com/products") == "html"


def test_detect_fetch_mode_head_pdf():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "application/pdf"}

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = False
    mock_client.head.return_value = mock_resp

    with patch("app.stages.crawl.httpx.Client", return_value=mock_client):
        assert detect_fetch_mode("https://shop.example.com/download") == "binary"


def test_parsed_document_from_markdown():
    md = """# Catalog

## Widget Pro
SKU: W-1
Price: $9.99

## Gadget Lite
SKU: G-2
"""
    doc = parsed_document_from_markdown(
        md,
        source_url="https://shop.example.com/catalog",
        title="Shop Catalog",
    )
    assert doc.page_count == 1
    assert doc.meta["source"] == "crawl4ai"
    assert doc.meta["source_url"] == "https://shop.example.com/catalog"
    assert doc.raw_docling_markdown == md.strip()
    assert len(doc.pages[0].blocks) >= 2
    text = doc.text_for_extraction()
    assert "Widget Pro" in text
    assert "G-2" in text
    assert "OCR skipped" in doc.structure_notes[0]


def test_parsed_document_empty_raises():
    with pytest.raises(CrawlError):
        parsed_document_from_markdown("   ", source_url="https://x.com")
