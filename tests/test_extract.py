"""Stage 2 extract helpers — chunking, JSON parse, retry."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.config import settings
from app.models.document import ParsedDocument, ParsedPage, TextBlock
from app.models.product import CandidateProduct
from app.stages.extract import (
    _format_pages_text,
    _chunk_pages,
    _dedupe_candidates,
    _extract_chunk,
    _parse_tool_args,
    extract_products,
)


def _page(n: int, text: str = "Widget") -> ParsedPage:
    return ParsedPage(
        page_number=n,
        blocks=[TextBlock(block_id=f"p{n}-b0", text=text, reading_order=0)],
        full_text=text,
    )


def test_parse_tool_args_happy_path():
    raw = '{"products": [{"name": "Amul Butter", "page": 1, "sku": "AB-1"}]}'
    products = _parse_tool_args(raw)
    assert len(products) == 1
    assert products[0].name == "Amul Butter"
    assert products[0].page == 1


def test_parse_tool_args_invalid_json():
    with pytest.raises(RuntimeError, match="not valid JSON"):
        _parse_tool_args('{"products": [{"name": "unterminated')


def test_chunk_pages_five_into_three():
    pages = [_page(i) for i in range(1, 6)]
    chunks = _chunk_pages(pages, 2)
    assert len(chunks) == 3
    assert [p.page_number for p in chunks[0]] == [1, 2]
    assert [p.page_number for p in chunks[1]] == [3, 4]
    assert [p.page_number for p in chunks[2]] == [5]


def test_chunk_pages_empty():
    assert _chunk_pages([], 2) == [[]]


def test_dedupe_candidates():
    items = [
        CandidateProduct(name="Butter", page=1, sku="A"),
        CandidateProduct(name="Butter", page=1, sku="A"),
        CandidateProduct(name="Butter", page=2, sku="A"),
        CandidateProduct(name="Cheese", page=1, sku="A"),
    ]
    out = _dedupe_candidates(items)
    assert len(out) == 3


def _tool_response(arguments: str, finish_reason: str = "tool_calls"):
    tool_call = SimpleNamespace(
        function=SimpleNamespace(name="submit_products", arguments=arguments)
    )
    message = SimpleNamespace(tool_calls=[tool_call], content=None)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


def test_extract_chunk_retries_once_on_bad_json(monkeypatch):
    monkeypatch.setattr(settings, "deepseek_api_key", "test-key")
    monkeypatch.setattr(settings, "extract_max_tokens", 8192)

    good = '{"products": [{"name": "Amul Ghee", "page": 1}]}'
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _tool_response('{"products": [{"name": "bad'),
        _tool_response(good),
    ]

    doc = ParsedDocument(
        source_filename="x.pdf",
        source_path="x.pdf",
        page_count=1,
        pages=[_page(1, "Amul Ghee")],
    )
    products = _extract_chunk(client, doc, doc.pages, chunk_index=1, chunk_total=1)
    assert len(products) == 1
    assert products[0].name == "Amul Ghee"
    assert client.chat.completions.create.call_count == 2


def test_extract_products_chunks_and_merges(monkeypatch):
    monkeypatch.setattr(settings, "deepseek_api_key", "test-key")
    monkeypatch.setattr(settings, "extract_pages_per_chunk", 2)
    monkeypatch.setattr(settings, "extract_max_tokens", 8192)

    responses = [
        _tool_response('{"products": [{"name": "P1", "page": 1}]}'),
        _tool_response('{"products": [{"name": "P3", "page": 3}]}'),
        _tool_response('{"products": [{"name": "P5", "page": 5}]}'),
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = responses

    monkeypatch.setattr("app.stages.extract._client", lambda: mock_client)

    doc = ParsedDocument(
        source_filename="cat.pdf",
        source_path="cat.pdf",
        page_count=5,
        pages=[_page(i, f"Product {i}") for i in range(1, 6)],
    )
    progress: list[str] = []

    def on_progress(stage, step, message, percent=None, detail=None):
        progress.append(message)

    products = extract_products(doc, on_progress=on_progress)
    assert [p.name for p in products] == ["P1", "P3", "P5"]
    assert mock_client.chat.completions.create.call_count == 3
    assert any("chunk 1/3" in m for m in progress)
    assert any("chunk 3/3" in m for m in progress)


def test_extract_requires_api_key(monkeypatch):
    monkeypatch.setattr(settings, "deepseek_api_key", "")
    doc = ParsedDocument(
        source_filename="x.png",
        source_path="x.png",
        page_count=1,
        pages=[_page(1)],
    )
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        extract_products(doc)


def test_format_pages_text_groups_same_row_blocks():
    page = ParsedPage(
        page_number=1,
        blocks=[
            TextBlock(block_id="p1-b0", text="Veg noodles", reading_order=2, row_id=0),
            TextBlock(block_id="p1-b1", text="139/-", reading_order=0, row_id=0),
            TextBlock(block_id="p1-b2", text="109/-", reading_order=1, row_id=0),
            TextBlock(block_id="p1-b3", text="Footer note", reading_order=3, row_id=None),
        ],
    )
    text = _format_pages_text([page], 5000)
    assert "[p1-b1] 139/- | 109/- | Veg noodles" in text
    assert "[p1-b3] Footer note" in text
