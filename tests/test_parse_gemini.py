"""Gemini Stage-1 parser mapping and failure handling (no live API)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.config import settings
from app.stages.parse_gemini import (
    _bbox_from_list,
    _extract_json_text,
    parse_document_gemini,
)


def test_extract_json_text_plain():
    payload = {
        "candidates": [
            {"content": {"parts": [{"text": '{"blocks":[{"text":"Hi"}]}'}]}}
        ]
    }
    assert json.loads(_extract_json_text(payload))["blocks"][0]["text"] == "Hi"


def test_extract_json_text_fenced():
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": '```json\n{"blocks":[{"text":"A"}]}\n```'}]
                }
            }
        ]
    }
    parsed = json.loads(_extract_json_text(payload))
    assert parsed["blocks"][0]["text"] == "A"


def test_extract_json_text_empty_raises():
    with pytest.raises(RuntimeError, match="no candidates"):
        _extract_json_text({})


def test_bbox_from_list():
    box = _bbox_from_list([1, 2, 10, 20])
    assert box is not None
    assert box.x0 == 1
    assert box.y1 == 20
    assert _bbox_from_list([1, 2]) is None
    assert _bbox_from_list("nope") is None


def test_parse_document_gemini_requires_api_key(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "")
    img_path = tmp_path / "page.png"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(img_path)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        parse_document_gemini(img_path)


def test_parse_document_gemini_maps_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(settings, "gemini_model", "gemini-2.5-flash")
    img_path = tmp_path / "page.png"
    Image.new("RGB", (12, 10), color=(255, 255, 255)).save(img_path)

    payload = {
        "blocks": [
            {"text": "Acme Widget", "bbox": [1, 2, 8, 9]},
            {"text": "SKU-1"},
            {"text": "   "},
        ]
    }
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]
    }
    fake_client = MagicMock()
    fake_client.__enter__.return_value = fake_client
    fake_client.__exit__.return_value = False
    fake_client.post.return_value = fake_resp

    with patch("app.stages.parse_gemini.httpx.Client", return_value=fake_client):
        doc = parse_document_gemini(img_path, source_filename="page.png")

    assert doc.page_count == 1
    assert doc.meta["ocr_engine"] == "gemini_vision"
    assert [b.text for b in doc.pages[0].blocks] == ["Acme Widget", "SKU-1"]
    assert doc.pages[0].blocks[0].bbox is not None
    assert doc.pages[0].blocks[0].bbox.x0 == 1
    assert "Gemini Vision" in doc.structure_notes[0]
    fake_client.post.assert_called_once()
