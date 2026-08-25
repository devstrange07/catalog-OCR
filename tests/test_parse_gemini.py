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
            {"content": {"parts": [{"text": '{"pages":[{"page_number":1,"blocks":[{"text":"Hi"}]}]}'}]}}
        ]
    }
    assert json.loads(_extract_json_text(payload))["pages"][0]["blocks"][0]["text"] == "Hi"


def test_extract_json_text_fenced():
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": '```json\n{"pages":[{"page_number":1,"blocks":[{"text":"A"}]}]}\n```'
                        }
                    ]
                }
            }
        ]
    }
    parsed = json.loads(_extract_json_text(payload))
    assert parsed["pages"][0]["blocks"][0]["text"] == "A"


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


def test_mime_for_path_uses_shared_map(tmp_path):
    from app.stages.parse_gemini import _mime_for_path

    assert _mime_for_path(tmp_path / "a.csv") == "text/csv"
    assert _mime_for_path(tmp_path / "a.json") == "application/json"
    assert _mime_for_path(tmp_path / "a.pdf") == "application/pdf"


def test_parse_document_gemini_requires_api_key(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "")
    img_path = tmp_path / "page.png"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(img_path)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        parse_document_gemini(img_path)


def _fake_client(post_resp: MagicMock) -> MagicMock:
    fake_client = MagicMock()
    fake_client.__enter__.return_value = fake_client
    fake_client.__exit__.return_value = False
    fake_client.post.return_value = post_resp
    return fake_client


def test_parse_document_gemini_maps_pages_inline_image(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(settings, "gemini_model", "gemini-2.5-flash")
    img_path = tmp_path / "page.png"
    Image.new("RGB", (12, 10), color=(255, 255, 255)).save(img_path)

    payload = {
        "pages": [
            {
                "page_number": 1,
                "blocks": [
                    {"text": "Acme Widget", "bbox": [1, 2, 8, 9]},
                    {"text": "SKU-1"},
                    {"text": "   "},
                ],
            }
        ]
    }
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]
    }
    fake_client = _fake_client(fake_resp)

    with patch("app.stages.parse_gemini.httpx.Client", return_value=fake_client):
        doc = parse_document_gemini(img_path, source_filename="page.png")

    assert doc.page_count == 1
    assert doc.meta["ocr_engine"] == "gemini_vision"
    assert doc.meta["input_mode"] == "direct_document"
    assert doc.meta["transport"] == "inline"
    assert [b.text for b in doc.pages[0].blocks] == ["Acme Widget", "SKU-1"]
    assert doc.pages[0].blocks[0].bbox is not None
    assert doc.pages[0].blocks[0].bbox.x0 == 1
    assert doc.pages[0].width is None
    assert "direct document pass" in doc.structure_notes[0]

    fake_client.post.assert_called_once()
    call_kwargs = fake_client.post.call_args
    body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    parts = body["contents"][0]["parts"]
    assert "inline_data" in parts[1]
    assert parts[1]["inline_data"]["mime_type"] == "image/png"
    assert "data" in parts[1]["inline_data"]


def test_parse_document_gemini_pdf_inline(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(settings, "gemini_model", "gemini-2.5-flash")
    pdf_path = tmp_path / "doc.pdf"
    # Minimal PDF bytes (not a valid renderable PDF; Gemini is mocked)
    pdf_path.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n")

    payload = {
        "pages": [
            {"page_number": 1, "blocks": [{"text": "Page One"}]},
            {"page_number": 2, "blocks": [{"text": "Page Two"}]},
        ]
    }
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]
    }
    fake_client = _fake_client(fake_resp)

    with patch("app.stages.parse_gemini.httpx.Client", return_value=fake_client):
        doc = parse_document_gemini(pdf_path)

    assert doc.page_count == 2
    assert [p.full_text for p in doc.pages] == ["Page One", "Page Two"]
    assert doc.meta["transport"] == "inline"

    body = fake_client.post.call_args.kwargs["json"]
    assert body["contents"][0]["parts"][1]["inline_data"]["mime_type"] == "application/pdf"


def test_parse_document_gemini_files_api_for_large_file(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(settings, "gemini_model", "gemini-2.5-flash")
    monkeypatch.setattr(settings, "gemini_inline_max_bytes", 100)

    pdf_path = tmp_path / "big.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n" + (b"x" * 200))

    upload_resp = MagicMock()
    upload_resp.status_code = 200
    upload_resp.json.return_value = {
        "file": {
            "name": "files/abc123",
            "uri": "https://generativelanguage.googleapis.com/v1beta/files/abc123",
            "state": "ACTIVE",
        }
    }

    gen_payload = {
        "pages": [{"page_number": 1, "blocks": [{"text": "From Files API"}]}]
    }
    gen_resp = MagicMock()
    gen_resp.status_code = 200
    gen_resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": json.dumps(gen_payload)}]}}]
    }

    delete_resp = MagicMock()
    delete_resp.status_code = 200

    fake_client = MagicMock()
    fake_client.__enter__.return_value = fake_client
    fake_client.__exit__.return_value = False

    def post_side_effect(url, **kwargs):
        if "upload" in str(url):
            return upload_resp
        return gen_resp

    fake_client.post.side_effect = post_side_effect
    fake_client.delete.return_value = delete_resp

    with patch("app.stages.parse_gemini.httpx.Client", return_value=fake_client):
        doc = parse_document_gemini(pdf_path)

    assert doc.meta["transport"] == "files_api"
    assert doc.pages[0].blocks[0].text == "From Files API"

    # Upload + generateContent
    assert fake_client.post.call_count == 2
    gen_call = fake_client.post.call_args_list[1]
    body = gen_call.kwargs["json"]
    file_part = body["contents"][0]["parts"][1]
    assert "file_data" in file_part
    assert file_part["file_data"]["mime_type"] == "application/pdf"
    assert "files/abc123" in file_part["file_data"]["file_uri"]
    fake_client.delete.assert_called_once()
