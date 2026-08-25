"""DeepSeek Vision direct extraction — mocked OpenAI SDK, no live API."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.config import settings
from app.models.product import CandidateProduct
from app.stages.extract import TOOL_NAME
from app.stages.extract_deepseek_direct import (
    PROMPT_VERSION,
    extract_products_deepseek_direct,
)


def _tool_response(products: list[dict] | str, *, finish_reason: str = "tool_calls") -> MagicMock:
    if isinstance(products, str):
        args = products
    else:
        args = json.dumps({"products": products})
    tool_call = SimpleNamespace(
        function=SimpleNamespace(name=TOOL_NAME, arguments=args),
    )
    choice = SimpleNamespace(
        finish_reason=finish_reason,
        message=SimpleNamespace(tool_calls=[tool_call]),
    )
    return SimpleNamespace(choices=[choice])


def test_deepseek_direct_requires_api_key(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "deepseek_api_key", "")
    img_path = tmp_path / "page.png"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(img_path)
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        extract_products_deepseek_direct(img_path)


def test_deepseek_direct_extract_uses_files_api_and_tool_call(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "deepseek_api_key", "test-key")
    monkeypatch.setattr(settings, "deepseek_vision_model", "deepseek-v4-flash-vision-exp")
    img_path = tmp_path / "page.png"
    Image.new("RGB", (12, 10), color=(255, 255, 255)).save(img_path)

    fake_oa = MagicMock()
    fake_oa.files.create.return_value = SimpleNamespace(id="file-api-test123")
    fake_oa.chat.completions.create.return_value = _tool_response(
        [{"name": "Widget", "price": "9.99", "currency": "USD", "page": 1}]
    )

    with patch("app.stages.deepseek_transport.OpenAI", return_value=fake_oa):
        candidates = extract_products_deepseek_direct(img_path)

    assert len(candidates) == 1
    assert isinstance(candidates[0], CandidateProduct)
    assert candidates[0].name == "Widget"
    assert candidates[0].price == "9.99"

    fake_oa.files.create.assert_called_once()
    create_kwargs = fake_oa.files.create.call_args.kwargs
    assert create_kwargs["purpose"] == "user_data"

    chat_kwargs = fake_oa.chat.completions.create.call_args.kwargs
    assert chat_kwargs["model"] == "deepseek-v4-flash-vision-exp"
    assert chat_kwargs["tool_choice"]["function"]["name"] == TOOL_NAME
    messages = chat_kwargs["messages"]
    user_content = messages[1]["content"]
    assert any(p.get("type") == "file" and p.get("file_id") == "file-api-test123" for p in user_content)
    assert not any("base64" in str(p).lower() for p in user_content)
    assert PROMPT_VERSION == "deepseek_direct_v1"

    fake_oa.files.delete.assert_called_once_with("file-api-test123")


def test_deepseek_direct_empty_products_is_success(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "deepseek_api_key", "test-key")
    img_path = tmp_path / "page.png"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(img_path)

    fake_oa = MagicMock()
    fake_oa.files.create.return_value = SimpleNamespace(id="file-api-empty")
    fake_oa.chat.completions.create.return_value = _tool_response([])

    with patch("app.stages.deepseek_transport.OpenAI", return_value=fake_oa):
        candidates = extract_products_deepseek_direct(img_path)
    assert candidates == []
    fake_oa.files.delete.assert_called_once_with("file-api-empty")


def test_deepseek_direct_rejects_pdf(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "deepseek_api_key", "test-key")
    pdf_path = tmp_path / "catalog.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_products_deepseek_direct(pdf_path)


def test_deepseek_direct_malformed_json_retries_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "deepseek_api_key", "test-key")
    img_path = tmp_path / "page.png"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(img_path)

    fake_oa = MagicMock()
    fake_oa.files.create.return_value = SimpleNamespace(id="file-api-retry")
    fake_oa.chat.completions.create.side_effect = [
        _tool_response("{not json"),
        _tool_response([{"name": "Fixed", "price": "1"}]),
    ]

    with patch("app.stages.deepseek_transport.OpenAI", return_value=fake_oa):
        candidates = extract_products_deepseek_direct(img_path)

    assert len(candidates) == 1
    assert candidates[0].name == "Fixed"
    assert fake_oa.chat.completions.create.call_count == 2
    fake_oa.files.delete.assert_called_once_with("file-api-retry")
