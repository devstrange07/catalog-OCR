"""Tests for Unlimited-OCR parse stage."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.config import settings
from app.models.document import ParsedDocument


def _fake_torch_no_cuda():
    """Fake torch module with cuda unavailable."""
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: False,
            set_device=lambda x: None,
        ),
        bfloat16="bfloat16",
    )


def test_get_model_raises_without_cuda(monkeypatch):
    import app.stages.parse_unlimited_ocr as mod

    monkeypatch.setattr(mod, "_model", None)
    monkeypatch.setattr(mod, "_tokenizer", None)

    fake_torch = _fake_torch_no_cuda()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with patch.dict(sys.modules, {"torch": fake_torch}):
        with pytest.raises(RuntimeError, match="CUDA GPU required"):
            mod._get_model()


def test_parse_builds_correct_document_shape(monkeypatch, tmp_path):
    from PIL import Image

    import app.stages.parse_unlimited_ocr as mod

    img = Image.new("RGB", (100, 80), "white")
    img_path = tmp_path / "test.png"
    img.save(str(img_path))

    fake_model = SimpleNamespace()
    fake_tokenizer = object()

    def fake_infer(tokenizer, **kwargs):
        output_path = Path(kwargs["output_path"])
        (output_path / "result.md").write_text("# Hello\nSome text", encoding="utf-8")

    fake_model.infer = fake_infer
    monkeypatch.setattr(mod, "_model", fake_model)
    monkeypatch.setattr(mod, "_tokenizer", fake_tokenizer)

    def fake_get_model():
        return fake_model, fake_tokenizer

    monkeypatch.setattr(mod, "_get_model", fake_get_model)
    monkeypatch.setattr(settings, "unlimited_ocr_remote_url", "")

    result = mod.parse_document_unlimited_ocr(img_path)

    assert isinstance(result, ParsedDocument)
    assert result.page_count == 1
    assert len(result.pages) == 1
    page = result.pages[0]
    assert page.blocks == []
    assert page.full_text == "# Hello\nSome text"
    assert result.meta["ocr_engine"] == "unlimited_ocr"


def test_text_for_extraction_uses_full_text_fallback(monkeypatch, tmp_path):
    from PIL import Image

    import app.stages.parse_unlimited_ocr as mod

    img = Image.new("RGB", (100, 80), "white")
    img_path = tmp_path / "test.jpg"
    img.save(str(img_path))

    fake_model = SimpleNamespace()
    fake_tokenizer = object()

    def fake_infer(tokenizer, **kwargs):
        output_path = Path(kwargs["output_path"])
        (output_path / "result.md").write_text("Product A | $10\nProduct B | $20", encoding="utf-8")

    fake_model.infer = fake_infer
    monkeypatch.setattr(mod, "_model", fake_model)
    monkeypatch.setattr(mod, "_tokenizer", fake_tokenizer)
    monkeypatch.setattr(mod, "_get_model", lambda: (fake_model, fake_tokenizer))
    monkeypatch.setattr(settings, "unlimited_ocr_remote_url", "")

    result = mod.parse_document_unlimited_ocr(img_path)
    text = result.text_for_extraction()

    assert "--- Page 1 ---" in text
    assert "Product A | $10" in text
    assert "Product B | $20" in text


def test_remote_parse_skips_local_cuda(monkeypatch, tmp_path):
    from PIL import Image

    import app.stages.parse_unlimited_ocr as mod

    img = Image.new("RGB", (64, 48), "white")
    img_path = tmp_path / "remote.png"
    img.save(str(img_path))

    monkeypatch.setattr(settings, "unlimited_ocr_remote_url", "https://example.trycloudflare.com")
    monkeypatch.setattr(settings, "unlimited_ocr_remote_api_key", "secret")

    def boom_get_model():
        raise AssertionError("_get_model must not be called for remote path")

    monkeypatch.setattr(mod, "_get_model", boom_get_model)
    monkeypatch.setattr(
        mod,
        "_remote_health_check",
        lambda url: {"status": "ok", "device": "cuda", "vram_pct": 42},
    )
    monkeypatch.setattr(
        mod,
        "_infer_page_markdown_remote",
        lambda image, url: ("# Remote MD", {"device": "cuda", "vram_pct": 42, "elapsed_sec": 1.2}),
    )

    result = mod.parse_document_unlimited_ocr(img_path)

    assert result.pages[0].full_text == "# Remote MD"
    assert result.meta["unlimited_ocr_remote"] is True
    assert result.meta["unlimited_ocr_remote_url"] == "https://example.trycloudflare.com"


def test_remote_cpu_offload_emits_progress(monkeypatch, tmp_path):
    from PIL import Image

    import app.stages.parse_unlimited_ocr as mod

    img = Image.new("RGB", (64, 48), "white")
    img_path = tmp_path / "cpu.png"
    img.save(str(img_path))

    monkeypatch.setattr(settings, "unlimited_ocr_remote_url", "https://example.trycloudflare.com")
    monkeypatch.setattr(
        mod,
        "_remote_health_check",
        lambda url: {"status": "ok", "device": "cuda", "vram_pct": 81},
    )
    monkeypatch.setattr(
        mod,
        "_infer_page_markdown_remote",
        lambda image, url: (
            "cpu text",
            {"device": "cpu", "vram_pct": 85, "elapsed_sec": 9.0, "markdown": "cpu text"},
        ),
    )

    events: list[tuple] = []

    def on_progress(stage, step, message, percent=None, detail=None):
        events.append((stage, step, message, detail))

    result = mod.parse_document_unlimited_ocr(img_path, on_progress=on_progress)

    assert result.pages[0].full_text == "cpu text"
    offload = [e for e in events if e[1] == "cpu_offload"]
    assert len(offload) == 1
    assert "CPU" in offload[0][2]
    assert offload[0][3]["device"] == "cpu"


def test_remote_health_unreachable_raises(monkeypatch, tmp_path):
    from PIL import Image

    import app.stages.parse_unlimited_ocr as mod

    img = Image.new("RGB", (32, 32), "white")
    img_path = tmp_path / "dead.png"
    img.save(str(img_path))

    monkeypatch.setattr(settings, "unlimited_ocr_remote_url", "https://dead.trycloudflare.com")

    def fail_health(url):
        raise RuntimeError(
            f"Colab Unlimited-OCR endpoint unreachable at {url}. "
            "Check the notebook is running and UNLIMITED_OCR_REMOTE_URL is up to date "
            "(the tunnel URL changes on every Colab restart)."
        )

    monkeypatch.setattr(mod, "_remote_health_check", fail_health)

    with pytest.raises(RuntimeError, match="unreachable"):
        mod.parse_document_unlimited_ocr(img_path)


def test_remote_health_check_wraps_http_errors(monkeypatch):
    import app.stages.parse_unlimited_ocr as mod

    class BoomClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, *args, **kwargs):
            raise ConnectionError("refused")

    monkeypatch.setattr(mod.httpx, "Client", BoomClient)

    with pytest.raises(RuntimeError, match="unreachable"):
        mod._remote_health_check("https://example.trycloudflare.com")
