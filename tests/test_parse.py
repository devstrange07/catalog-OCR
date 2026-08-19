"""Stage 1 parse runtime selection and fallback behavior."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import app.stages.parse as parse_stage
from app.config import settings
from app.models.document import ParsedPage, TextBlock
from PIL import Image


def test_get_ocr_uses_cuda_when_enabled(monkeypatch):
    calls = []
    fake_engine = object()

    def fake_paddleocr(**kwargs):
        calls.append(kwargs)
        return fake_engine

    monkeypatch.setattr(settings, "ocr_use_gpu", True)
    monkeypatch.setattr(settings, "ocr_gpu_id", 2)
    monkeypatch.setattr(parse_stage, "_ocr_engine", None)
    monkeypatch.setattr(parse_stage, "_ocr_runtime", "cpu")
    monkeypatch.setattr(parse_stage, "_paddle_has_cuda", lambda: True)
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=fake_paddleocr))

    engine = parse_stage._get_ocr()

    assert engine is fake_engine
    assert calls[0]["device"] == "gpu:2"
    assert calls[0]["lang"] == settings.ocr_lang
    assert parse_stage._ocr_runtime == "cuda"
    assert parse_stage.os.environ["CUDA_VISIBLE_DEVICES"] == "2"


def test_get_ocr_falls_back_to_cpu_when_cuda_init_fails(monkeypatch):
    calls = []
    fake_engine = object()

    def fake_paddleocr(**kwargs):
        calls.append(kwargs)
        if str(kwargs.get("device", "")).startswith("gpu"):
            raise RuntimeError("cuda init failed")
        return fake_engine

    monkeypatch.setattr(settings, "ocr_use_gpu", True)
    monkeypatch.setattr(settings, "ocr_gpu_id", 0)
    monkeypatch.setattr(parse_stage, "_ocr_engine", None)
    monkeypatch.setattr(parse_stage, "_ocr_runtime", "cpu")
    monkeypatch.setattr(parse_stage, "_paddle_has_cuda", lambda: True)
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=fake_paddleocr))

    engine = parse_stage._get_ocr()

    assert engine is fake_engine
    assert len(calls) == 2
    assert calls[0]["device"] == "gpu:0"
    assert calls[1]["device"] == "cpu"
    assert parse_stage._ocr_runtime == "cpu"


def test_get_ocr_skips_gpu_when_paddle_lacks_cuda(monkeypatch):
    calls = []
    fake_engine = object()

    def fake_paddleocr(**kwargs):
        calls.append(kwargs)
        return fake_engine

    monkeypatch.setattr(settings, "ocr_use_gpu", True)
    monkeypatch.setattr(settings, "ocr_gpu_id", 0)
    monkeypatch.setattr(parse_stage, "_ocr_engine", None)
    monkeypatch.setattr(parse_stage, "_ocr_runtime", "cpu")
    monkeypatch.setattr(parse_stage, "_paddle_has_cuda", lambda: False)
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=fake_paddleocr))

    engine = parse_stage._get_ocr()

    assert engine is fake_engine
    assert len(calls) == 1
    assert calls[0]["device"] == "cpu"
    assert parse_stage._ocr_runtime == "cpu"


def test_parse_document_emits_effective_runtime_meta(tmp_path, monkeypatch):
    img_path = tmp_path / "page.png"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(img_path)

    monkeypatch.setattr(parse_stage, "_ocr_engine", object())
    monkeypatch.setattr(parse_stage, "_ocr_runtime", "cuda")
    monkeypatch.setattr(parse_stage, "_get_ocr", lambda: parse_stage._ocr_engine)
    monkeypatch.setattr(parse_stage, "_docling_structure", lambda _p: (None, []))
    monkeypatch.setattr(
        parse_stage,
        "_ocr_page",
        lambda _img, page_number: ParsedPage(
            page_number=page_number,
            width=8,
            height=8,
            blocks=[TextBlock(block_id=f"p{page_number}-b0", text="Sample", reading_order=0)],
            full_text="Sample",
        ),
    )
    monkeypatch.setattr(settings, "ocr_use_gpu", True)
    monkeypatch.setattr(settings, "ocr_gpu_id", 1)

    doc = parse_stage.parse_document(img_path)
    assert doc.meta["ocr_runtime"] == "cuda"
    assert doc.meta["ocr_use_gpu_requested"] is True
    assert doc.meta["ocr_gpu_id_requested"] == 1
