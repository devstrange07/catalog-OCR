"""Parse-channel validation, job metadata, and Stage-1 dispatch."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.models.document import ParsedDocument, ParsedPage, TextBlock
from app.models.pipeline import ParseChannel
from app.pipeline import (
    _parse_document_for_channel,
    _resolve_parse_channel,
    create_job_from_bytes,
    create_job_from_url,
)


def _doc(name: str = "x.png") -> ParsedDocument:
    return ParsedDocument(
        source_filename=name,
        source_path=name,
        page_count=1,
        pages=[
            ParsedPage(
                page_number=1,
                blocks=[TextBlock(block_id="p1-b0", text="Widget", reading_order=0)],
                full_text="Widget",
            )
        ],
    )


def test_resolve_parse_channel_default(monkeypatch):
    monkeypatch.setattr(settings, "parse_channel_default", "paddleocr")
    assert _resolve_parse_channel(None) == ParseChannel.paddleocr


def test_resolve_parse_channel_enum_and_string():
    assert _resolve_parse_channel("gemini") == ParseChannel.gemini
    assert _resolve_parse_channel("gemini_direct") == ParseChannel.gemini_direct
    assert _resolve_parse_channel(ParseChannel.paddleocr) == ParseChannel.paddleocr


def test_resolve_parse_channel_rejects_unknown():
    with pytest.raises(ValueError, match="Unsupported parse channel"):
        _resolve_parse_channel("tesseract")


def test_create_job_from_bytes_stores_channel(tmp_path, monkeypatch):
    monkeypatch.setattr("app.pipeline.OUTPUTS_DIR", tmp_path)
    job = create_job_from_bytes("cat.png", b"abc", parse_channel="gemini")
    assert job.meta["parse_channel"] == "gemini"
    assert job.meta["source_kind"] == "upload"


def test_create_job_from_url_defaults_to_paddle(tmp_path, monkeypatch):
    monkeypatch.setattr("app.pipeline.OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(settings, "parse_channel_default", "paddleocr")
    job = create_job_from_url("https://example.com/cat.pdf")
    assert job.meta["parse_channel"] == "paddleocr"
    assert job.meta["source_kind"] == "url"


def test_parse_document_for_channel_dispatches_gemini(monkeypatch):
    called = {"gemini": False, "paddle": False}

    def fake_gemini(source, source_filename=None, on_progress=None):
        called["gemini"] = True
        return _doc(source_filename or "g.png")

    def fake_paddle(source, source_filename=None, on_progress=None):
        called["paddle"] = True
        return _doc(source_filename or "p.png")

    monkeypatch.setattr("app.pipeline.parse_document_gemini", fake_gemini)
    monkeypatch.setattr("app.pipeline.parse_document", fake_paddle)

    doc = _parse_document_for_channel(Path("x.png"), "x.png", ParseChannel.gemini, None)
    assert called == {"gemini": True, "paddle": False}
    assert doc.source_filename == "x.png"


def test_parse_document_for_channel_dispatches_paddle(monkeypatch):
    called = {"gemini": False, "paddle": False}

    def fake_gemini(source, source_filename=None, on_progress=None):
        called["gemini"] = True
        return _doc("g.png")

    def fake_paddle(source, source_filename=None, on_progress=None):
        called["paddle"] = True
        return _doc(source_filename or "p.png")

    monkeypatch.setattr("app.pipeline.parse_document_gemini", fake_gemini)
    monkeypatch.setattr("app.pipeline.parse_document", fake_paddle)

    _parse_document_for_channel(Path("x.png"), "x.png", ParseChannel.paddleocr, None)
    assert called == {"gemini": False, "paddle": True}


def test_api_from_url_rejects_unknown_channel():
    client = TestClient(app)
    res = client.post(
        "/api/jobs/from-url",
        json={"url": "https://example.com/cat.pdf", "parse_channel": "tesseract"},
    )
    assert res.status_code == 422


def test_api_upload_accepts_gemini_channel(tmp_path, monkeypatch):
    monkeypatch.setattr("app.pipeline.OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr("app.api.routes.pipe.run_pipeline", MagicMock())
    client = TestClient(app)
    res = client.post(
        "/api/jobs",
        files={"file": ("cat.png", b"abc", "image/png")},
        data={"parse_channel": "gemini"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["meta"]["parse_channel"] == "gemini"


def test_api_upload_accepts_gemini_direct_channel(tmp_path, monkeypatch):
    monkeypatch.setattr("app.pipeline.OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr("app.api.routes.pipe.run_pipeline", MagicMock())
    client = TestClient(app)
    res = client.post(
        "/api/jobs",
        files={"file": ("cat.png", b"abc", "image/png")},
        data={"parse_channel": "gemini_direct"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["meta"]["parse_channel"] == "gemini_direct"


def test_parse_document_for_channel_rejects_gemini_direct():
    with pytest.raises(RuntimeError, match="does not produce a ParsedDocument"):
        _parse_document_for_channel(Path("x.png"), "x.png", ParseChannel.gemini_direct, None)
