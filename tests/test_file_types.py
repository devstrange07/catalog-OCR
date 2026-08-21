"""Channel-aware file allowlists and MIME helpers."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from app.file_types import (
    GEMINI_TEXT_SUFFIXES,
    IMAGE_PDF_SUFFIXES,
    allowed_suffixes_for_channel,
    mime_for_suffix,
)
from app.main import app
from app.models.pipeline import ParseChannel


def test_mime_for_suffix_text_and_image():
    assert mime_for_suffix(".csv") == "text/csv"
    assert mime_for_suffix(".json") == "application/json"
    assert mime_for_suffix(".png") == "image/png"
    with pytest.raises(ValueError):
        mime_for_suffix(".xlsx")


def test_allowed_suffixes_channel_aware():
    paddle = allowed_suffixes_for_channel(ParseChannel.paddleocr)
    gemini = allowed_suffixes_for_channel(ParseChannel.gemini)
    assert paddle == IMAGE_PDF_SUFFIXES
    assert ".csv" not in paddle
    assert GEMINI_TEXT_SUFFIXES <= gemini
    assert IMAGE_PDF_SUFFIXES <= gemini


def test_api_rejects_csv_on_paddleocr(tmp_path, monkeypatch):
    monkeypatch.setattr("app.pipeline.run_pipeline", lambda job_id: None)
    client = TestClient(app)
    resp = client.post(
        "/api/jobs",
        files={"file": ("prices.csv", io.BytesIO(b"name,price\nA,1\n"), "text/csv")},
        data={"parse_channel": "paddleocr"},
    )
    assert resp.status_code == 400
    assert "csv" in resp.json()["detail"].lower()
    assert "gemini" in resp.json()["detail"].lower()


def test_api_accepts_csv_on_gemini(tmp_path, monkeypatch):
    monkeypatch.setattr("app.pipeline.run_pipeline", lambda job_id: None)
    client = TestClient(app)
    resp = client.post(
        "/api/jobs",
        files={"file": ("prices.csv", io.BytesIO(b"name,price\nA,1\n"), "text/csv")},
        data={"parse_channel": "gemini"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["meta"]["parse_channel"] == "gemini"
    assert body["filename"] == "prices.csv"


def test_api_rejects_xlsx_even_on_gemini(monkeypatch):
    monkeypatch.setattr("app.pipeline.run_pipeline", lambda job_id: None)
    client = TestClient(app)
    resp = client.post(
        "/api/jobs",
        files={
            "file": (
                "sheet.xlsx",
                io.BytesIO(b"PK\x03\x04fake"),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
        data={"parse_channel": "gemini"},
    )
    assert resp.status_code == 400
    assert "xlsx" in resp.json()["detail"].lower()


def test_api_url_rejects_csv_on_paddleocr(monkeypatch):
    monkeypatch.setattr("app.pipeline.run_url_pipeline", lambda job_id: None)
    client = TestClient(app)
    resp = client.post(
        "/api/jobs/from-url",
        json={
            "url": "https://example.com/catalog.csv",
            "parse_channel": "paddleocr",
        },
    )
    assert resp.status_code == 400
    assert "csv" in resp.json()["detail"].lower()
