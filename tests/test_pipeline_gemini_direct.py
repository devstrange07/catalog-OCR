"""Pipeline routing for gemini_direct vs two-stage gemini."""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import MagicMock

from app.config import settings
from app.models.document import ParsedDocument, ParsedPage, TextBlock
from app.models.product import CandidateProduct, ProductStatus
from app.pipeline import create_job_from_bytes, create_job_from_url, get_job, run_pipeline, run_url_pipeline
from app.stages.crawl import BinaryFetch, HtmlFetch
from app.stages.extract_gemini_direct import (
    QUALITY_WARNING_HOLLOW,
    DirectExtractResult,
    DirectQualityReport,
)
from app.stages.parse_gemini import parse_document_gemini as real_parse_document_gemini


def _doc(name: str = "x.png") -> ParsedDocument:
    return ParsedDocument(
        source_filename=name,
        source_path=name,
        page_count=1,
        pages=[
            ParsedPage(
                page_number=1,
                blocks=[TextBlock(block_id="p1-b0", text="Widget", reading_order=0)],
                full_text="Widget $1",
            )
        ],
    )


def _direct_result(
    candidates: list[CandidateProduct],
    *,
    quality: DirectQualityReport | None = None,
    call_count: int = 1,
) -> DirectExtractResult:
    if quality is None:
        quality = DirectQualityReport(total=len(candidates), suspicious=False)
    return DirectExtractResult(
        candidates=candidates,
        quality=quality,
        call_count=call_count,
    )


def test_run_pipeline_gemini_direct_skips_parse_and_stage2(tmp_path, monkeypatch):
    monkeypatch.setattr("app.pipeline.OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")

    called = {"parse": 0, "stage2": 0, "direct": 0}

    def fake_parse(*args, **kwargs):
        called["parse"] += 1
        return _doc()

    def fake_stage2(*args, **kwargs):
        called["stage2"] += 1
        return []

    def fake_direct(source, on_progress=None):
        called["direct"] += 1
        return _direct_result(
            [CandidateProduct(name="Direct Widget", price="2.50", page=1)]
        )

    monkeypatch.setattr("app.pipeline.parse_document_gemini", fake_parse)
    monkeypatch.setattr("app.pipeline.extract_products_gemini", fake_stage2)
    monkeypatch.setattr("app.pipeline.extract_products_gemini_direct", fake_direct)

    job = create_job_from_bytes("cat.png", b"abc", parse_channel="gemini_direct")
    run_pipeline(job.id)

    assert called == {"parse": 0, "stage2": 0, "direct": 1}
    out = tmp_path / job.id
    assert not (out / "parsed.json").exists()
    products = json.loads((out / "products.json").read_text(encoding="utf-8"))
    assert products[0]["name"] == "Direct Widget"
    assert "status" in products[0]
    candidates = json.loads((out / "candidates.json").read_text(encoding="utf-8"))
    assert candidates[0]["name"] == "Direct Widget"

    final = get_job(job.id)
    assert final is not None
    assert final.stages["parse"].status.value == "skipped"
    assert final.status.value == "completed"
    assert final.validated_count == 1


def test_run_pipeline_gemini_still_two_stage(tmp_path, monkeypatch):
    monkeypatch.setattr("app.pipeline.OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")

    called = {"parse": 0, "stage2": 0, "direct": 0}

    def fake_parse(*args, **kwargs):
        called["parse"] += 1
        return _doc()

    def fake_stage2(parsed, on_progress=None):
        called["stage2"] += 1
        assert parsed.pages[0].full_text.startswith("Widget")
        return [CandidateProduct(name="Two Stage", price="1.00", page=1)]

    def fake_direct(*args, **kwargs):
        called["direct"] += 1
        raise AssertionError("gemini_direct extract must not run for parse_channel=gemini")

    monkeypatch.setattr("app.pipeline.parse_document_gemini", fake_parse)
    monkeypatch.setattr("app.pipeline.extract_products_gemini", fake_stage2)
    monkeypatch.setattr("app.pipeline.extract_products_gemini_direct", fake_direct)

    job = create_job_from_bytes("cat.png", b"abc", parse_channel="gemini")
    run_pipeline(job.id)

    assert called == {"parse": 1, "stage2": 1, "direct": 0}
    out = tmp_path / job.id
    assert (out / "parsed.json").exists()
    products = json.loads((out / "products.json").read_text(encoding="utf-8"))
    assert products[0]["name"] == "Two Stage"
    # real function still exists for the two-stage channel
    assert callable(real_parse_document_gemini)


def test_run_url_pipeline_gemini_direct_html_uses_text_extract(tmp_path, monkeypatch):
    monkeypatch.setattr("app.pipeline.OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")

    called = {"parse": 0, "stage2": 0, "direct_file": 0, "direct_text": 0}

    def fake_fetch(url, on_progress=None):
        return HtmlFetch(
            markdown="# Widget\nPrice 10",
            title="Page",
            source_url=url,
            final_url=url,
        )

    monkeypatch.setattr("app.pipeline.fetch_url", fake_fetch)
    monkeypatch.setattr(
        "app.pipeline.parse_document_gemini",
        lambda *a, **k: called.__setitem__("parse", called["parse"] + 1) or _doc(),
    )
    monkeypatch.setattr(
        "app.pipeline.extract_products_gemini",
        lambda *a, **k: called.__setitem__("stage2", called["stage2"] + 1) or [],
    )
    monkeypatch.setattr(
        "app.pipeline.extract_products_gemini_direct",
        lambda *a, **k: called.__setitem__("direct_file", called["direct_file"] + 1)
        or _direct_result([]),
    )

    def fake_text(markdown, on_progress=None):
        called["direct_text"] += 1
        assert "Widget" in markdown
        return _direct_result(
            [CandidateProduct(name="Crawled Item", price="10", page=1)]
        )

    monkeypatch.setattr("app.pipeline.extract_products_gemini_direct_from_text", fake_text)

    job = create_job_from_url("https://example.com/menu", parse_channel="gemini_direct")
    run_url_pipeline(job.id)

    assert called["parse"] == 0
    assert called["stage2"] == 0
    assert called["direct_file"] == 0
    assert called["direct_text"] == 1
    products = json.loads((tmp_path / job.id / "products.json").read_text(encoding="utf-8"))
    assert products[0]["name"] == "Crawled Item"
    assert not (tmp_path / job.id / "parsed.json").exists()
    assert (tmp_path / job.id / "crawl.json").exists()


def test_run_url_pipeline_gemini_direct_binary_uses_file_extract(tmp_path, monkeypatch):
    monkeypatch.setattr("app.pipeline.OUTPUTS_DIR", tmp_path)

    def fake_fetch(url, on_progress=None):
        return BinaryFetch(
            data=b"%PDF-1.4",
            filename="cat.pdf",
            content_type="application/pdf",
            source_url=url,
            final_url=url,
        )

    monkeypatch.setattr("app.pipeline.fetch_url", fake_fetch)
    monkeypatch.setattr("app.pipeline.parse_document_gemini", MagicMock(side_effect=AssertionError("parse")))
    monkeypatch.setattr("app.pipeline.extract_products_gemini", MagicMock(side_effect=AssertionError("stage2")))

    def fake_direct(source, on_progress=None):
        assert source.name == "cat.pdf"
        return _direct_result([CandidateProduct(name="PDF Item", price="3", page=2)])

    monkeypatch.setattr("app.pipeline.extract_products_gemini_direct", fake_direct)

    job = create_job_from_url("https://example.com/cat.pdf", parse_channel="gemini_direct")
    run_url_pipeline(job.id)
    products = json.loads((tmp_path / job.id / "products.json").read_text(encoding="utf-8"))
    assert products[0]["name"] == "PDF Item"
    final = get_job(job.id)
    assert final.page_count == 2


def test_run_pipeline_observability_span_calls(tmp_path, monkeypatch):
    monkeypatch.setattr("app.pipeline.OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")

    called_spans: list[str] = []

    @contextmanager
    def fake_span(name, **kwargs):
        called_spans.append(name)
        yield None

    monkeypatch.setattr("app.pipeline.trace_span", fake_span)
    monkeypatch.setattr("app.pipeline.parse_document_gemini", lambda *a, **k: _doc())
    monkeypatch.setattr(
        "app.pipeline.extract_products_gemini",
        lambda *a, **k: [CandidateProduct(name="Two Stage", price="1.00", page=1)],
    )

    job = create_job_from_bytes("cat.png", b"abc", parse_channel="gemini")
    run_pipeline(job.id)

    assert "pipeline.run" in called_spans
    assert "pipeline.stage.parse" in called_spans
    assert "pipeline.stage.extract" in called_spans
    assert "pipeline.stage.validate" in called_spans


def test_run_pipeline_direct_quality_warning_keeps_validated(tmp_path, monkeypatch):
    """Schema-valid but incomplete direct output stays validated with quality warnings."""
    monkeypatch.setattr("app.pipeline.OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")

    quality = DirectQualityReport(
        total=3,
        sku_count=3,
        hollow_sku_count=3,
        hollow_sku_ratio=1.0,
        suspicious=True,
        reasons=["hollow_skus"],
        repaired=True,
        repair_improved=False,
    )

    def fake_direct(source, on_progress=None):
        return _direct_result(
            [
                CandidateProduct(name="2BHK Flat", sku="RY-0001"),
                CandidateProduct(name="3BHK Flat", sku="RY-0002"),
                CandidateProduct(name="4BHK Flat", sku="RY-0003"),
            ],
            quality=quality,
            call_count=2,
        )

    monkeypatch.setattr("app.pipeline.extract_products_gemini_direct", fake_direct)
    monkeypatch.setattr(
        "app.pipeline.parse_document_gemini",
        MagicMock(side_effect=AssertionError("parse")),
    )
    monkeypatch.setattr(
        "app.pipeline.extract_products_gemini",
        MagicMock(side_effect=AssertionError("stage2")),
    )

    job = create_job_from_bytes("cat.png", b"abc", parse_channel="gemini_direct")
    run_pipeline(job.id)

    products = json.loads((tmp_path / job.id / "products.json").read_text(encoding="utf-8"))
    assert all(p["status"] == ProductStatus.validated.value for p in products)
    assert all(QUALITY_WARNING_HOLLOW in p["warnings"] for p in products)
    final = get_job(job.id)
    assert final is not None
    assert final.validated_count == 3
    assert final.meta.get("direct_quality", {}).get("suspicious") is True
    assert not (tmp_path / job.id / "parsed.json").exists()
