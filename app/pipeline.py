"""Pipeline orchestrator — runs stages in order and persists artifacts."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.config import OUTPUTS_DIR, settings
from app.models.document import ParsedDocument
from app.models.pipeline import JobRecord, JobStatus, ParseChannel, ProgressEvent, StageInfo, StageStatus
from app.models.product import CandidateProduct, ProductStatus
from app.observability.langsmith import trace_span
from app.stages.crawl import (
    BinaryFetch,
    CrawlError,
    HtmlFetch,
    crawl_result_to_artifact,
    fetch_url,
    normalize_url,
    parsed_document_from_markdown,
)
from app.stages.extract_gemini import extract_products_gemini
from app.stages.extract_gemini_direct import (
    DirectQualityReport,
    apply_direct_quality_warnings,
    extract_products_gemini_direct,
    extract_products_gemini_direct_from_text,
)
from app.stages.parse import parse_document
from app.stages.parse_gemini import parse_document_gemini
from app.stages.parse_unlimited_ocr import parse_document_unlimited_ocr
from app.stages.validate import validate_products

logger = logging.getLogger(__name__)

_jobs: dict[str, JobRecord] = {}
_lock = threading.Lock()

STAGE_CRAWL = "crawl"
STAGE_PARSE = "parse"
STAGE_EXTRACT = "extract"
STAGE_VALIDATE = "validate"

PROGRESS_LOG_CAP = 80

ProgressCallback = Callable[..., None]


def _resolve_parse_channel(value: str | ParseChannel | None) -> ParseChannel:
    raw = value.value if isinstance(value, ParseChannel) else value
    if raw is None:
        raw = settings.parse_channel_default
    normalized = str(raw).strip().lower()
    try:
        return ParseChannel(normalized)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported parse channel '{raw}'. Use paddleocr, gemini, gemini_direct, or unlimited_ocr."
        ) from exc


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _job_dir(job_id: str) -> Path:
    path = OUTPUTS_DIR / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )


def _default_stages(*, include_crawl: bool = False) -> dict[str, StageInfo]:
    stages: dict[str, StageInfo] = {}
    if include_crawl:
        stages[STAGE_CRAWL] = StageInfo(name="URL Crawl")
    stages[STAGE_PARSE] = StageInfo(name="Document Parsing")
    stages[STAGE_EXTRACT] = StageInfo(name="Product Extraction")
    stages[STAGE_VALIDATE] = StageInfo(name="Validation")
    return stages


def get_job(job_id: str) -> JobRecord | None:
    with _lock:
        job = _jobs.get(job_id)
        return job.model_copy(deep=True) if job else None


def list_jobs() -> list[JobRecord]:
    with _lock:
        return [j.model_copy(deep=True) for j in sorted(_jobs.values(), key=lambda x: x.created_at, reverse=True)]


def _update_job(job_id: str, **kwargs: Any) -> JobRecord:
    with _lock:
        job = _jobs[job_id]
        data = job.model_dump()
        data.update(kwargs)
        data["updated_at"] = _now()
        updated = JobRecord.model_validate(data)
        _jobs[job_id] = updated
        return updated.model_copy(deep=True)


def _set_stage(job_id: str, stage_key: str, status: StageStatus, error: str | None = None) -> None:
    with _lock:
        job = _jobs[job_id]
        stage = job.stages[stage_key]
        stage.status = status
        stage.error = error
        if status == StageStatus.running:
            stage.started_at = _now()
        if status in {StageStatus.completed, StageStatus.failed, StageStatus.skipped}:
            stage.finished_at = _now()
        job.updated_at = _now()


def _emit_progress(
    job_id: str,
    stage: str,
    step: str,
    message: str,
    percent: int | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    event = ProgressEvent(
        ts=_now(),
        stage=stage,
        step=step,
        message=message,
        percent=percent,
        detail=detail,
    )
    with _lock:
        job = _jobs[job_id]
        job.progress = event
        job.progress_log.append(event)
        if len(job.progress_log) > PROGRESS_LOG_CAP:
            job.progress_log = job.progress_log[-PROGRESS_LOG_CAP:]
        job.updated_at = _now()


def _make_progress_cb(job_id: str) -> ProgressCallback:
    def on_progress(
        stage: str,
        step: str,
        message: str,
        percent: int | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        _emit_progress(job_id, stage, step, message, percent=percent, detail=detail)

    return on_progress


def create_job_from_bytes(
    filename: str,
    data: bytes,
    parse_channel: str | ParseChannel | None = None,
) -> JobRecord:
    job_id = uuid.uuid4().hex[:12]
    dest_dir = _job_dir(job_id)
    path = dest_dir / filename
    path.write_bytes(data)
    channel = _resolve_parse_channel(parse_channel)
    job = JobRecord(
        id=job_id,
        filename=filename,
        status=JobStatus.queued,
        stages=_default_stages(),
        meta={
            "source_path": str(path),
            "source_kind": "upload",
            "parse_channel": channel.value,
        },
    )
    with _lock:
        _jobs[job_id] = job
    _emit_progress(job_id, "pipeline", "queued", "Job queued — waiting to start…", percent=0)
    _write_json(dest_dir / "job.json", get_job(job_id).model_dump(mode="json"))  # type: ignore[union-attr]
    return get_job(job_id)  # type: ignore[return-value]


def create_job_from_url(url: str, parse_channel: str | ParseChannel | None = None) -> JobRecord:
    cleaned = normalize_url(url)
    job_id = uuid.uuid4().hex[:12]
    dest_dir = _job_dir(job_id)
    host = cleaned.split("://", 1)[-1].split("/", 1)[0]
    filename = f"url:{host}"
    channel = _resolve_parse_channel(parse_channel)
    job = JobRecord(
        id=job_id,
        filename=filename,
        status=JobStatus.queued,
        stages=_default_stages(include_crawl=True),
        meta={
            "source_url": cleaned,
            "source_kind": "url",
            "parse_channel": channel.value,
        },
    )
    with _lock:
        _jobs[job_id] = job
    _emit_progress(job_id, "pipeline", "queued", "URL job queued — waiting to start…", percent=0)
    _write_json(dest_dir / "job.json", get_job(job_id).model_dump(mode="json"))  # type: ignore[union-attr]
    return get_job(job_id)  # type: ignore[return-value]


def _fail_pipeline(job_id: str, exc: Exception, stage_order: list[str]) -> None:
    with trace_span(
        "pipeline.fail",
        run_type="tool",
        metadata={
            "job_id": job_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
        },
        tags=["pipeline", "failure"],
    ):
        pass
    logger.exception("Pipeline failed for job %s", job_id)
    current = get_job(job_id)
    failed_stage = None
    if current:
        for key, stage in current.stages.items():
            if stage.status == StageStatus.running:
                failed_stage = key
                break
    if failed_stage:
        _set_stage(job_id, failed_stage, StageStatus.failed, error=str(exc))
        if failed_stage in stage_order:
            for key in stage_order[stage_order.index(failed_stage) + 1 :]:
                if key in (current.stages if current else {}):
                    _set_stage(job_id, key, StageStatus.skipped, error="Skipped due to earlier failure")
    _emit_progress(
        job_id,
        failed_stage or "pipeline",
        "failed",
        f"Failed: {exc}",
        percent=None,
        detail={"error": str(exc)},
    )
    _update_job(job_id, status=JobStatus.failed, error=str(exc))


def _page_count_from_candidates(candidates: list[CandidateProduct]) -> int | None:
    pages = [c.page for c in candidates if c.page is not None]
    return max(pages) if pages else None


def _persist_candidates_and_validate(
    job_id: str,
    candidates: list[CandidateProduct],
    parsed: ParsedDocument | None,
    on_progress: ProgressCallback,
    *,
    quality: DirectQualityReport | None = None,
) -> None:
    out = _job_dir(job_id)
    _emit_progress(
        job_id,
        STAGE_EXTRACT,
        "saving_candidates",
        "Saving candidates.json…",
        percent=83,
        detail={"candidate_count": len(candidates)},
    )
    _write_json(
        out / "candidates.json",
        [c.model_dump(mode="json") for c in candidates],
    )
    _set_stage(job_id, STAGE_EXTRACT, StageStatus.completed)
    _update_job(job_id, candidate_count=len(candidates))
    _emit_progress(
        job_id,
        STAGE_EXTRACT,
        "extract_complete",
        f"Extraction complete — {len(candidates)} candidate(s)",
        percent=85,
        detail={"candidate_count": len(candidates)},
    )

    _update_job(job_id, status=JobStatus.validating)
    _set_stage(job_id, STAGE_VALIDATE, StageStatus.running)
    _emit_progress(
        job_id,
        STAGE_VALIDATE,
        "start",
        "Starting validation…",
        percent=86,
    )
    with trace_span(
        "pipeline.stage.validate",
        run_type="tool",
        metadata={"job_id": job_id, "stage": STAGE_VALIDATE, "candidate_count": len(candidates)},
        tags=["pipeline", "validate"],
    ):
        products = validate_products(candidates, parsed=parsed, on_progress=on_progress)
    if quality is not None:
        apply_direct_quality_warnings(products, quality)
        job = get_job(job_id)
        if job is not None:
            meta = dict(job.meta)
            meta["direct_quality"] = quality.as_dict()
            _update_job(job_id, meta=meta)
    _emit_progress(
        job_id,
        STAGE_VALIDATE,
        "saving_products",
        "Saving products.json…",
        percent=97,
    )
    _write_json(
        out / "products.json",
        [p.model_dump(mode="json") for p in products],
    )
    validated = sum(1 for p in products if p.status == ProductStatus.validated)
    rejected = sum(1 for p in products if p.status == ProductStatus.rejected)
    _set_stage(job_id, STAGE_VALIDATE, StageStatus.completed)
    _update_job(
        job_id,
        status=JobStatus.completed,
        products=products,
        validated_count=validated,
        rejected_count=rejected,
    )
    _emit_progress(
        job_id,
        "pipeline",
        "completed",
        f"Done — {validated} validated, {rejected} rejected",
        percent=100,
        detail={
            "validated": validated,
            "rejected": rejected,
            **({"direct_quality": quality.as_dict()} if quality is not None else {}),
        },
    )


def _run_extract_validate(
    job_id: str, parsed: ParsedDocument, on_progress: ProgressCallback
) -> None:
    _update_job(job_id, status=JobStatus.extracting)
    _set_stage(job_id, STAGE_EXTRACT, StageStatus.running)
    _emit_progress(
        job_id,
        STAGE_EXTRACT,
        "start",
        "Starting product extraction…",
        percent=56,
    )
    with trace_span(
        "pipeline.stage.extract",
        run_type="tool",
        metadata={"job_id": job_id, "stage": STAGE_EXTRACT, "channel": "gemini"},
        tags=["pipeline", "extract"],
    ):
        candidates = extract_products_gemini(parsed, on_progress=on_progress)
    _persist_candidates_and_validate(job_id, candidates, parsed, on_progress)


def _run_direct_extract_validate(
    job_id: str,
    on_progress: ProgressCallback,
    *,
    source: Path | None = None,
    markdown: str | None = None,
) -> None:
    _update_job(job_id, status=JobStatus.extracting)
    _set_stage(job_id, STAGE_EXTRACT, StageStatus.running)
    _emit_progress(
        job_id,
        STAGE_EXTRACT,
        "start",
        "Starting Gemini direct product extraction…",
        percent=56,
        detail={"parse_channel": ParseChannel.gemini_direct.value},
    )
    with trace_span(
        "pipeline.stage.extract",
        run_type="tool",
        metadata={
            "job_id": job_id,
            "stage": STAGE_EXTRACT,
            "channel": ParseChannel.gemini_direct.value,
            "source_type": "markdown" if markdown is not None else "file",
        },
        tags=["pipeline", "extract", "gemini_direct"],
    ):
        if markdown is not None:
            result = extract_products_gemini_direct_from_text(
                markdown, on_progress=on_progress
            )
        elif source is not None:
            result = extract_products_gemini_direct(source, on_progress=on_progress)
        else:
            raise RuntimeError("Gemini direct extraction requires a source file or crawl markdown.")
    _update_job(
        job_id,
        page_count=_page_count_from_candidates(result.candidates),
        parsed_preview=None,
    )
    _persist_candidates_and_validate(
        job_id,
        result.candidates,
        parsed=None,
        on_progress=on_progress,
        quality=result.quality,
    )
    job = get_job(job_id)
    if job:
        logger.info(
            "gemini_direct Stage 3 validation validated=%s rejected=%s candidates=%s "
            "quality_suspicious=%s repaired=%s",
            job.validated_count,
            job.rejected_count,
            job.candidate_count,
            result.quality.suspicious,
            result.quality.repaired,
        )


def _skip_parse_for_direct(job_id: str) -> None:
    _set_stage(job_id, STAGE_PARSE, StageStatus.skipped)
    _emit_progress(
        job_id,
        STAGE_PARSE,
        "skipped",
        "Skipping document parse — Gemini direct extraction from original file",
        percent=1,
        detail={"parse_channel": ParseChannel.gemini_direct.value},
    )


def _parse_document_for_channel(
    source: Path,
    filename: str,
    parse_channel: ParseChannel,
    on_progress: ProgressCallback,
) -> ParsedDocument:
    if parse_channel == ParseChannel.gemini:
        return parse_document_gemini(source, source_filename=filename, on_progress=on_progress)
    if parse_channel == ParseChannel.gemini_direct:
        raise RuntimeError("gemini_direct does not produce a ParsedDocument")
    if parse_channel == ParseChannel.unlimited_ocr:
        return parse_document_unlimited_ocr(source, source_filename=filename, on_progress=on_progress)
    return parse_document(source, source_filename=filename, on_progress=on_progress)


def run_pipeline(job_id: str) -> None:
    """Execute Stage1 → Stage2 → Stage3. Safe to run in a background thread."""
    job = get_job(job_id)
    if job is None:
        logger.error("Unknown job %s", job_id)
        return

    source = Path(job.meta["source_path"])
    parse_channel = _resolve_parse_channel(job.meta.get("parse_channel"))
    out = _job_dir(job_id)
    on_progress = _make_progress_cb(job_id)
    stage_order = [STAGE_PARSE, STAGE_EXTRACT, STAGE_VALIDATE]

    try:
        with trace_span(
            "pipeline.run",
            run_type="chain",
            inputs={"job_id": job_id, "parse_channel": parse_channel.value, "source_kind": "upload"},
            metadata={"job_id": job_id, "filename": job.filename},
            tags=["pipeline", "upload"],
        ):
            if parse_channel == ParseChannel.gemini_direct:
                _skip_parse_for_direct(job_id)
                _run_direct_extract_validate(job_id, on_progress, source=source)
            else:
                _update_job(job_id, status=JobStatus.parsing)
                _set_stage(job_id, STAGE_PARSE, StageStatus.running)
                _emit_progress(
                    job_id,
                    STAGE_PARSE,
                    "start",
                    f"Starting document parsing via {parse_channel.value}…",
                    percent=1,
                )
                with trace_span(
                    "pipeline.stage.parse",
                    run_type="tool",
                    metadata={"job_id": job_id, "stage": STAGE_PARSE, "channel": parse_channel.value},
                    tags=["pipeline", "parse"],
                ):
                    parsed = _parse_document_for_channel(
                        source=source,
                        filename=job.filename,
                        parse_channel=parse_channel,
                        on_progress=on_progress,
                    )
                _emit_progress(
                    job_id,
                    STAGE_PARSE,
                    "saving_parsed",
                    "Saving parsed.json…",
                    percent=52,
                )
                _write_json(out / "parsed.json", parsed.model_dump(mode="json"))
                preview = parsed.text_for_extraction(4000)
                _set_stage(job_id, STAGE_PARSE, StageStatus.completed)
                _update_job(job_id, page_count=parsed.page_count, parsed_preview=preview)
                _emit_progress(
                    job_id,
                    STAGE_PARSE,
                    "parse_complete",
                    f"Parsing complete via {parse_channel.value} — {parsed.page_count} page(s)",
                    percent=55,
                    detail={"page_count": parsed.page_count},
                )

                _run_extract_validate(job_id, parsed, on_progress)
    except Exception as exc:
        _fail_pipeline(job_id, exc, stage_order)

    final = get_job(job_id)
    if final:
        _write_json(out / "job.json", final.model_dump(mode="json"))


def run_url_pipeline(job_id: str) -> None:
    """Crawl a single URL (hybrid), then parse → extract → validate."""
    job = get_job(job_id)
    if job is None:
        logger.error("Unknown job %s", job_id)
        return

    out = _job_dir(job_id)
    parse_channel = _resolve_parse_channel(job.meta.get("parse_channel"))
    on_progress = _make_progress_cb(job_id)
    stage_order = [STAGE_CRAWL, STAGE_PARSE, STAGE_EXTRACT, STAGE_VALIDATE]
    source_url = job.meta.get("source_url") or ""

    try:
        with trace_span(
            "pipeline.run_url",
            run_type="chain",
            inputs={"job_id": job_id, "parse_channel": parse_channel.value, "source_kind": "url"},
            metadata={"job_id": job_id, "source_url": source_url},
            tags=["pipeline", "url"],
        ):
            _update_job(job_id, status=JobStatus.parsing)
            _set_stage(job_id, STAGE_CRAWL, StageStatus.running)
            _emit_progress(job_id, STAGE_CRAWL, "start", "Starting URL crawl…", percent=1)

            try:
                with trace_span(
                    "pipeline.stage.crawl",
                    run_type="tool",
                    metadata={"job_id": job_id, "stage": STAGE_CRAWL},
                    tags=["pipeline", "crawl"],
                ):
                    fetched = fetch_url(source_url, on_progress=on_progress)
            except Exception as exc:
                if isinstance(exc, CrawlError):
                    raise
                raise CrawlError(str(exc)) from exc

            _write_json(out / "crawl.json", crawl_result_to_artifact(fetched))
            _set_stage(job_id, STAGE_CRAWL, StageStatus.completed)
            _emit_progress(
                job_id,
                STAGE_CRAWL,
                "crawl_complete",
                f"Crawl complete — {fetched.kind}",
                percent=15,
                detail={"kind": fetched.kind},
            )

            if parse_channel == ParseChannel.gemini_direct:
                _skip_parse_for_direct(job_id)
                if isinstance(fetched, BinaryFetch):
                    path = out / fetched.filename
                    path.write_bytes(fetched.data)
                    _update_job(
                        job_id,
                        filename=fetched.filename,
                        meta={
                            **job.meta,
                            "source_path": str(path),
                            "source_url": fetched.source_url,
                            "final_url": fetched.final_url,
                            "fetch_kind": "binary",
                        },
                    )
                    _run_direct_extract_validate(job_id, on_progress, source=path)
                elif isinstance(fetched, HtmlFetch):
                    _update_job(
                        job_id,
                        filename=fetched.title or job.filename,
                        meta={
                            **job.meta,
                            "source_url": fetched.source_url,
                            "final_url": fetched.final_url,
                            "fetch_kind": "html",
                            "links_internal": fetched.links_internal,
                        },
                    )
                    _run_direct_extract_validate(
                        job_id, on_progress, markdown=fetched.markdown
                    )
                else:
                    raise CrawlError(f"Unknown fetch kind: {type(fetched)}")
            else:
                _set_stage(job_id, STAGE_PARSE, StageStatus.running)
                _emit_progress(
                    job_id,
                    STAGE_PARSE,
                    "start",
                    f"Starting document parsing via {parse_channel.value}…",
                    percent=16,
                )

                with trace_span(
                    "pipeline.stage.parse",
                    run_type="tool",
                    metadata={"job_id": job_id, "stage": STAGE_PARSE, "channel": parse_channel.value},
                    tags=["pipeline", "parse"],
                ):
                    if isinstance(fetched, BinaryFetch):
                        path = out / fetched.filename
                        path.write_bytes(fetched.data)
                        _update_job(
                            job_id,
                            filename=fetched.filename,
                            meta={
                                **job.meta,
                                "source_path": str(path),
                                "source_url": fetched.source_url,
                                "final_url": fetched.final_url,
                                "fetch_kind": "binary",
                            },
                        )
                        parsed = _parse_document_for_channel(
                            source=path,
                            filename=fetched.filename,
                            parse_channel=parse_channel,
                            on_progress=on_progress,
                        )
                    elif isinstance(fetched, HtmlFetch):
                        _update_job(
                            job_id,
                            filename=fetched.title or job.filename,
                            meta={
                                **job.meta,
                                "source_url": fetched.source_url,
                                "final_url": fetched.final_url,
                                "fetch_kind": "html",
                                "links_internal": fetched.links_internal,
                            },
                        )
                        _emit_progress(
                            job_id,
                            STAGE_PARSE,
                            "markdown_to_parsed",
                            "Building ParsedDocument from crawl markdown (OCR skipped)…",
                            percent=40,
                        )
                        parsed = parsed_document_from_markdown(
                            fetched.markdown,
                            source_url=fetched.final_url or fetched.source_url,
                            title=fetched.title,
                        )
                    else:
                        raise CrawlError(f"Unknown fetch kind: {type(fetched)}")

                _emit_progress(
                    job_id,
                    STAGE_PARSE,
                    "saving_parsed",
                    "Saving parsed.json…",
                    percent=52,
                )
                _write_json(out / "parsed.json", parsed.model_dump(mode="json"))
                preview = parsed.text_for_extraction(4000)
                _set_stage(job_id, STAGE_PARSE, StageStatus.completed)
                _update_job(job_id, page_count=parsed.page_count, parsed_preview=preview)
                _emit_progress(
                    job_id,
                    STAGE_PARSE,
                    "parse_complete",
                    f"Parsing complete via {parse_channel.value} — {parsed.page_count} page(s)",
                    percent=55,
                    detail={"page_count": parsed.page_count},
                )

                _run_extract_validate(job_id, parsed, on_progress)
    except Exception as exc:
        _fail_pipeline(job_id, exc, stage_order)

    final = get_job(job_id)
    if final:
        _write_json(out / "job.json", final.model_dump(mode="json"))


def artifact_path(job_id: str, name: str) -> Path | None:
    allowed = {"parsed.json", "candidates.json", "products.json", "job.json", "crawl.json"}
    if name not in allowed:
        return None
    path = _job_dir(job_id) / name
    return path if path.exists() else None
