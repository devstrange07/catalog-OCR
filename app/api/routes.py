"""API routes."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import pipeline as pipe
from app.config import OUTPUTS_DIR
from app.file_types import GEMINI_TEXT_SUFFIXES, allowed_suffixes_for_channel
from app.models.pipeline import JobRecord, ParseChannel
from app.stages.crawl import CrawlError, normalize_url

router = APIRouter(prefix="/api")


class UrlJobRequest(BaseModel):
    url: str = Field(..., min_length=1, description="Single catalog/page URL to fetch")
    parse_channel: ParseChannel | None = Field(
        default=None,
        description="Stage-1 parser channel: paddleocr, gemini, unlimited_ocr, or deepseek_direct",
    )


def _suffix_from_url(url: str) -> str:
    path = unquote(urlparse(url).path or "")
    return Path(path).suffix.lower()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/jobs", response_model=list[JobRecord])
def list_jobs() -> list[JobRecord]:
    return pipe.list_jobs()


@router.post("/jobs", response_model=JobRecord)
async def create_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    parse_channel: ParseChannel | None = Form(default=None),
) -> JobRecord:
    filename = Path(file.filename or "upload.bin").name
    suffix = Path(filename).suffix.lower()
    channel = pipe._resolve_parse_channel(parse_channel)
    allowed = allowed_suffixes_for_channel(channel)
    if suffix not in allowed:
        hint = ""
        if suffix in GEMINI_TEXT_SUFFIXES and channel != ParseChannel.gemini:
            hint = " Text formats (csv/txt/md/html/xml/json/rtf) require parse_channel=gemini."
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{suffix}' for channel '{channel.value}'. "
                f"Allowed: {sorted(allowed)}.{hint}"
            ),
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    job = pipe.create_job_from_bytes(filename, data, parse_channel=channel)
    background_tasks.add_task(pipe.run_pipeline, job.id)
    return job


@router.post("/jobs/from-url", response_model=JobRecord)
async def create_job_from_url(
    body: UrlJobRequest,
    background_tasks: BackgroundTasks,
) -> JobRecord:
    try:
        url = normalize_url(body.url)
    except CrawlError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    channel = pipe._resolve_parse_channel(body.parse_channel)
    if channel == ParseChannel.deepseek_direct:
        raise HTTPException(
            status_code=400,
            detail="deepseek_direct is upload-only; use POST /api/jobs.",
        )
    suffix = _suffix_from_url(url)
    if suffix:
        allowed = allowed_suffixes_for_channel(channel)
        # Only gate known binary/text catalog suffixes; bare HTML page URLs have no/odd suffixes.
        from app.file_types import IMAGE_PDF_SUFFIXES

        if suffix in (IMAGE_PDF_SUFFIXES | GEMINI_TEXT_SUFFIXES) and suffix not in allowed:
            hint = ""
            if suffix in GEMINI_TEXT_SUFFIXES and channel != ParseChannel.gemini:
                hint = " Text formats require parse_channel=gemini."
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported file type '{suffix}' for channel '{channel.value}'. "
                    f"Allowed: {sorted(allowed)}.{hint}"
                ),
            )

    job = pipe.create_job_from_url(url, parse_channel=channel)
    background_tasks.add_task(pipe.run_url_pipeline, job.id)
    return job


@router.get("/jobs/{job_id}", response_model=JobRecord)
def get_job(job_id: str) -> JobRecord:
    job = pipe.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/jobs/{job_id}/artifacts/{name}")
def get_artifact(job_id: str, name: str) -> FileResponse:
    path = pipe.artifact_path(job_id, name)
    if path is None:
        job_exists = pipe.get_job(job_id) is not None or (OUTPUTS_DIR / job_id).exists()
        if not job_exists:
            raise HTTPException(status_code=404, detail="Job not found")
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path, media_type="application/json", filename=name)
