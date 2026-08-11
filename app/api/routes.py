"""API routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app import pipeline as pipe
from app.config import OUTPUTS_DIR
from app.models.pipeline import JobRecord

router = APIRouter(prefix="/api")

ALLOWED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


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
) -> JobRecord:
    filename = Path(file.filename or "upload.bin").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_SUFFIXES)}",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    job = pipe.create_job_from_bytes(filename, data)
    background_tasks.add_task(pipe.run_pipeline, job.id)
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
