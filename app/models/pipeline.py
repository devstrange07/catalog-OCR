"""Pipeline job / result models."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.models.document import ParsedDocument
from app.models.product import CandidateProduct, ProductRecord


class JobStatus(str, Enum):
    queued = "queued"
    parsing = "parsing"
    extracting = "extracting"
    validating = "validating"
    completed = "completed"
    failed = "failed"


class ParseChannel(str, Enum):
    paddleocr = "paddleocr"
    gemini = "gemini"
    gemini_direct = "gemini_direct"
    unlimited_ocr = "unlimited_ocr"


class StageStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    skipped = "skipped"


class StageInfo(BaseModel):
    name: str
    status: StageStatus = StageStatus.pending
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class ProgressEvent(BaseModel):
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    stage: str
    step: str
    message: str
    percent: int | None = None
    detail: dict[str, Any] | None = None


class JobRecord(BaseModel):
    id: str
    filename: str
    status: JobStatus = JobStatus.queued
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    stages: dict[str, StageInfo] = Field(default_factory=dict)
    error: str | None = None
    page_count: int | None = None
    candidate_count: int | None = None
    validated_count: int | None = None
    rejected_count: int | None = None
    products: list[ProductRecord] = Field(default_factory=list)
    parsed_preview: str | None = None
    progress: ProgressEvent | None = None
    progress_log: list[ProgressEvent] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class PipelineResult(BaseModel):
    job_id: str
    parsed: ParsedDocument | None = None
    candidates: list[CandidateProduct] = Field(default_factory=list)
    products: list[ProductRecord] = Field(default_factory=list)
