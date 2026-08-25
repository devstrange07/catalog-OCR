"""FastAPI entrypoint."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import STATIC_DIR
from app.observability.langsmith import configure_langsmith

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

app = FastAPI(
    title="Catalog Product Extraction POC",
    description=(
        "3-stage pipeline: Document Parsing (PaddleOCR+Docling, Gemini Vision, or Unlimited-OCR) → "
        "Product Extraction (gemini-2.5-flash structured JSON) → Validation. "
        "OCR never emits final product records."
    ),
    version="0.1.0",
)

configure_langsmith()
app.include_router(router)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
