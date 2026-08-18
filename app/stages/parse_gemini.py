"""Stage 1 — Document Parsing via Gemini Vision.

Uses Gemini vision to extract text blocks from PDF/image pages and emits
ParsedDocument only (no product fields).
"""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from PIL import Image

from app.config import settings
from app.models.document import BBox, ParsedDocument, ParsedPage, TextBlock

logger = logging.getLogger(__name__)

_PROMPT = """You are parsing one catalog page image.
Extract all visible text in reading order.
Return JSON only with this shape:
{
  "blocks": [
    {"text": "string", "bbox": [x0, y0, x1, y1]}
  ]
}
Rules:
- Keep every block concise but complete.
- bbox values are pixel coordinates for this image.
- If a bbox is unknown, omit bbox.
- Do not include commentary or markdown fences.
"""


def _notify(
    on_progress,
    step: str,
    message: str,
    percent: int | None = None,
    detail: dict | None = None,
) -> None:
    if on_progress:
        on_progress("parse", step, message, percent=percent, detail=detail)


def _rasterize(path: Path) -> list[Image.Image]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(path))
        pages: list[Image.Image] = []
        try:
            for i in range(len(pdf)):
                page = pdf[i]
                bitmap = page.render(scale=2.0)
                pages.append(bitmap.to_pil().convert("RGB"))
        finally:
            pdf.close()
        return pages

    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
        return [Image.open(path).convert("RGB")]

    raise ValueError(f"Unsupported file type: {suffix}")


def _image_to_base64(image: Image.Image) -> str:
    buff = io.BytesIO()
    image.save(buff, format="JPEG", quality=92)
    return base64.b64encode(buff.getvalue()).decode("ascii")


def _extract_json_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini returned no candidates.")
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
    text = "\n".join(t for t in texts if t).strip()
    if not text:
        raise RuntimeError("Gemini returned empty content.")
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    return text


def _bbox_from_list(raw: Any) -> BBox | None:
    if not isinstance(raw, list) or len(raw) != 4:
        return None
    try:
        vals = np.asarray(raw, dtype=float).reshape(4)
    except Exception:
        return None
    return BBox(x0=float(vals[0]), y0=float(vals[1]), x1=float(vals[2]), y1=float(vals[3]))


def _gemini_extract_blocks(image: Image.Image, page_number: int) -> ParsedPage:
    if not settings.gemini_api_key.strip():
        raise RuntimeError("GEMINI_API_KEY is missing. Set it in .env for parse_channel=gemini.")

    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.gemini_model}:generateContent"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": _PROMPT},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": _image_to_base64(image),
                        }
                    },
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }

    with httpx.Client(timeout=60) as client:
        resp = client.post(endpoint, params={"key": settings.gemini_api_key}, json=payload)
    if resp.status_code >= 400:
        raise RuntimeError(f"Gemini request failed ({resp.status_code}): {resp.text[:300]}")

    body = resp.json()
    raw_text = _extract_json_text(body)
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Gemini returned invalid JSON for parse blocks.") from exc

    raw_blocks = parsed.get("blocks") if isinstance(parsed, dict) else None
    if not isinstance(raw_blocks, list):
        raw_blocks = []

    blocks: list[TextBlock] = []
    lines: list[str] = []
    for idx, item in enumerate(raw_blocks):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        block = TextBlock(
            block_id=f"p{page_number}-b{idx}",
            text=text,
            bbox=_bbox_from_list(item.get("bbox")),
            confidence=None,
            reading_order=idx,
        )
        blocks.append(block)
        lines.append(text)

    return ParsedPage(
        page_number=page_number,
        width=image.width,
        height=image.height,
        blocks=blocks,
        full_text="\n".join(lines),
    )


def parse_document_gemini(
    source_path: Path,
    source_filename: str | None = None,
    on_progress=None,
) -> ParsedDocument:
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(path)

    filename = source_filename or path.name
    _notify(on_progress, "loading_gemini", "Preparing Gemini vision parser…", percent=3)

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        _notify(on_progress, "rasterize", "Rasterizing PDF pages…", percent=8)
    else:
        _notify(on_progress, "rasterize", "Loading image…", percent=8)

    raster_pages = _rasterize(path)
    total = len(raster_pages)
    _notify(
        on_progress,
        "rasterize",
        f"Loaded {total} page image(s)",
        percent=12,
        detail={"page_count": total},
    )

    pages: list[ParsedPage] = []
    for i, img in enumerate(raster_pages):
        page_num = i + 1
        pct = 12 + int((page_num / max(total, 1)) * 33)
        _notify(
            on_progress,
            "gemini_page",
            f"Gemini parse page {page_num}/{total}",
            percent=pct,
            detail={"page": page_num, "total_pages": total},
        )
        pages.append(_gemini_extract_blocks(img, page_num))

    block_count = sum(len(p.blocks) for p in pages)
    _notify(
        on_progress,
        "parse_done",
        f"Parsed {len(pages)} page(s), {block_count} text block(s)",
        percent=50,
        detail={"page_count": len(pages), "block_count": block_count},
    )

    return ParsedDocument(
        source_filename=filename,
        source_path=str(path.resolve()),
        page_count=len(pages),
        pages=pages,
        structure_notes=[
            f"Primary text source: Gemini Vision ({settings.gemini_model}).",
            "Docling layout step skipped for Gemini channel.",
            "This stage emits document structure only — no product records.",
        ],
        raw_docling_markdown=None,
        meta={
            "ocr_engine": "gemini_vision",
            "parser": "gemini",
            "gemini_model": settings.gemini_model,
        },
    )
