"""Stage 1 — Document Parsing via Gemini Vision.

Sends the original PDF/image to Gemini (inline or Files API) and emits
ParsedDocument only (no product fields).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from app.config import settings
from app.models.document import BBox, ParsedDocument, ParsedPage, TextBlock
from app.stages.gemini_transport import (
    GENERATE_TIMEOUT as _GENERATE_TIMEOUT,
    build_file_part,
    delete_gemini_file as _delete_gemini_file,
    extract_json_text as _extract_json_text,
    generate_content_url,
    mime_for_path as _mime_for_path,
)
from app.stages.spatial import group_into_rows

logger = logging.getLogger(__name__)

_PROMPT = """You are parsing a catalog document (PDF, image, or text file such as CSV/HTML).
Extract all visible text in reading order across every page.
Return JSON only with this shape:
{
  "pages": [
    {
      "page_number": 1,
      "blocks": [
        {"text": "string", "bbox": [x0, y0, x1, y1]}
      ]
    }
  ]
}
Rules:
- Include every page that has text. Images and text files count as a single page (page_number 1) unless clearly multi-page.
- Keep every block concise but complete.
- bbox values are approximate page pixel coordinates when known; omit bbox if unsure (especially for plain text/CSV).
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


def _bbox_from_list(raw: Any) -> BBox | None:
    if not isinstance(raw, list) or len(raw) != 4:
        return None
    try:
        vals = np.asarray(raw, dtype=float).reshape(4)
    except Exception:
        return None
    return BBox(x0=float(vals[0]), y0=float(vals[1]), x1=float(vals[2]), y1=float(vals[3]))


def _page_from_raw(raw_page: dict[str, Any], page_number: int) -> ParsedPage:
    raw_blocks = raw_page.get("blocks")
    if not isinstance(raw_blocks, list):
        raw_blocks = []

    blocks: list[TextBlock] = []
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
    blocks = group_into_rows(blocks)
    lines = [block.text for block in sorted(blocks, key=lambda b: b.reading_order)]
    return ParsedPage(
        page_number=page_number,
        width=None,
        height=None,
        blocks=blocks,
        full_text="\n".join(lines),
    )


def _pages_from_response(parsed: Any) -> list[ParsedPage]:
    if not isinstance(parsed, dict):
        raise RuntimeError("Gemini returned invalid JSON for parse pages.")

    raw_pages = parsed.get("pages")
    # Backward-compatible: single-page {blocks: [...]} without pages wrapper
    if not isinstance(raw_pages, list):
        if isinstance(parsed.get("blocks"), list):
            raw_pages = [{"page_number": 1, "blocks": parsed["blocks"]}]
        else:
            raw_pages = []

    pages: list[ParsedPage] = []
    for item in raw_pages:
        if not isinstance(item, dict):
            continue
        try:
            page_number = int(item.get("page_number") or (len(pages) + 1))
        except (TypeError, ValueError):
            page_number = len(pages) + 1
        pages.append(_page_from_raw(item, page_number))

    if not pages:
        raise RuntimeError("Gemini returned no pages for document parse.")
    pages.sort(key=lambda p: p.page_number)
    return pages


def parse_document_gemini(
    source_path: Path,
    source_filename: str | None = None,
    on_progress=None,
) -> ParsedDocument:
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(path)

    if not settings.gemini_api_key.strip():
        raise RuntimeError("GEMINI_API_KEY is missing. Set it in .env for parse_channel=gemini.")

    filename = source_filename or path.name
    _notify(on_progress, "loading_gemini", "Preparing Gemini vision parser…", percent=3)

    endpoint = generate_content_url()
    uploaded_uri: str | None = None
    transport = "inline"

    with httpx.Client(timeout=_GENERATE_TIMEOUT) as client:
        file_part, transport, uploaded_uri, _, _ = build_file_part(
            client, path, on_progress=on_progress, progress_stage="parse"
        )
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": _PROMPT},
                        file_part,
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
            },
        }

        _notify(
            on_progress,
            "gemini_document",
            f"Gemini parsing document via {transport}…",
            percent=25,
            detail={"transport": transport, "model": settings.gemini_model},
        )
        try:
            resp = client.post(endpoint, params={"key": settings.gemini_api_key}, json=payload)
        finally:
            if uploaded_uri:
                _delete_gemini_file(client, uploaded_uri)

    if resp.status_code >= 400:
        raise RuntimeError(f"Gemini request failed ({resp.status_code}): {resp.text[:300]}")

    _notify(on_progress, "map_response", "Mapping Gemini parse response…", percent=42)
    body = resp.json()
    raw_text = _extract_json_text(body)
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Gemini returned invalid JSON for parse blocks.") from exc

    pages = _pages_from_response(parsed)
    block_count = sum(len(p.blocks) for p in pages)
    _notify(
        on_progress,
        "parse_done",
        f"Parsed {len(pages)} page(s), {block_count} text block(s)",
        percent=50,
        detail={"page_count": len(pages), "block_count": block_count, "transport": transport},
    )

    return ParsedDocument(
        source_filename=filename,
        source_path=str(path.resolve()),
        page_count=len(pages),
        pages=pages,
        structure_notes=[
            f"Primary text source: Gemini Vision ({settings.gemini_model}), direct document pass.",
            "Docling layout step skipped for Gemini channel.",
            "Bounding boxes are model estimates when present (no local raster).",
            "This stage emits document structure only — no product records.",
        ],
        raw_docling_markdown=None,
        meta={
            "ocr_engine": "gemini_vision",
            "parser": "gemini",
            "gemini_model": settings.gemini_model,
            "input_mode": "direct_document",
            "transport": transport,
        },
    )
