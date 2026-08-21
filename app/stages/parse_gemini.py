"""Stage 1 — Document Parsing via Gemini Vision.

Sends the original PDF/image to Gemini (inline or Files API) and emits
ParsedDocument only (no product fields).
"""

from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from app.config import settings
from app.file_types import mime_for_suffix
from app.models.document import BBox, ParsedDocument, ParsedPage, TextBlock
from app.stages.spatial import group_into_rows

logger = logging.getLogger(__name__)

_INLINE_MAX_BYTES = 15 * 1024 * 1024
_GENERATE_TIMEOUT = 180.0
_FILES_POLL_TIMEOUT = 120.0
_FILES_POLL_INTERVAL = 1.5

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


def _mime_for_path(path: Path) -> str:
    return mime_for_suffix(path.suffix.lower())


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


def _upload_to_gemini_files(
    client: httpx.Client,
    path: Path,
    mime: str,
    data: bytes,
) -> str:
    """Upload via Files API; return file URI. Polls until ACTIVE."""
    key = settings.gemini_api_key
    # Resumable / simple upload: multipart metadata + file
    meta = {"file": {"display_name": path.name}}
    resp = client.post(
        "https://generativelanguage.googleapis.com/upload/v1beta/files",
        params={"key": key},
        headers={"X-Goog-Upload-Protocol": "multipart"},
        files={
            "metadata": ("metadata.json", json.dumps(meta), "application/json"),
            "file": (path.name, data, mime),
        },
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Gemini Files upload failed ({resp.status_code}): {resp.text[:300]}")

    body = resp.json()
    file_obj = body.get("file") if isinstance(body.get("file"), dict) else body
    name = file_obj.get("name")
    uri = file_obj.get("uri")
    state = (file_obj.get("state") or "").upper()
    if not name:
        raise RuntimeError("Gemini Files upload returned no file name.")

    deadline = time.monotonic() + _FILES_POLL_TIMEOUT
    while state in {"", "PROCESSING"}:
        if time.monotonic() > deadline:
            raise RuntimeError(f"Gemini Files upload timed out waiting for ACTIVE (state={state!r}).")
        time.sleep(_FILES_POLL_INTERVAL)
        poll = client.get(
            f"https://generativelanguage.googleapis.com/v1beta/{name}",
            params={"key": key},
        )
        if poll.status_code >= 400:
            raise RuntimeError(f"Gemini Files poll failed ({poll.status_code}): {poll.text[:300]}")
        file_obj = poll.json()
        state = (file_obj.get("state") or "").upper()
        uri = file_obj.get("uri") or uri

    if state == "FAILED":
        raise RuntimeError("Gemini Files upload entered FAILED state.")
    if state != "ACTIVE" or not uri:
        raise RuntimeError(f"Gemini Files upload not usable (state={state!r}, uri={uri!r}).")
    return uri


def _delete_gemini_file(client: httpx.Client, file_uri: str) -> None:
    """Best-effort delete of an uploaded Files API object."""
    # URI looks like https://generativelanguage.googleapis.com/v1beta/files/xxx
    # or files/xxx — DELETE needs v1beta/files/xxx
    name = file_uri.rstrip("/").split("/")[-1]
    if not name:
        return
    try:
        client.delete(
            f"https://generativelanguage.googleapis.com/v1beta/files/{name}",
            params={"key": settings.gemini_api_key},
        )
    except Exception:
        logger.debug("Best-effort Gemini file delete failed for %s", name, exc_info=True)


def _build_file_part(
    client: httpx.Client,
    path: Path,
    *,
    on_progress=None,
) -> tuple[dict[str, Any], str, str | None]:
    """Return (file_part, transport, uploaded_file_uri_or_none)."""
    mime = _mime_for_path(path)
    data = path.read_bytes()
    size = len(data)

    if size <= _INLINE_MAX_BYTES:
        _notify(
            on_progress,
            "prepare_file",
            f"Sending document inline ({size} bytes)…",
            percent=12,
            detail={"transport": "inline", "bytes": size, "mime": mime},
        )
        part = {
            "inline_data": {
                "mime_type": mime,
                "data": base64.b64encode(data).decode("ascii"),
            }
        }
        return part, "inline", None

    _notify(
        on_progress,
        "upload_file",
        f"Uploading document via Files API ({size} bytes)…",
        percent=10,
        detail={"transport": "files_api", "bytes": size, "mime": mime},
    )
    uri = _upload_to_gemini_files(client, path, mime, data)
    _notify(
        on_progress,
        "upload_file",
        "Files API upload ready",
        percent=18,
        detail={"transport": "files_api", "file_uri": uri},
    )
    part = {"file_data": {"mime_type": mime, "file_uri": uri}}
    return part, "files_api", uri


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

    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.gemini_model}:generateContent"
    )
    uploaded_uri: str | None = None
    transport = "inline"

    with httpx.Client(timeout=_GENERATE_TIMEOUT) as client:
        file_part, transport, uploaded_uri = _build_file_part(client, path, on_progress=on_progress)
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
