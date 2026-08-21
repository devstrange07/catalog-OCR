"""Stage 2 — Product Extraction via Gemini structured output.

Takes ParsedDocument text and uses Gemini's responseSchema
(structured JSON output) to extract products. Reuses chunking/dedupe logic from
extract.py and the same business-rule system prompt. Model: settings.gemini_model
(default gemini-2.5-flash).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from app.config import settings
from app.models.document import ParsedDocument
from app.models.product import CandidateProduct, SubmitProductsArgs
from app.stages.extract import (
    SYSTEM_PROMPT,
    _chunk_pages,
    _dedupe_candidates,
    _format_pages_text,
)

logger = logging.getLogger(__name__)

_GEMINI_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "- You MUST call the submit_products tool. Do not answer with free-form JSON or prose as the product list.",
    "- Return a JSON object with a single key 'products' containing your product array.",
)

PRODUCTS_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "products": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "brand": {"type": "STRING", "nullable": True},
                    "sku": {"type": "STRING", "nullable": True},
                    "product_code": {"type": "STRING", "nullable": True},
                    "price": {"type": "STRING", "nullable": True},
                    "currency": {"type": "STRING", "nullable": True},
                    "unit": {"type": "STRING", "nullable": True},
                    "pack_size": {"type": "STRING", "nullable": True},
                    "description": {"type": "STRING", "nullable": True},
                    "page": {"type": "INTEGER", "nullable": True},
                    "source_snippet": {"type": "STRING", "nullable": True},
                    "confidence": {"type": "NUMBER", "nullable": True},
                    "price_variants": {
                        "type": "ARRAY",
                        "nullable": True,
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "attribute_name": {"type": "STRING", "nullable": True},
                                "label": {"type": "STRING", "nullable": True},
                                "price": {"type": "STRING", "nullable": True},
                                "currency": {"type": "STRING", "nullable": True},
                                "unit": {"type": "STRING", "nullable": True},
                                "pack_size": {"type": "STRING", "nullable": True},
                                "sku": {"type": "STRING", "nullable": True},
                                "available": {"type": "STRING", "nullable": True},
                                "is_default": {"type": "BOOLEAN", "nullable": True},
                                "details": {"type": "STRING", "nullable": True},
                            },
                        },
                    },
                },
                "required": ["name"],
            },
        }
    },
    "required": ["products"],
}


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


def _call_gemini_structured(markdown_text: str, *, chunk_label: str) -> list[CandidateProduct]:
    if not settings.gemini_api_key.strip():
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Set it in .env — required for "
            "Stage-2 product extraction (Gemini structured output)."
        )

    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.gemini_model}:generateContent"
    )

    user_content = (
        f"Extract all products from the following parsed document markdown ({chunk_label}).\n"
        f"Return JSON with a 'products' array.\n\n"
        f"{markdown_text}"
    )

    payload: dict[str, Any] = {
        "systemInstruction": {
            "parts": [{"text": _GEMINI_SYSTEM_PROMPT}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_content}],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": PRODUCTS_RESPONSE_SCHEMA,
        },
    }

    with httpx.Client(timeout=120) as client:
        resp = client.post(endpoint, params={"key": settings.gemini_api_key}, json=payload)
    if resp.status_code >= 400:
        raise RuntimeError(
            f"Gemini extraction request failed ({resp.status_code}): {resp.text[:400]}"
        )

    body = resp.json()
    raw_text = _extract_json_text(body)

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Gemini extraction returned invalid JSON: {exc}"
        ) from exc

    parsed = SubmitProductsArgs.model_validate(data)
    return parsed.products


def extract_products_gemini(
    document: ParsedDocument, on_progress=None
) -> list[CandidateProduct]:
    """Stage 2 entrypoint: Gemini structured product extraction."""

    def _notify(step: str, message: str, percent: int | None = None, detail=None) -> None:
        if on_progress:
            on_progress("extract", step, message, percent=percent, detail=detail)

    _notify("build_prompt", "Building Gemini extraction prompt…", percent=58)

    chunks = _chunk_pages(document.pages, settings.extract_pages_per_chunk)
    chunk_total = max(len(chunks), 1)
    all_candidates: list[CandidateProduct] = []

    for i, pages in enumerate(chunks, start=1):
        pct = 62 + int((i - 1) / chunk_total * 18)
        _notify(
            "llm_call",
            f"Gemini extracting chunk {i}/{chunk_total}…",
            percent=pct,
            detail={
                "model": settings.gemini_model,
                "chunk": i,
                "chunk_total": chunk_total,
                "pages": [p.page_number for p in pages],
            },
        )
        doc_text = _format_pages_text(pages, settings.max_extract_chars)
        chunk_label = (
            f"chunk {i}/{chunk_total}, "
            f"pages {pages[0].page_number}-{pages[-1].page_number}"
            if len(pages) > 1
            else f"chunk {i}/{chunk_total}, page {pages[0].page_number}"
            if pages
            else f"chunk {i}/{chunk_total}"
        )
        try:
            chunk_candidates = _call_gemini_structured(doc_text, chunk_label=chunk_label)
        except RuntimeError as exc:
            msg = str(exc)
            if "invalid JSON" in msg:
                logger.warning(
                    "Chunk %d/%d Gemini extraction failed (%s); retrying once",
                    i, chunk_total, msg,
                )
                chunk_candidates = _call_gemini_structured(doc_text, chunk_label=chunk_label)
            else:
                raise
        all_candidates.extend(chunk_candidates)
        logger.info(
            "Stage 2 (Gemini) chunk %d/%d extracted %d candidate(s)",
            i, chunk_total, len(chunk_candidates),
        )

    _notify("parse_tool", "Merging chunk candidates…", percent=80)
    candidates = _dedupe_candidates(all_candidates)
    logger.info("Stage 2 (Gemini) extracted %d candidate products", len(candidates))
    _notify(
        "extract_done",
        f"Extracted {len(candidates)} candidate product(s) via Gemini",
        percent=82,
        detail={"candidate_count": len(candidates), "chunks": chunk_total},
    )
    return candidates
