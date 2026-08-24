"""Optional Stage-3 fill for snippet-vs-field coverage gaps.

Runs only when settings.validate_coverage_repair is true and a record was
already flagged. Fills empty structured fields; never overwrites populated ones.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from app.config import settings
from app.models.product import ProductRecord, Promotion

logger = logging.getLogger(__name__)

_REPAIR_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "includes": {
            "type": "ARRAY",
            "nullable": True,
            "items": {"type": "STRING"},
        },
        "conditions": {
            "type": "ARRAY",
            "nullable": True,
            "items": {"type": "STRING"},
        },
        "description": {"type": "STRING", "nullable": True},
        "promotion": {
            "type": "OBJECT",
            "nullable": True,
            "properties": {
                "type": {"type": "STRING", "nullable": True},
                "label": {"type": "STRING", "nullable": True},
                "details": {"type": "STRING", "nullable": True},
                "conditions": {"type": "STRING", "nullable": True},
            },
        },
        "tags": {
            "type": "ARRAY",
            "nullable": True,
            "items": {"type": "STRING"},
        },
    },
}

_SYSTEM = (
    "Fill missing structured product fields from the source snippet. "
    "Use only the snippet and existing JSON. Do not invent values. "
    "Return JSON only. Leave a field null when it is not supported."
)


def _prefer_list(base: list[str] | None, incoming: list[str] | None) -> list[str] | None:
    if base:
        return base
    if not incoming:
        return base
    cleaned = [str(v).strip() for v in incoming if str(v).strip()]
    return cleaned or base


def _prefer_text(base: str | None, incoming: str | None) -> str | None:
    if base is not None and str(base).strip():
        return base
    if incoming is not None and str(incoming).strip():
        return incoming
    return base


def apply_coverage_repair_payload(record: ProductRecord, payload: dict[str, Any]) -> bool:
    """Merge repair JSON into empty fields. Returns True if anything changed."""
    changed = False
    includes = _prefer_list(record.includes, payload.get("includes"))
    if includes != record.includes:
        record.includes = includes
        changed = True
    conditions = _prefer_list(record.conditions, payload.get("conditions"))
    if conditions != record.conditions:
        record.conditions = conditions
        changed = True
    tags = _prefer_list(record.tags, payload.get("tags"))
    if tags != record.tags:
        record.tags = tags
        changed = True
    description = _prefer_text(record.description, payload.get("description"))
    if description != record.description:
        record.description = description
        changed = True
    if record.promotion is None and isinstance(payload.get("promotion"), dict):
        promo = Promotion.model_validate(payload["promotion"])
        if any([promo.type, promo.label, promo.details, promo.conditions]):
            record.promotion = promo
            changed = True
    if changed:
        note = "Coverage repair filled empty structured fields from source snippet"
        if note not in record.warnings:
            record.warnings.append(note)
    return changed


def _extract_json_text(body: dict[str, Any]) -> str:
    candidates = body.get("candidates") or []
    if not candidates:
        raise RuntimeError("Coverage repair returned no candidates.")
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
    text = "\n".join(t for t in texts if t).strip()
    if not text:
        raise RuntimeError("Coverage repair returned empty content.")
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    return text


def _repair_one(record: ProductRecord, client: httpx.Client) -> None:
    snippet_flags = [w for w in record.warnings if "Source snippet mentions" in w]
    user = (
        "Existing product JSON (fill only empty fields):\n"
        f"{record.model_dump_json()}\n\n"
        f"Source snippet: {record.source_snippet}\n"
        f"Coverage flags: {snippet_flags}"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": _SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": _REPAIR_SCHEMA,
        },
    }
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.gemini_model}:generateContent"
    )
    resp = client.post(
        endpoint, params={"key": settings.gemini_api_key}, json=payload
    )
    if resp.status_code >= 400:
        raise RuntimeError(
            f"Coverage repair request failed ({resp.status_code}): {resp.text[:400]}"
        )
    data = json.loads(_extract_json_text(resp.json()))
    if isinstance(data, dict):
        apply_coverage_repair_payload(record, data)


def repair_coverage_gaps(records: list[ProductRecord]) -> None:
    if not records:
        return
    if not settings.gemini_api_key.strip():
        logger.warning("Coverage repair skipped: GEMINI_API_KEY is missing")
        return
    with httpx.Client(timeout=60) as client:
        for record in records:
            try:
                _repair_one(record, client)
            except Exception:
                logger.warning(
                    "Coverage repair failed for %s", record.name, exc_info=True
                )
