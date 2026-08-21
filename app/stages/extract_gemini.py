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

_DISCOUNT_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "nullable": True,
    "properties": {
        "amount": {"type": "NUMBER", "nullable": True},
        "percent": {"type": "NUMBER", "nullable": True},
        "description": {"type": "STRING", "nullable": True},
    },
}

_PROMOTION_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "nullable": True,
    "description": (
        "Active offer scoped to THIS item's section only "
        "(BOGO, trial, happy_hour, flash_sale, volume_rebate, etc.)"
    ),
    "properties": {
        "type": {
            "type": "STRING",
            "nullable": True,
            "description": "Offer type: BOGO, trial, happy_hour, flash_sale, other, etc.",
        },
        "label": {
            "type": "STRING",
            "nullable": True,
            "description": "Short promo label as printed",
        },
        "details": {
            "type": "STRING",
            "nullable": True,
            "description": "Extra promo detail (e.g. on 3BHK+)",
        },
        "conditions": {
            "type": "STRING",
            "nullable": True,
            "description": "Promo-specific conditions as printed",
        },
    },
}

_PRICE_TIER_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "label": {
            "type": "STRING",
            "description": "Tier label: EMI, per sqft, list, sale, monthly, annual, etc.",
        },
        "price": {
            "type": "STRING",
            "nullable": True,
            "description": "Numeric price for this tier. Leave null if non-numeric.",
        },
        "details": {"type": "STRING", "nullable": True},
        "is_current": {
            "type": "BOOLEAN",
            "nullable": True,
            "description": "False for struck-through / superseded / was prices",
        },
        "currency": {"type": "STRING", "nullable": True},
        "billing_cycle": {
            "type": "STRING",
            "nullable": True,
            "description": "monthly, annual, one_time — if this tier has its own cycle",
        },
    },
    "required": ["label"],
}

_VARIANT_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "attribute": {
            "type": "STRING",
            "description": "Dimension: Size, Plan, Color, Portion, Room Type, etc.",
        },
        "label": {
            "type": "STRING",
            "description": "Value: 500ml, Pro, Red, Half, Private, etc.",
        },
        "price": {
            "type": "STRING",
            "nullable": True,
            "description": "Numeric price. Leave null if non-numeric; use available instead.",
        },
        "currency": {"type": "STRING", "nullable": True},
        "unit": {"type": "STRING", "nullable": True},
        "pack_size": {"type": "STRING", "nullable": True},
        "sku": {"type": "STRING", "nullable": True},
        "description": {"type": "STRING", "nullable": True},
        "available": {
            "type": "STRING",
            "nullable": True,
            "description": "Non-numeric availability (free, included, on_request) when price is null",
        },
        "is_default": {
            "type": "BOOLEAN",
            "nullable": True,
            "description": "True for exactly one variant per product whose price mirrors top-level price",
        },
    },
    "required": ["attribute", "label"],
}

PRODUCTS_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "products": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {
                        "type": "STRING",
                        "description": "Item name as printed",
                    },
                    "brand": {"type": "STRING", "nullable": True},
                    "sku": {"type": "STRING", "nullable": True},
                    "product_code": {"type": "STRING", "nullable": True},
                    "item_type": {
                        "type": "STRING",
                        "nullable": True,
                        "description": (
                            "product | service | bundle | plan | event | "
                            "addon | listing | announcement | other"
                        ),
                    },
                    "category": {
                        "type": "STRING",
                        "nullable": True,
                        "description": "Document section heading",
                    },
                    "project_name": {
                        "type": "STRING",
                        "nullable": True,
                        "description": (
                            "Sub-section project / listing group "
                            "(e.g. Skyline Vistas)"
                        ),
                    },
                    "location": {
                        "type": "STRING",
                        "nullable": True,
                        "description": (
                            "Locality / city when printed "
                            "(e.g. Whitefield, Bangalore)"
                        ),
                    },
                    "price": {
                        "type": "STRING",
                        "nullable": True,
                        "description": (
                            "Current effective / selling price as printed "
                            "(numeric string in document currency)"
                        ),
                    },
                    "list_price": {
                        "type": "STRING",
                        "nullable": True,
                        "description": (
                            "Reference/original/recommended price "
                            "(MSRP, MRP, RRP, rack rate, standard price)"
                        ),
                    },
                    "currency": {
                        "type": "STRING",
                        "nullable": True,
                        "description": "ISO currency code preferred (INR, USD, EUR) or symbol",
                    },
                    "price_qualifier": {
                        "type": "STRING",
                        "nullable": True,
                        "description": "from | up_to | approx | negotiable | on_request | null=exact",
                    },
                    "billing_cycle": {
                        "type": "STRING",
                        "nullable": True,
                        "description": "one_time | monthly | annual | per_use | per_session",
                    },
                    "discount": _DISCOUNT_SCHEMA,
                    "unit": {
                        "type": "STRING",
                        "nullable": True,
                        "description": "Pricing basis: pc, kg, hour, seat, sqft, session",
                    },
                    "pack_size": {
                        "type": "STRING",
                        "nullable": True,
                        "description": "Measure of one item (1080 sqft, 500g, 60 min) — not a count",
                    },
                    "quantity": {
                        "type": "INTEGER",
                        "nullable": True,
                        "description": "Count of items (6 cones, 10 cups)",
                    },
                    "promotion": _PROMOTION_SCHEMA,
                    "validity": {
                        "type": "STRING",
                        "nullable": True,
                        "description": "Time-bound constraints as printed",
                    },
                    "includes": {
                        "type": "ARRAY",
                        "nullable": True,
                        "items": {"type": "STRING"},
                        "description": "Bundle contents / plan features / package components",
                    },
                    "description": {
                        "type": "STRING",
                        "nullable": True,
                        "description": "Actual item description only — not promo text or conditions",
                    },
                    "conditions": {
                        "type": "ARRAY",
                        "nullable": True,
                        "items": {"type": "STRING"},
                        "description": (
                            "Item-level terms and fine print scoped to THIS item's section only"
                        ),
                    },
                    "minimum_order": {
                        "type": "STRING",
                        "nullable": True,
                        "description": "MOQ / minimum spend / minimum pax as printed",
                    },
                    "tags": {
                        "type": "ARRAY",
                        "nullable": True,
                        "items": {"type": "STRING"},
                        "description": "Open labels: new, vegetarian, bestseller, RoHS, etc.",
                    },
                    "availability": {
                        "type": "STRING",
                        "nullable": True,
                        "description": (
                            "available | out_of_stock | pre_order | coming_soon | "
                            "discontinued | on_request | waitlist | limited | unknown"
                        ),
                    },
                    "page": {
                        "type": "INTEGER",
                        "nullable": True,
                        "description": (
                            "Document page number as printed (e.g. from 'Page 2 of 8'), "
                            "not parser index"
                        ),
                    },
                    "source_snippet": {
                        "type": "STRING",
                        "nullable": True,
                        "description": "Short quote from source text (max 120 chars)",
                    },
                    "confidence": {
                        "type": "NUMBER",
                        "nullable": True,
                        "description": "0-1 how clearly the source supports this item",
                    },
                    "price_tiers": {
                        "type": "ARRAY",
                        "nullable": True,
                        "description": (
                            "Different price points for the SAME item config "
                            "(EMI, per-sqft, was/now, monthly vs annual)"
                        ),
                        "items": _PRICE_TIER_SCHEMA,
                    },
                    "variants": {
                        "type": "ARRAY",
                        "nullable": True,
                        "description": (
                            "Genuinely different configurations "
                            "(size, color, plan, portion, room type)"
                        ),
                        "items": _VARIANT_SCHEMA,
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
        f"IMPORTANT: Promotions and conditions belong ONLY to items in the same "
        f"section/table where the text appears. Do not broadcast them across "
        f"different project headings. Use price_tiers for secondary prices "
        f"(EMI, per-sqft rates).\n\n"
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
