"""Stage 2 — Product Extraction via deepseek-v4-flash enforced tool call.

OCR/parsed text is input only. Products come exclusively from the
submit_products tool arguments — never from free-form model text or OCR mapping.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

from openai import OpenAI

from app.config import settings
from app.models.document import ParsedDocument, ParsedPage
from app.models.product import CandidateProduct, PriceTier, SubmitProductsArgs, Variant

logger = logging.getLogger(__name__)

TOOL_NAME = "submit_products"

_DISCOUNT_SCHEMA: dict[str, Any] = {
    "type": ["object", "null"],
    "properties": {
        "amount": {"type": ["number", "null"]},
        "percent": {"type": ["number", "null"]},
        "description": {"type": ["string", "null"]},
    },
    "additionalProperties": False,
}

_PROMOTION_SCHEMA: dict[str, Any] = {
    "type": ["object", "null"],
    "properties": {
        "type": {"type": ["string", "null"]},
        "label": {"type": ["string", "null"]},
        "details": {"type": ["string", "null"]},
        "conditions": {"type": ["string", "null"]},
    },
    "additionalProperties": False,
}

_PRICE_TIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "price": {
            "type": ["number", "string", "null"],
            "description": "Numeric price for this tier. Leave null if non-numeric.",
        },
        "details": {"type": ["string", "null"]},
        "is_current": {
            "type": ["boolean", "null"],
            "description": "False for struck-through / superseded / was prices",
        },
        "currency": {"type": ["string", "null"]},
        "billing_cycle": {
            "type": ["string", "null"],
            "description": "monthly, annual, one_time — if this tier has its own cycle",
        },
    },
    "required": ["label"],
    "additionalProperties": False,
}

_VARIANT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "attribute": {
            "type": "string",
            "description": "Dimension: Size, Plan, Color, Portion, Room Type, etc.",
        },
        "label": {
            "type": "string",
            "description": "Value: 500ml, Pro, Red, Half, Private, etc.",
        },
        "price": {
            "type": ["number", "string", "null"],
            "description": "Numeric price. Leave null if non-numeric; use available instead.",
        },
        "currency": {"type": ["string", "null"]},
        "unit": {"type": ["string", "null"]},
        "pack_size": {"type": ["string", "null"]},
        "sku": {"type": ["string", "null"]},
        "description": {"type": ["string", "null"]},
        "available": {
            "type": ["string", "boolean", "null"],
            "description": "Non-numeric availability (free, included, on_request) when price is null",
        },
        "is_default": {
            "type": ["boolean", "null"],
            "description": "True for exactly one variant per product whose price mirrors top-level price",
        },
    },
    "required": ["attribute", "label"],
    "additionalProperties": False,
}

SUBMIT_PRODUCTS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Submit all products/services found in the catalog, menu, rate card, "
            "pricing page, or pamphlet. Call this tool with a products array. "
            "Do not invent items that are not supported by the parsed text."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "products": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Item name as printed",
                            },
                            "brand": {"type": ["string", "null"]},
                            "sku": {"type": ["string", "null"]},
                            "product_code": {"type": ["string", "null"]},
                            "item_type": {
                                "type": ["string", "null"],
                                "description": (
                                    "product | service | bundle | plan | event | "
                                    "addon | listing | announcement | other"
                                ),
                            },
                            "category": {
                                "type": ["string", "null"],
                                "description": "Document section heading",
                            },
                            "project_name": {
                                "type": ["string", "null"],
                                "description": (
                                    "Sub-section project / listing group "
                                    "(e.g. Skyline Vistas)"
                                ),
                            },
                            "location": {
                                "type": ["string", "null"],
                                "description": (
                                    "Locality / city when printed "
                                    "(e.g. Whitefield, Bangalore)"
                                ),
                            },
                            "price": {
                                "type": ["number", "string", "null"],
                                "description": "Current effective / selling price",
                            },
                            "list_price": {
                                "type": ["number", "string", "null"],
                                "description": (
                                    "Reference/original/recommended price "
                                    "(MSRP, MRP, RRP, rack rate, standard price)"
                                ),
                            },
                            "currency": {
                                "type": ["string", "null"],
                                "description": "ISO currency code preferred (INR, USD, EUR) or symbol",
                            },
                            "price_qualifier": {
                                "type": ["string", "null"],
                                "description": "from | up_to | approx | negotiable | on_request | null=exact",
                            },
                            "billing_cycle": {
                                "type": ["string", "null"],
                                "description": "one_time | monthly | annual | per_use | per_session",
                            },
                            "discount": _DISCOUNT_SCHEMA,
                            "unit": {
                                "type": ["string", "null"],
                                "description": "Pricing basis: pc, kg, hour, seat, sqft, session",
                            },
                            "pack_size": {
                                "type": ["string", "null"],
                                "description": "Measure of one item: 100ml, 500g, 60 min",
                            },
                            "quantity": {
                                "type": ["integer", "null"],
                                "description": "Count of items (6 cones, 10 cups)",
                            },
                            "promotion": _PROMOTION_SCHEMA,
                            "validity": {
                                "type": ["string", "null"],
                                "description": "Time-bound constraints as printed",
                            },
                            "includes": {
                                "type": ["array", "null"],
                                "items": {"type": "string"},
                                "description": "Bundle contents / plan features / package components",
                            },
                            "description": {
                                "type": ["string", "null"],
                                "description": "Actual item description only — not promo text or conditions",
                            },
                            "conditions": {
                                "type": ["array", "null"],
                                "items": {"type": "string"},
                                "description": "Terms, prerequisites, fine print",
                            },
                            "minimum_order": {
                                "type": ["string", "null"],
                                "description": "MOQ / minimum spend / minimum pax as printed",
                            },
                            "tags": {
                                "type": ["array", "null"],
                                "items": {"type": "string"},
                                "description": "Open labels: new, vegetarian, bestseller, RoHS, etc.",
                            },
                            "availability": {
                                "type": ["string", "null"],
                                "description": (
                                    "available | out_of_stock | pre_order | coming_soon | "
                                    "discontinued | on_request | waitlist | limited | unknown"
                                ),
                            },
                            "page": {"type": ["integer", "null"]},
                            "source_snippet": {
                                "type": ["string", "null"],
                                "description": "Short quote from parsed blocks (<=120 chars)",
                            },
                            "confidence": {
                                "type": ["number", "null"],
                                "minimum": 0,
                                "maximum": 1,
                                "description": "0-1 how clearly the source supports this item",
                            },
                            "price_tiers": {
                                "type": ["array", "null"],
                                "description": (
                                    "Different price points for the SAME configuration "
                                    "(list vs sale, monthly vs annual, volume bands)"
                                ),
                                "items": _PRICE_TIER_SCHEMA,
                            },
                            "variants": {
                                "type": ["array", "null"],
                                "description": (
                                    "Genuinely different configurations "
                                    "(size, color, plan, portion, room type)"
                                ),
                                "items": _VARIANT_SCHEMA,
                            },
                        },
                        "required": ["name"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["products"],
            "additionalProperties": False,
        },
    },
}

SYSTEM_PROMPT = """You extract products and services from catalog, menu, rate-card, pricing-page, and pamphlet text that was already OCR'd and structured.

Rules:
- Use ONLY the provided parsed document text.
- Identify distinct items (name, optional brand/sku/price/unit/description). Works for physical goods, services, plans, bundles, events, listings, and announcements.
- For multi-item pages, return every distinct item you can support with evidence.
- Set source_snippet to a short quote from the text (max ~120 characters); keep description brief and reserved for actual item description — not promo slogans, not conditions, not validity.
- Set page to the document's own printed page number when visible (e.g. "Page 2 of 8"). Fall back to the parser header "--- Page N ---" only if no printed page number exists.
- Set confidence (0-1) based on how clearly the source supports the extracted fields.
- Set item_type when clear: product, service, bundle, plan, event, addon, listing, announcement, other.
- Set category to the document section heading when present (e.g. "Starters", "Team Plans", "Fasteners").
- Set project_name to the sub-section project / listing group when present (e.g. "Skyline Vistas"); set location to locality/city when present (e.g. "Whitefield, Bangalore").
- Pricing:
  - price = current effective / selling price.
  - list_price = reference/original/recommended price when printed separately (MSRP, MRP, RRP, rack rate, standard price).
  - price_qualifier = from | up_to | approx | negotiable | on_request when the printed price is not exact; leave null when exact.
  - billing_cycle = one_time | monthly | annual | per_use | per_session when recurring or usage-based.
  - discount = structured {amount, percent, description} when savings are printed.
  - When a table has multiple price columns (e.g. total price + EMI + price-per-sqft), capture the primary sale price in `price` and encode secondary amounts as price_tiers entries (label="EMI", label="per sqft").
  - Section-header psf rates (e.g. "₹9,500/sqft") should appear as a price_tier on each item in that section, or be used to set `unit` = "sqft" if it is the sole pricing basis.
- Sizing:
  - unit = pricing basis (pc, kg, hour, seat, sqft, session).
  - pack_size = measure of one item (100ml, 500g, 60 min) — not a count.
  - quantity = integer count of items when printed (6, 10, 1000).
- price_tiers vs variants (do not conflate):
  - price_tiers: different prices for the SAME configuration (list vs sale, was/now, monthly vs annual, volume bands, rack vs online). Set is_current=false for struck-through / was / superseded prices.
  - variants: genuinely different configurations (size, flavor, plan, portion, room type). Set attribute to the dimension and label to the value. Mark exactly one variant per product with is_default=true when variants exist; keep top-level price equal to that default variant's price.
  - Non-numeric price phrase for one option (free, on request): set that variant's price to null and put the literal phrase in available. Never write a non-numeric phrase into price.
- promotion: active offer {type, label, details, conditions} — BOGO, trial, happy_hour, flash_sale, volume_rebate, etc.
- validity: time-bound constraints as printed (today only, lunch 12-3 PM, offer ends Dec 31).
- includes: list of bundle contents / plan features / package components.
- conditions: item-level terms and fine print (taxes, prerequisites, non-refundable).
- minimum_order: MOQ / minimum spend / minimum pax as printed.
- tags: open labels (new, vegetarian, bestseller, RoHS_compliant, most_popular).
- availability: available | out_of_stock | pre_order | coming_soon | discontinued | on_request | waitlist | limited | unknown. Use coming_soon / announcement for "price list TBA" style items.
- Scope & locality:
  - promotion, conditions, and description belong ONLY to items in the same section/box/table where the text appears. Never broadcast a promotion or condition across different project headings.
  - If a section heading names a project/location (e.g. "SKYLINE VISTAS, WHITEFIELD"), items under it share that context — but items under a DIFFERENT heading do not.
  - Page-wide legal lines (GST, stamp duty, RERA) that appear in a page header/footer may apply to all items. Section-specific terms (possession date, booking %, bank tie-ups) apply only to items in that section.
  - When in doubt, leave promotion/conditions null rather than guessing scope.
- Prefer ISO currency codes (INR, USD, EUR) when known; symbols like ₹/$/€ are acceptable.
- If no products are present, call submit_products with an empty products array.
- You MUST call the submit_products tool. Do not answer with free-form JSON or prose as the product list.
"""

REPAIR_USER_SUFFIX = (
    "\n\nYour previous submit_products tool arguments were not valid JSON "
    "(truncated or malformed). Call submit_products again with valid, complete JSON only. "
    "Keep source_snippet <=120 characters and descriptions brief."
)


def _client() -> OpenAI:
    if not settings.deepseek_api_key.strip():
        raise RuntimeError(
            "DEEPSEEK_API_KEY is missing. Set it in .env — Stage 2 cannot invent products from OCR."
        )
    return OpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
    )


def _format_pages_text(pages: list[ParsedPage], max_chars: int) -> str:
    parts: list[str] = []
    for page in pages:
        parts.append(f"--- Page {page.page_number} ---")
        ordered_blocks = sorted(page.blocks, key=lambda b: b.reading_order)
        grouped: dict[int, list] = defaultdict(list)
        no_row_blocks = []
        for block in ordered_blocks:
            if block.row_id is None:
                no_row_blocks.append(block)
            else:
                grouped[block.row_id].append(block)

        for row_id in sorted(grouped):
            row_blocks = sorted(grouped[row_id], key=lambda b: b.reading_order)
            if row_blocks:
                lead = f"[{row_blocks[0].block_id}] "
                row_text = " | ".join(b.text for b in row_blocks)
                parts.append(lead + row_text)

        for block in no_row_blocks:
            parts.append(f"[{block.block_id}] {block.text}")
        if not page.blocks and page.full_text:
            parts.append(page.full_text)
    text = "\n".join(parts)
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[truncated]"
    return text


def _chunk_pages(pages: list[ParsedPage], pages_per_chunk: int) -> list[list[ParsedPage]]:
    if pages_per_chunk < 1:
        raise ValueError("pages_per_chunk must be >= 1")
    if not pages:
        return [[]]
    return [pages[i : i + pages_per_chunk] for i in range(0, len(pages), pages_per_chunk)]


def _variant_match_key(variant: Variant) -> str:
    return f"{(variant.attribute or '').strip().lower()}|{(variant.label or '').strip().lower()}"


def _tier_match_key(tier: PriceTier) -> str:
    return (tier.label or "").strip().lower()


def _merge_variants(
    base: list[Variant] | None, incoming: list[Variant] | None
) -> list[Variant] | None:
    """Union two variant lists, matching on (attribute, label). On a match, keep
    whichever side already has a non-null price; unmatched incoming are appended."""
    if not incoming:
        return base
    if not base:
        return list(incoming)

    merged = list(base)
    index_by_key = {
        _variant_match_key(v): i for i, v in enumerate(merged) if _variant_match_key(v)
    }
    for variant in incoming:
        key = _variant_match_key(variant)
        if key and key in index_by_key:
            existing = merged[index_by_key[key]]
            if existing.price is None and variant.price is not None:
                merged[index_by_key[key]] = variant
        else:
            merged.append(variant)
            if key:
                index_by_key[key] = len(merged) - 1
    return merged


def _merge_price_tiers(
    base: list[PriceTier] | None, incoming: list[PriceTier] | None
) -> list[PriceTier] | None:
    if not incoming:
        return base
    if not base:
        return list(incoming)

    merged = list(base)
    index_by_key = {
        _tier_match_key(t): i for i, t in enumerate(merged) if _tier_match_key(t)
    }
    for tier in incoming:
        key = _tier_match_key(tier)
        if key and key in index_by_key:
            existing = merged[index_by_key[key]]
            if existing.price is None and tier.price is not None:
                merged[index_by_key[key]] = tier
        else:
            merged.append(tier)
            if key:
                index_by_key[key] = len(merged) - 1
    return merged


def _dedupe_candidates(candidates: list[CandidateProduct]) -> list[CandidateProduct]:
    """Groups candidates by (name, page, sku); duplicate-keyed candidates are
    merged into the first occurrence, unioning variants and price_tiers."""
    order: list[tuple[str, int | None, str | None]] = []
    groups: dict[tuple[str, int | None, str | None], CandidateProduct] = {}
    for c in candidates:
        key = (c.name.strip().lower(), c.page, c.sku)
        if key not in groups:
            groups[key] = c
            order.append(key)
        else:
            base = groups[key]
            base.variants = _merge_variants(base.variants, c.variants)
            base.price_tiers = _merge_price_tiers(base.price_tiers, c.price_tiers)
    return [groups[key] for key in order]


def _parse_tool_args(raw: str) -> list[CandidateProduct]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"submit_products arguments were not valid JSON: {exc}") from exc

    parsed = SubmitProductsArgs.model_validate(data)
    return parsed.products


def _extract_from_tool_response(response: Any) -> list[CandidateProduct]:
    choice = response.choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason == "length":
        raise RuntimeError(
            "submit_products response was truncated (finish_reason=length). "
            "Try fewer pages per chunk or a higher extract_max_tokens."
        )

    message = choice.message
    tool_calls = message.tool_calls or []
    if not tool_calls:
        raise RuntimeError(
            "Model did not call submit_products. Refusing to treat free-form output as products."
        )

    submit_call = next((tc for tc in tool_calls if tc.function.name == TOOL_NAME), None)
    if submit_call is None:
        names = [tc.function.name for tc in tool_calls]
        raise RuntimeError(f"Expected tool {TOOL_NAME}, got: {names}")

    return _parse_tool_args(submit_call.function.arguments)


def _call_submit_products(
    client: OpenAI,
    messages: list[dict[str, Any]],
) -> list[CandidateProduct]:
    response = client.chat.completions.create(
        model=settings.deepseek_model,
        messages=messages,
        tools=[SUBMIT_PRODUCTS_TOOL],
        tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
        temperature=0.1,
        max_tokens=settings.extract_max_tokens,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return _extract_from_tool_response(response)


def _extract_chunk(
    client: OpenAI,
    document: ParsedDocument,
    pages: list[ParsedPage],
    *,
    chunk_index: int,
    chunk_total: int,
) -> list[CandidateProduct]:
    page_nums = [p.page_number for p in pages] if pages else []
    page_label = (
        f"pages {page_nums[0]}-{page_nums[-1]}"
        if len(page_nums) > 1
        else f"page {page_nums[0]}"
        if page_nums
        else "no pages"
    )
    doc_text = _format_pages_text(pages, settings.max_extract_chars)
    user_content = (
        f"Source file: {document.source_filename}\n"
        f"Document pages: {document.page_count}\n"
        f"Extracting chunk {chunk_index}/{chunk_total} ({page_label}). "
        f"Return products found in these pages only.\n\n"
        f"Parsed document (block ids in brackets):\n{doc_text}"
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        return _call_submit_products(client, messages)
    except RuntimeError as exc:
        msg = str(exc)
        retryable = (
            "not valid JSON" in msg
            or "finish_reason=length" in msg
        )
        if not retryable:
            raise
        logger.warning(
            "Chunk %d/%d extract failed (%s); retrying once",
            chunk_index,
            chunk_total,
            msg,
        )
        repair_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content + REPAIR_USER_SUFFIX},
        ]
        return _call_submit_products(client, repair_messages)


def extract_products(document: ParsedDocument, on_progress=None) -> list[CandidateProduct]:
    """Stage 2 entrypoint: enforced tool call → CandidateProduct list."""

    def _notify(step: str, message: str, percent: int | None = None, detail=None) -> None:
        if on_progress:
            on_progress("extract", step, message, percent=percent, detail=detail)

    _notify("build_prompt", "Building extraction prompt…", percent=58)
    client = _client()

    chunks = _chunk_pages(document.pages, settings.extract_pages_per_chunk)
    chunk_total = max(len(chunks), 1)
    all_candidates: list[CandidateProduct] = []

    for i, pages in enumerate(chunks, start=1):
        # Map chunk progress across 62–80
        pct = 62 + int((i - 1) / chunk_total * 18)
        _notify(
            "llm_call",
            f"Extracting chunk {i}/{chunk_total} via {settings.deepseek_model}…",
            percent=pct,
            detail={
                "model": settings.deepseek_model,
                "chunk": i,
                "chunk_total": chunk_total,
                "pages": [p.page_number for p in pages],
            },
        )
        chunk_candidates = _extract_chunk(
            client,
            document,
            pages,
            chunk_index=i,
            chunk_total=chunk_total,
        )
        all_candidates.extend(chunk_candidates)
        logger.info(
            "Stage 2 chunk %d/%d extracted %d candidate(s)",
            i,
            chunk_total,
            len(chunk_candidates),
        )

    _notify("parse_tool", "Merging chunk candidates…", percent=80)
    candidates = _dedupe_candidates(all_candidates)
    logger.info("Stage 2 extracted %d candidate products via tool call", len(candidates))
    _notify(
        "extract_done",
        f"Extracted {len(candidates)} candidate product(s)",
        percent=82,
        detail={"candidate_count": len(candidates), "chunks": chunk_total},
    )
    return candidates
