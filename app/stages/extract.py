"""Stage 2 — Product Extraction via deepseek-v4-flash enforced tool call.

OCR/parsed text is input only. Products come exclusively from the
submit_products tool arguments — never from free-form model text or OCR mapping.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from app.config import settings
from app.models.document import ParsedDocument, ParsedPage, format_page_text
from app.models.product import CandidateProduct, PriceVariant, SubmitProductsArgs

logger = logging.getLogger(__name__)

TOOL_NAME = "submit_products"

SUBMIT_PRODUCTS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Submit all products found in the catalog/pamphlet. "
            "Call this tool with a products array. Do not invent products "
            "that are not supported by the parsed text."
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
                                "description": "Product name as printed",
                            },
                            "brand": {"type": ["string", "null"]},
                            "sku": {"type": ["string", "null"]},
                            "product_code": {"type": ["string", "null"]},
                            "price": {
                                "type": ["number", "string", "null"],
                                "description": "Numeric price if present",
                            },
                            "currency": {
                                "type": ["string", "null"],
                                "description": "ISO-ish currency code e.g. USD, INR, EUR",
                            },
                            "unit": {"type": ["string", "null"]},
                            "pack_size": {"type": ["string", "null"]},
                            "description": {
                                "type": ["string", "null"],
                                "description": "Brief product description if present",
                            },
                            "page": {"type": ["integer", "null"]},
                            "source_snippet": {
                                "type": ["string", "null"],
                                "description": "Short quote from parsed blocks (<=120 chars)",
                            },
                            "region_id": {
                                "type": ["string", "null"],
                                "description": (
                                    "The [region ...] id this product's name/title line came from, "
                                    "exactly as printed in the parsed text. Every field for this "
                                    "product must come from this same region."
                                ),
                            },
                            "source_block_ids": {
                                "type": ["array", "null"],
                                "description": "Block ids (e.g. p1-b3) this product's fields were read from.",
                                "items": {"type": "string"},
                            },
                            "confidence": {
                                "type": ["number", "null"],
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "price_variants": {
                                "type": ["array", "null"],
                                "description": "Distinct variants of this product/service (price columns, attribute options, tiers, etc.)",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "attribute_name": {
                                            "type": ["string", "null"],
                                            "description": "Category the variant differs by, e.g. age group, size, portion, tier, color",
                                        },
                                        "label": {"type": ["string", "null"]},
                                        "price": {
                                            "type": ["number", "string", "null"],
                                            "description": "Numeric price for this variant. Leave null if the variant has no numeric price; use available instead.",
                                        },
                                        "currency": {"type": ["string", "null"]},
                                        "unit": {"type": ["string", "null"]},
                                        "pack_size": {"type": ["string", "null"]},
                                        "sku": {"type": ["string", "null"]},
                                        "available": {
                                            "type": ["string", "boolean", "null"],
                                            "description": "Non-numeric availability phrase for this variant (e.g. free, included, on request) when price is null",
                                        },
                                        "is_default": {
                                            "type": ["boolean", "null"],
                                            "description": "True for exactly one variant per product: the one the top-level price/unit should mirror",
                                        },
                                        "details": {
                                            "type": ["string", "null"],
                                            "description": "Short description/inclusions specific to this variant",
                                        },
                                    },
                                    "additionalProperties": False,
                                },
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

SYSTEM_PROMPT = """You extract products from catalog and pamphlet text that was already OCR'd and structured.

Rules:
- Use ONLY the provided parsed document text.
- The text is grouped into "[region <id>]" blocks. Each region is one visual
  card/heading on the page. Treat regions as hard boundaries: a product's
  name, price, pack size, SKU, MRP/"was" price, offer code, and description
  must all come from the SAME region. Never combine text from two different
  region ids into one product, even if they appear on adjacent lines of the
  rendered text.
- Set region_id to the id of the region this product's name came from, and
  source_block_ids to the block ids (e.g. p1-b3) you actually read fields
  from. Leave a field out entirely (null) rather than pulling its value from
  a different region — do not guess based on proximity in the raw text.
- Identify distinct products (name, optional brand/sku/price/unit/description).
- For multi-product pages, return every distinct product you can support with evidence.
- Do not fold a separate size/attribute label (e.g. a short "Large"/"Small"
  block next to the title) into the product name — put it in pack_size or a
  price_variants.attribute_name instead, and keep the name as just the title
  block's own text.
- Set source_snippet to a short quote from the text (max ~120 characters); keep description brief.
- Set page when known from headers like "--- Page N ---".
- Populate price_variants whenever a single product/service has more than one
  distinct option. Classify each case as follows:
  - Tabular multi-column price layout: one row, several price-bearing columns
    under distinct headers. Emit one variant per column, in left-to-right
    order, with attribute_name set to the column-header category and label
    set to the header text.
  - Enumerated attribute list with a shared price: options differ only by a
    named attribute (size, color, pack) with no price difference. Set
    attribute_name to the attribute category and price to the shared price
    (or null if not restated per option).
  - Tiered or plan-based offering described without a table: distinct named
    tiers/plans, each with its own price or price-equivalent phrase. Treat
    identically to the tabular case, with attribute_name set to a tier/plan
    designation.
  - Non-numeric price phrase for one option (e.g. a free or on-request
    option): set that variant's price to null and put the literal phrase in
    available. Never write a non-numeric phrase into price.
- Mark exactly one variant per product with is_default=true — the option the
  top-level price/unit fields should mirror. Keep the top-level price equal
  to that default variant's price for backward compatibility.
- If a variant has its own description/inclusions text, capture it in
  price_variants.details and keep it brief (<=120 chars).
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
    text = "\n".join(format_page_text(page) for page in pages)
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[truncated]"
    return text


def _chunk_pages(pages: list[ParsedPage], pages_per_chunk: int) -> list[list[ParsedPage]]:
    if pages_per_chunk < 1:
        raise ValueError("pages_per_chunk must be >= 1")
    if not pages:
        return [[]]
    return [pages[i : i + pages_per_chunk] for i in range(0, len(pages), pages_per_chunk)]


def _variant_match_key(variant: PriceVariant) -> str:
    return (variant.attribute_name or variant.label or "").strip().lower()


def _merge_variants(
    base: list[PriceVariant] | None, incoming: list[PriceVariant] | None
) -> list[PriceVariant] | None:
    """Unions two variant lists, matching on (attribute_name or label). On a
    match, keeps whichever side already has a non-null price; unmatched
    incoming variants are appended."""
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


def _dedupe_candidates(candidates: list[CandidateProduct]) -> list[CandidateProduct]:
    """Groups candidates by (name, page, sku); duplicate-keyed candidates are
    merged into the first occurrence instead of dropped, unioning their
    price_variants so a variant table split across chunks isn't lost."""
    order: list[tuple[str, int | None, str | None]] = []
    groups: dict[tuple[str, int | None, str | None], CandidateProduct] = {}
    for c in candidates:
        key = (c.name.strip().lower(), c.page, c.sku)
        if key not in groups:
            groups[key] = c
            order.append(key)
        else:
            base = groups[key]
            base.price_variants = _merge_variants(base.price_variants, c.price_variants)
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
