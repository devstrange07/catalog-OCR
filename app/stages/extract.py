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
from app.models.document import ParsedDocument, ParsedPage
from app.models.product import CandidateProduct, SubmitProductsArgs

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
                            "confidence": {
                                "type": ["number", "null"],
                                "minimum": 0,
                                "maximum": 1,
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
- Identify distinct products (name, optional brand/sku/price/unit/description).
- For multi-product pages, return every distinct product you can support with evidence.
- Set source_snippet to a short quote from the text (max ~120 characters); keep description brief.
- Set page when known from headers like "--- Page N ---".
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
        for block in sorted(page.blocks, key=lambda b: b.reading_order):
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


def _dedupe_candidates(candidates: list[CandidateProduct]) -> list[CandidateProduct]:
    seen: set[tuple[str, int | None, str | None]] = set()
    out: list[CandidateProduct] = []
    for c in candidates:
        key = (c.name.strip().lower(), c.page, c.sku)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


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
