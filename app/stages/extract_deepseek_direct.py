"""Direct DeepSeek Vision product extraction — original image to CandidateProduct list.

Bypasses ParsedDocument and Stage 2 text extraction. Sends the image via the
DeepSeek Files API (raw multipart, no base64) and forces a submit_products
tool call. Output is the existing CandidateProduct contract for Stage 3.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from openai import OpenAI

from app.config import settings
from app.models.product import CandidateProduct
from app.stages.deepseek_transport import client as deepseek_client
from app.stages.deepseek_transport import delete_file, upload_image_file
from app.stages.extract import (
    SUBMIT_PRODUCTS_TOOL,
    TOOL_NAME,
    _dedupe_candidates,
    _extract_from_tool_response,
)

logger = logging.getLogger(__name__)

PROMPT_VERSION = "deepseek_direct_v1"

DIRECT_VISION_SYSTEM_PROMPT = """# Context
You inspect the original catalog, menu, rate-card, pricing page, or pamphlet
image. There is no prior OCR pass and no ParsedDocument.

# Objective
Identify every distinct product or service visible in the image and extract
fields according to the product schema. Do not invent products or values.
Preserve source values accurately. Use null when a field is unavailable.
Distinguish separate products vs variants vs price_tiers.

# Capacity priority
When output space is limited, prioritize complete SKU / inventory / table rows
(name, sku, project_name, location, pack_size/area, primary price, currency,
unit, EMI and other secondary prices as price_tiers, page, source_snippet)
over cover-page "from" marketing summaries. Prefer fewer complete rows over
many hollow ID-only rows.

# Table and matrix rules
- SKU inventory rows: one product per SKU. Keep project_name, property type
  (name), pack_size (area), primary sale price, currency/unit, and EMI as a
  price_tier with label "EMI".
- City × unit-type rental matrices: one product per row unit type; put each
  city price in price_tiers with label = city name and billing_cycle = monthly.
- Promotions and conditions belong ONLY to items in the same section/table
  where the text appears. Never broadcast a promotion across different
  project headings.
- Table footnotes (*, †, Note:) and header-stated rates (e.g. ₹9,500/sqft)
  apply to EVERY row of that same table. Put footnotes in conditions and
  header rates as a price_tier (label "per sqft" or "rate_per_unit") on
  each row — not on a subset.

# Style
Structured extraction only. No commentary.

# Tone
Cautious and literal — prefer omitting a field over guessing.

# Audience
A downstream validator that treats this JSON as LLM candidates, not final records.

# Field rules
- Use ONLY what is visible in the original image.
- Identify distinct items (name, optional brand/sku/price/unit/description).
- Set source_snippet to a short quote (max ~120 characters).
- Set page to the document's own printed page number when visible
  (e.g. "Page 2 of 8"). Fall back to 1 for a single-page image.
- Set confidence (0-1) based on how clearly the source supports the fields.
- Set item_type when clear: product, service, bundle, plan, event, addon,
  listing, announcement, other.
- Set category to the document section heading when present.
- Set project_name to the sub-section project / listing group when present;
  set location to locality/city when present.
- Pricing:
  - price = current effective / selling price.
  - list_price = reference/original/recommended price when printed separately.
  - price_qualifier = from | up_to | approx | negotiable | on_request when not exact.
  - billing_cycle = one_time | monthly | annual | per_use | per_session when applicable.
  - When a table has multiple price columns (total + EMI + per-sqft), put the
    primary sale price in price and secondary amounts in price_tiers
    (label="EMI", label="per sqft").
- Sizing: unit = pricing basis; pack_size = measure of one item (e.g. 1080 sqft);
  quantity = integer count when printed.
- price_tiers vs variants: do not conflate. price_tiers = different prices for
  the SAME configuration; variants = genuinely different configurations.
- Prefer ISO currency codes (INR, USD, EUR) when known.
- Do not invent OCR block IDs or bounding boxes.

# Response
You MUST call the submit_products tool with a products array.
If the image contains no products, call submit_products with an empty products array.
Do not answer with free-form JSON or prose as the product list.
"""

_USER_PROMPT = (
    "Extract all products from the attached original image. "
    "Return products via the submit_products tool. "
    "Prioritize complete SKU/table rows over cover summaries. "
    "Promotions and conditions belong ONLY to items in the same section/table "
    "where the text appears. Table footnotes and header rates apply to every "
    "row of that table. Use price_tiers for secondary prices (EMI, per-sqft) "
    "and for city/locality matrix prices (label = city, billing_cycle = monthly)."
)

REPAIR_USER_SUFFIX = (
    " Your previous submit_products tool arguments were not valid JSON "
    "(truncated or malformed), or you did not call the tool. "
    "Call submit_products again with valid, complete JSON only. "
    "Keep source_snippet <=120 characters and descriptions brief."
)


def _notify(on_progress, step: str, message: str, percent: int | None = None, detail=None) -> None:
    if on_progress:
        on_progress("extract", step, message, percent=percent, detail=detail)


def _call_submit_products(
    oa_client: OpenAI,
    messages: list[dict[str, Any]],
) -> list[CandidateProduct]:
    response = oa_client.chat.completions.create(
        model=settings.deepseek_vision_model,
        messages=messages,
        tools=[SUBMIT_PRODUCTS_TOOL],
        tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
        temperature=0.1,
        max_tokens=settings.extract_max_tokens,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return _extract_from_tool_response(response)


def extract_products_deepseek_direct(
    source_path: Path,
    *,
    on_progress=None,
) -> list[CandidateProduct]:
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(path)

    _notify(on_progress, "start", "Starting DeepSeek Vision direct product extraction…", percent=56)

    oa_client = deepseek_client()
    file_id: str | None = None

    try:
        _notify(
            on_progress,
            "upload_file",
            f"Uploading image via DeepSeek Files API ({path.name})…",
            percent=58,
            detail={"transport": "files_api", "filename": path.name},
        )
        file_id = upload_image_file(oa_client, path)
        _notify(
            on_progress,
            "upload_file",
            "Files API upload ready",
            percent=62,
            detail={"transport": "files_api", "file_id": file_id},
        )

        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": _USER_PROMPT},
            {"type": "file", "file_id": file_id},
        ]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": DIRECT_VISION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        _notify(
            on_progress,
            "llm_call",
            f"DeepSeek Vision extracting via {settings.deepseek_vision_model}…",
            percent=65,
            detail={
                "model": settings.deepseek_vision_model,
                "prompt_version": PROMPT_VERSION,
                "transport": "files_api",
                "file_id": file_id,
            },
        )

        try:
            candidates = _call_submit_products(oa_client, messages)
        except RuntimeError as exc:
            msg = str(exc)
            retryable = (
                "not valid JSON" in msg
                or "finish_reason=length" in msg
                or "did not call submit_products" in msg
            )
            if not retryable:
                raise
            logger.warning("DeepSeek Vision direct extract failed (%s); retrying once", msg)
            repair_content: list[dict[str, Any]] = [
                {"type": "text", "text": _USER_PROMPT + REPAIR_USER_SUFFIX},
                {"type": "file", "file_id": file_id},
            ]
            repair_messages: list[dict[str, Any]] = [
                {"role": "system", "content": DIRECT_VISION_SYSTEM_PROMPT},
                {"role": "user", "content": repair_content},
            ]
            candidates = _call_submit_products(oa_client, repair_messages)

        candidates = _dedupe_candidates(candidates)
        logger.info(
            "deepseek_direct extract complete transport=files_api model=%s "
            "prompt_version=%s products=%s",
            settings.deepseek_vision_model,
            PROMPT_VERSION,
            len(candidates),
        )
        _notify(
            on_progress,
            "extract_done",
            f"Extracted {len(candidates)} candidate product(s) via DeepSeek Vision direct",
            percent=82,
            detail={
                "candidate_count": len(candidates),
                "transport": "files_api",
                "prompt_version": PROMPT_VERSION,
                "model": settings.deepseek_vision_model,
            },
        )
        return candidates
    finally:
        if file_id:
            delete_file(oa_client, file_id)
