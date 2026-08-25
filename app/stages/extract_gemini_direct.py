"""Direct Gemini product extraction — original file to CandidateProduct list.

Bypasses ParsedDocument and Stage 2 text extraction. Output is the existing
SubmitProductsArgs / CandidateProduct contract for Stage 3.

After the first generation, a deterministic completeness check may trigger one
focused repair call against the same file part. That second call is not a
ParsedDocument / Stage-2 fallback.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from app.config import settings
from app.models.product import CandidateProduct, Promotion, SubmitProductsArgs
from app.observability.langsmith import trace_span
from app.stages.extract import _dedupe_candidates, _merge_price_tiers, _merge_variants
from app.stages.extract_gemini import PRODUCTS_RESPONSE_SCHEMA
from app.stages.gemini_transport import (
    GENERATE_TIMEOUT,
    assert_generate_ok,
    build_file_part,
    delete_gemini_file,
    extract_json_text,
    generate_content_url,
)

logger = logging.getLogger(__name__)

PROMPT_VERSION = "gemini_direct_v2"

_GENERIC_TIER_LABELS = frozenset({"", "price", "tier", "amount", "value", "rate"})

DIRECT_SYSTEM_PROMPT = """# Context
You inspect the original catalog, menu, rate-card, pricing page, or pamphlet
(document, image, or text file). There is no prior OCR pass and no ParsedDocument.

# Objective
Identify every distinct product or service in the entire source (all pages) and
extract fields according to the product schema. Do not invent products or values.
Preserve source values accurately. Use null when a field is unavailable.
Distinguish separate products vs variants vs price_tiers. Process every page.

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
  Do not dump all city numbers into source_snippet alone.
- Multi-column locality tables (Studio / 1BHK by locality): one product per
  column unit type; tiers labeled with the locality/city.
- Promotions and conditions belong ONLY to items in the same section/table
  where the text appears. Never broadcast a promotion across different
  project headings.
- Table footnotes (*, †, Note:) and header-stated rates (e.g. ₹9,500/sqft)
  apply to EVERY row of that same table. Put footnotes in conditions and
  header rates as a price_tier (label "per sqft" or "rate_per_unit") on
  each row — not on a subset.
- Cover "from" summary lines are optional listings; never emit them instead of
  the detailed inventory rows they summarize.

# Style
Structured extraction only. No commentary.

# Tone
Cautious and literal — prefer omitting a field over guessing.

# Audience
A downstream validator that treats this JSON as LLM candidates, not final records.

# Field rules
- Use ONLY what is visible in the original document.
- Identify distinct items (name, optional brand/sku/price/unit/description).
- Set source_snippet to a short quote (max ~120 characters).
- Set page to the document's own printed page number when visible
  (e.g. "Page 2 of 8"). Fall back to the visual page index (1-based) if no
  printed page number exists.
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
Return a JSON object with a single key 'products' containing the product array.
If the document contains no products, return {"products": []}. That is success.
Return JSON only. Do not answer with free-form prose.
"""

REPAIR_SYSTEM_PROMPT = """# Context
A previous extraction of this document returned incomplete product rows
(SKU/name present but missing price, area, project, EMI, page, or matrix axes).

# Objective
Re-extract products from the original document. Prefer complete SKU and table
rows over marketing summaries. Fill primary price, pack_size, project_name,
location, currency, unit, EMI price_tiers, page, and source_snippet whenever
visible. Preserve city→price relationships as labeled price_tiers.

# Rules
- Use ONLY what is visible. Do not invent values.
- Promotions/conditions stay section-local.
- Table footnotes and header-stated rates apply to every row of that table.
- Do not invent OCR block IDs or bounding boxes.
- Return JSON {"products": [...]} only.
"""

_USER_FILE = (
    "Extract all products from the attached original document. "
    "Inspect every page. Return JSON with a 'products' array. "
    "Prioritize complete SKU/table rows over cover summaries. "
    "Promotions and conditions belong ONLY to items in the same section/table "
    "where the text appears. Table footnotes and header rates apply to every "
    "row of that table. Use price_tiers for secondary prices (EMI, per-sqft) "
    "and for city/locality matrix prices (label = city, billing_cycle = monthly)."
)

_USER_FILE_REPAIR = (
    "Re-extract products from the attached original document with COMPLETE row "
    "payloads. For every SKU row include project_name, pack_size/area, primary "
    "price, currency/unit, EMI as price_tier label 'EMI', page, and source_snippet. "
    "For rental matrices keep UnitType → City → monthly price as labeled tiers. "
    "Do not return hollow SKU-only shells. Return JSON with a 'products' array."
)

_USER_TEXT = (
    "Extract all products from the following crawled page markdown. "
    "Return JSON with a 'products' array. "
    "Prioritize complete inventory rows over marketing summaries. "
    "Promotions and conditions belong ONLY to items in the same section/table "
    "where the text appears. Table footnotes and header rates apply to every "
    "row of that table. Use price_tiers for secondary prices (EMI, per-sqft) "
    "and for city/locality matrix prices.\n\n"
)

_USER_TEXT_REPAIR = (
    "Re-extract products from the following crawled page markdown with COMPLETE "
    "row payloads (price, pack_size, project_name, EMI tiers, page, snippet). "
    "Do not return hollow SKU-only shells. Return JSON with a 'products' array.\n\n"
)

QUALITY_WARNING_HOLLOW = (
    "Direct extraction quality: many SKU rows lack price/area/project/EMI "
    "(schema-valid but possibly incomplete source representation)"
)
QUALITY_WARNING_NO_PROVENANCE = (
    "Direct extraction quality: page/source_snippet provenance missing on most records"
)
QUALITY_WARNING_GENERIC_TIERS = (
    "Direct extraction quality: price_tiers use generic labels; "
    "city/matrix dimensions may be lost"
)


@dataclass
class DirectQualityReport:
    total: int = 0
    sku_count: int = 0
    hollow_sku_count: int = 0
    hollow_sku_ratio: float = 0.0
    with_page: int = 0
    with_snippet: int = 0
    with_price: int = 0
    generic_tier_count: int = 0
    suspicious: bool = False
    reasons: list[str] = field(default_factory=list)
    repaired: bool = False
    repair_improved: bool = False

    def warning_messages(self) -> list[str]:
        if not self.suspicious and not self.reasons:
            return []
        messages: list[str] = []
        for reason in self.reasons:
            if reason == "hollow_skus" and QUALITY_WARNING_HOLLOW not in messages:
                messages.append(QUALITY_WARNING_HOLLOW)
            elif reason == "missing_provenance" and QUALITY_WARNING_NO_PROVENANCE not in messages:
                messages.append(QUALITY_WARNING_NO_PROVENANCE)
            elif reason == "generic_tiers" and QUALITY_WARNING_GENERIC_TIERS not in messages:
                messages.append(QUALITY_WARNING_GENERIC_TIERS)
        if self.repaired and not self.repair_improved and QUALITY_WARNING_HOLLOW not in messages:
            messages.append(QUALITY_WARNING_HOLLOW)
        return messages

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DirectExtractResult:
    candidates: list[CandidateProduct]
    quality: DirectQualityReport
    call_count: int
    prompt_version: str = PROMPT_VERSION
    transport: str | None = None
    latency_ms: int | None = None


def _notify(on_progress, step: str, message: str, percent: int | None = None, detail=None) -> None:
    if on_progress:
        on_progress("extract", step, message, percent=percent, detail=detail)


def _parse_products_payload(body: dict[str, Any]) -> list[CandidateProduct]:
    assert_generate_ok(body)
    raw_text = extract_json_text(body)
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Gemini extraction returned invalid JSON: {exc}") from exc
    try:
        parsed = SubmitProductsArgs.model_validate(data)
    except ValidationError as exc:
        raise RuntimeError(f"Gemini extraction failed schema validation: {exc}") from exc
    return parsed.products


def _generation_config() -> dict[str, Any]:
    return {
        "temperature": 0,
        "responseMimeType": "application/json",
        "responseSchema": PRODUCTS_RESPONSE_SCHEMA,
        "maxOutputTokens": settings.extract_max_tokens,
        "thinkingConfig": {"thinkingBudget": -1},
    }


def _post_generate(
    client: httpx.Client,
    parts: list[dict[str, Any]],
    *,
    system_prompt: str = DIRECT_SYSTEM_PROMPT,
    call_kind: str = "extract",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": _generation_config(),
    }
    with trace_span(
        "gemini_direct.generate",
        run_type="llm",
        metadata={
            "model": settings.gemini_model,
            "max_output_tokens": settings.extract_max_tokens,
            "call_kind": call_kind,
            "prompt_version": PROMPT_VERSION,
        },
        tags=["gemini_direct", "generate", call_kind],
    ):
        resp = client.post(
            generate_content_url(),
            params={"key": settings.gemini_api_key},
            json=payload,
        )
    if resp.status_code >= 400:
        raise RuntimeError(
            f"Gemini direct extraction request failed ({resp.status_code}): {resp.text[:400]}"
        )
    return resp.json()


def _call_with_invalid_json_retry(
    client: httpx.Client,
    parts: list[dict[str, Any]],
    *,
    system_prompt: str = DIRECT_SYSTEM_PROMPT,
    call_kind: str = "extract",
) -> tuple[list[CandidateProduct], int]:
    call_count = 1
    try:
        with trace_span(
            "gemini_direct.attempt",
            run_type="tool",
            metadata={"attempt": call_count, "call_kind": call_kind},
            tags=["gemini_direct", "attempt", call_kind],
        ):
            body = _post_generate(
                client, parts, system_prompt=system_prompt, call_kind=call_kind
            )
            return _parse_products_payload(body), call_count
    except RuntimeError as exc:
        if "invalid JSON" not in str(exc):
            raise
        logger.warning(
            "Gemini direct extraction returned invalid JSON; retrying once (call_kind=%s)",
            call_kind,
        )
        call_count = 2
        with trace_span(
            "gemini_direct.attempt",
            run_type="tool",
            metadata={"attempt": call_count, "retry_reason": "invalid_json", "call_kind": call_kind},
            tags=["gemini_direct", "attempt", "retry", call_kind],
        ):
            body = _post_generate(
                client, parts, system_prompt=system_prompt, call_kind=call_kind
            )
            return _parse_products_payload(body), call_count


def _has_material_price(candidate: CandidateProduct) -> bool:
    if candidate.price not in (None, ""):
        return True
    if candidate.list_price not in (None, ""):
        return True
    for variant in candidate.variants or []:
        if variant.price not in (None, ""):
            return True
    for tier in candidate.price_tiers or []:
        if tier.price not in (None, ""):
            return True
    return False


def _is_hollow_sku(candidate: CandidateProduct) -> bool:
    if not (candidate.sku or "").strip():
        return False
    if _has_material_price(candidate):
        return False
    if (candidate.pack_size or "").strip():
        return False
    if (candidate.project_name or "").strip():
        return False
    return True


def _generic_tier_count(candidates: list[CandidateProduct]) -> int:
    count = 0
    for candidate in candidates:
        for tier in candidate.price_tiers or []:
            label = (tier.label or "").strip().lower()
            if label in _GENERIC_TIER_LABELS:
                count += 1
    return count


def assess_direct_quality(candidates: list[CandidateProduct]) -> DirectQualityReport:
    """Deterministic completeness check — does not mutate candidates."""
    total = len(candidates)
    sku_items = [c for c in candidates if (c.sku or "").strip()]
    sku_count = len(sku_items)
    hollow_sku_count = sum(1 for c in sku_items if _is_hollow_sku(c))
    hollow_sku_ratio = (hollow_sku_count / sku_count) if sku_count else 0.0
    with_page = sum(1 for c in candidates if c.page is not None)
    with_snippet = sum(1 for c in candidates if (c.source_snippet or "").strip())
    with_price = sum(1 for c in candidates if _has_material_price(c))
    generic_tier_count = _generic_tier_count(candidates)

    reasons: list[str] = []
    if sku_count >= 2 and hollow_sku_ratio >= 0.5:
        reasons.append("hollow_skus")
    if (
        total >= 5
        and sku_count >= 2
        and with_page == 0
        and with_snippet / max(total, 1) < 0.25
        and hollow_sku_ratio >= 0.3
    ):
        reasons.append("missing_provenance")
    if (
        generic_tier_count >= 3
        and generic_tier_count >= max(1, sum(len(c.price_tiers or []) for c in candidates) // 2)
        and sku_count == 0
    ):
        # Cover-style generic tiers without SKU inventory — soft signal only when
        # combined with other issues; alone not enough for repair.
        pass
    if (
        generic_tier_count >= 5
        and with_price > 0
        and sku_count >= 2
        and hollow_sku_ratio >= 0.5
    ):
        reasons.append("generic_tiers")

    return DirectQualityReport(
        total=total,
        sku_count=sku_count,
        hollow_sku_count=hollow_sku_count,
        hollow_sku_ratio=round(hollow_sku_ratio, 4),
        with_page=with_page,
        with_snippet=with_snippet,
        with_price=with_price,
        generic_tier_count=generic_tier_count,
        suspicious=bool(reasons),
        reasons=reasons,
    )


def _prefer_text(base: str | None, incoming: str | None) -> str | None:
    if base is not None and str(base).strip():
        return base
    if incoming is not None and str(incoming).strip():
        return incoming
    return base if base is not None else incoming


def _prefer_value(base: Any, incoming: Any) -> Any:
    if base not in (None, ""):
        return base
    if incoming not in (None, ""):
        return incoming
    return base


def _merge_promotion(base: Promotion | None, incoming: Promotion | None) -> Promotion | None:
    """Never adopt a repair promotion onto a different product shell.

    Promotions are section-scoped and easy to mis-attach. Keep the base value
    only; do not fill null promotions from the repair pass.
    """
    return base


def merge_direct_candidates(
    base: list[CandidateProduct],
    incoming: list[CandidateProduct],
) -> list[CandidateProduct]:
    """Non-destructive merge: fill nulls from incoming; never overwrite populated fields."""
    if not incoming:
        return list(base)
    if not base:
        return list(incoming)

    merged = [c.model_copy(deep=True) for c in base]
    index_by_sku: dict[str, int] = {}
    index_by_name_page: dict[tuple[str, int | None], int] = {}
    for i, candidate in enumerate(merged):
        sku = (candidate.sku or "").strip().lower()
        if sku:
            index_by_sku[sku] = i
        index_by_name_page[(candidate.name.strip().lower(), candidate.page)] = i

    extras: list[CandidateProduct] = []
    for item in incoming:
        sku = (item.sku or "").strip().lower()
        idx: int | None = None
        if sku and sku in index_by_sku:
            idx = index_by_sku[sku]
        else:
            key = (item.name.strip().lower(), item.page)
            idx = index_by_name_page.get(key)

        if idx is None:
            # Only append SKU-bearing or priced extras — avoid flooding with
            # duplicate cover summaries from the repair pass.
            if (item.sku or "").strip() or _has_material_price(item):
                extras.append(item.model_copy(deep=True))
            continue

        target = merged[idx]
        target.brand = _prefer_text(target.brand, item.brand)
        target.sku = _prefer_text(target.sku, item.sku)
        target.product_code = _prefer_text(target.product_code, item.product_code)
        target.item_type = _prefer_value(target.item_type, item.item_type)
        target.category = _prefer_text(target.category, item.category)
        target.project_name = _prefer_text(target.project_name, item.project_name)
        target.location = _prefer_text(target.location, item.location)
        target.price = _prefer_value(target.price, item.price)
        target.list_price = _prefer_value(target.list_price, item.list_price)
        target.currency = _prefer_text(target.currency, item.currency)
        target.price_qualifier = _prefer_text(target.price_qualifier, item.price_qualifier)
        target.billing_cycle = _prefer_text(target.billing_cycle, item.billing_cycle)
        target.discount = target.discount if target.discount is not None else item.discount
        target.unit = _prefer_text(target.unit, item.unit)
        target.pack_size = _prefer_text(target.pack_size, item.pack_size)
        target.quantity = _prefer_value(target.quantity, item.quantity)
        target.promotion = _merge_promotion(target.promotion, item.promotion)
        target.validity = _prefer_text(target.validity, item.validity)
        target.includes = target.includes if target.includes else item.includes
        target.description = _prefer_text(target.description, item.description)
        target.conditions = target.conditions if target.conditions else item.conditions
        target.minimum_order = _prefer_text(target.minimum_order, item.minimum_order)
        target.tags = target.tags if target.tags else item.tags
        target.availability = _prefer_value(target.availability, item.availability)
        target.page = _prefer_value(target.page, item.page)
        target.source_snippet = _prefer_text(target.source_snippet, item.source_snippet)
        target.confidence = _prefer_value(target.confidence, item.confidence)
        target.variants = _merge_variants(target.variants, item.variants)
        target.price_tiers = _merge_price_tiers(target.price_tiers, item.price_tiers)

    return _dedupe_candidates(merged + extras)


def apply_direct_quality_warnings(
    products: list[Any],
    quality: DirectQualityReport,
) -> None:
    """Append direct-only quality warnings onto ProductRecord.warnings in place."""
    messages = quality.warning_messages()
    if not messages:
        return
    for product in products:
        warnings = getattr(product, "warnings", None)
        if warnings is None:
            continue
        for message in messages:
            if message not in warnings:
                warnings.append(message)


def _run_extract_with_quality_gate(
    client: httpx.Client,
    parts: list[dict[str, Any]],
    *,
    repair_parts: list[dict[str, Any]],
    on_progress=None,
) -> tuple[list[CandidateProduct], DirectQualityReport, int]:
    candidates, call_count = _call_with_invalid_json_retry(client, parts)
    candidates = _dedupe_candidates(candidates)
    quality = assess_direct_quality(candidates)

    if not quality.suspicious:
        return candidates, quality, call_count

    logger.info(
        "gemini_direct quality suspicious reasons=%s hollow_sku_ratio=%s; running one repair call",
        quality.reasons,
        quality.hollow_sku_ratio,
    )
    _notify(
        on_progress,
        "quality_repair",
        "Direct extraction looks incomplete — running one focused repair call…",
        percent=72,
        detail=quality.as_dict(),
    )

    with trace_span(
        "gemini_direct.repair",
        run_type="tool",
        metadata={"reasons": list(quality.reasons), "prompt_version": PROMPT_VERSION},
        tags=["gemini_direct", "repair"],
    ):
        repair_candidates, repair_calls = _call_with_invalid_json_retry(
            client,
            repair_parts,
            system_prompt=REPAIR_SYSTEM_PROMPT,
            call_kind="repair",
        )
    call_count += repair_calls
    quality.repaired = True

    before = assess_direct_quality(candidates)
    merged = merge_direct_candidates(candidates, _dedupe_candidates(repair_candidates))
    after = assess_direct_quality(merged)
    after.repaired = True
    after.repair_improved = after.hollow_sku_count < before.hollow_sku_count or (
        after.with_price > before.with_price
    )
    # Keep original suspicion reasons if still incomplete after repair.
    if after.suspicious or before.suspicious:
        reasons = list(dict.fromkeys(before.reasons + after.reasons))
        after.reasons = reasons or before.reasons
        after.suspicious = True
    return merged, after, call_count


def extract_products_gemini_direct(
    source_path: Path,
    *,
    on_progress=None,
) -> DirectExtractResult:
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if not settings.gemini_api_key.strip():
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Set it in .env for parse_channel=gemini_direct."
        )

    _notify(on_progress, "start", "Starting Gemini direct product extraction…", percent=56)

    uploaded_uri: str | None = None
    started = time.monotonic()
    with httpx.Client(timeout=GENERATE_TIMEOUT) as client:
        file_part, transport, uploaded_uri, mime, size = build_file_part(
            client, path, on_progress=on_progress, progress_stage="extract"
        )
        _notify(
            on_progress,
            "llm_call",
            f"Gemini direct extracting via {transport}…",
            percent=62,
            detail={
                "model": settings.gemini_model,
                "prompt_version": PROMPT_VERSION,
                "transport": transport,
                "mime": mime,
                "bytes": size,
            },
        )
        parts = [{"text": _USER_FILE}, file_part]
        repair_parts = [{"text": _USER_FILE_REPAIR}, file_part]
        try:
            candidates, quality, call_count = _run_extract_with_quality_gate(
                client, parts, repair_parts=repair_parts, on_progress=on_progress
            )
        finally:
            if uploaded_uri:
                delete_gemini_file(client, uploaded_uri)

    elapsed_ms = int((time.monotonic() - started) * 1000)
    logger.info(
        "gemini_direct extract complete input_type=file mime=%s bytes=%s transport=%s "
        "model=%s prompt_version=%s call_count=%s latency_ms=%s products=%s "
        "schema_ok=true quality_suspicious=%s hollow_sku_ratio=%s repaired=%s",
        mime,
        size,
        transport,
        settings.gemini_model,
        PROMPT_VERSION,
        call_count,
        elapsed_ms,
        len(candidates),
        quality.suspicious,
        quality.hollow_sku_ratio,
        quality.repaired,
    )
    _notify(
        on_progress,
        "extract_done",
        f"Extracted {len(candidates)} candidate product(s) via Gemini direct",
        percent=82,
        detail={
            "candidate_count": len(candidates),
            "call_count": call_count,
            "transport": transport,
            "prompt_version": PROMPT_VERSION,
            "latency_ms": elapsed_ms,
            "quality": quality.as_dict(),
        },
    )
    return DirectExtractResult(
        candidates=candidates,
        quality=quality,
        call_count=call_count,
        prompt_version=PROMPT_VERSION,
        transport=transport,
        latency_ms=elapsed_ms,
    )


def extract_products_gemini_direct_from_text(
    markdown: str,
    *,
    on_progress=None,
) -> DirectExtractResult:
    if not settings.gemini_api_key.strip():
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Set it in .env for parse_channel=gemini_direct."
        )

    _notify(on_progress, "start", "Starting Gemini direct extraction from crawl markdown…", percent=56)
    started = time.monotonic()
    parts = [{"text": _USER_TEXT + markdown}]
    repair_parts = [{"text": _USER_TEXT_REPAIR + markdown}]
    with httpx.Client(timeout=GENERATE_TIMEOUT) as client:
        _notify(
            on_progress,
            "llm_call",
            "Gemini direct extracting from crawled markdown…",
            percent=62,
            detail={
                "model": settings.gemini_model,
                "prompt_version": PROMPT_VERSION,
                "transport": "text",
            },
        )
        candidates, quality, call_count = _run_extract_with_quality_gate(
            client, parts, repair_parts=repair_parts, on_progress=on_progress
        )

    elapsed_ms = int((time.monotonic() - started) * 1000)
    logger.info(
        "gemini_direct extract complete input_type=text mime=text/markdown bytes=%s "
        "transport=text model=%s prompt_version=%s call_count=%s latency_ms=%s products=%s "
        "schema_ok=true quality_suspicious=%s hollow_sku_ratio=%s repaired=%s",
        len(markdown.encode("utf-8")),
        settings.gemini_model,
        PROMPT_VERSION,
        call_count,
        elapsed_ms,
        len(candidates),
        quality.suspicious,
        quality.hollow_sku_ratio,
        quality.repaired,
    )
    _notify(
        on_progress,
        "extract_done",
        f"Extracted {len(candidates)} candidate product(s) via Gemini direct",
        percent=82,
        detail={
            "candidate_count": len(candidates),
            "call_count": call_count,
            "transport": "text",
            "prompt_version": PROMPT_VERSION,
            "latency_ms": elapsed_ms,
            "quality": quality.as_dict(),
        },
    )
    return DirectExtractResult(
        candidates=candidates,
        quality=quality,
        call_count=call_count,
        prompt_version=PROMPT_VERSION,
        transport="text",
        latency_ms=elapsed_ms,
    )
