"""Post-validation Prism export stage.

Converts validated ProductRecord rows into RawProduct / RawPlan / RawPriceList
via Gemini structured output (responseSchema), then numerically validates the
result. Standalone — not wired into the main pipeline.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from typing import Any

import httpx

from app.config import settings
from app.models.prism import (
    PrismExport,
    PrismExportResult,
    PrismIssueSeverity,
    PrismValidationIssue,
    RawPriceList,
    RawProduct,
)
from app.models.product import ProductRecord, ProductStatus

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You convert already-validated catalog product records into the
Prism (product-rag) ingestion shape: RawProduct → RawPlan → RawPriceList.

Rules:
- Emit exactly one RawProduct per source product. Do not invent products.
- Map name, category, unit, brand, description from the source.
- Set kind from item_type when present (e.g. product, service, plan, bundle).
- plans: one plan for the base product configuration, plus one plan per variant
  (variants are distinct configurations — size, color, plan tier, etc.).
- Inside each plan, price_lists capture monetary amounts:
  - Primary selling price as one entry (name like "default" or the plan name).
  - list_price (MSRP/MRP) as a separate entry when present (name "list").
  - Each price_tier as its own entry (name = tier label).
- amount = the pre-discount figure (list/rack/original when discount applies,
  otherwise the selling price). discount = absolute amount subtracted from amount
  when savings are known; leave discount null when unknown.
- Prefer ISO currency codes (INR, USD, EUR). Infer country from currency when
  obvious (INR→IN, USD→US, EUR→DE, GBP→GB); otherwise use a best-effort ISO-3166
  alpha-2 code or "XX".
- billing_period / pricing_type / interval / meter: copy from source billing_cycle
  or tier when clear; otherwise leave null.
- id and updated_at will be overwritten by the caller — you may use placeholders.
- Return JSON with a single key "products" containing the RawProduct array.
- Do not invent numeric amounts that are not supported by the source JSON.
"""

_PRICE_LIST_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "name": {"type": "STRING"},
        "country": {"type": "STRING"},
        "currency": {"type": "STRING"},
        "amount": {"type": "NUMBER"},
        "discount": {"type": "NUMBER", "nullable": True},
        "discount_strategy": {"type": "STRING", "nullable": True},
    },
    "required": ["name", "country", "currency", "amount"],
}

_PLAN_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "id": {"type": "STRING"},
        "name": {"type": "STRING"},
        "feature_text": {"type": "STRING", "nullable": True},
        "price_lists": {
            "type": "ARRAY",
            "items": _PRICE_LIST_SCHEMA,
        },
        "billing_period": {"type": "STRING", "nullable": True},
        "pricing_type": {"type": "STRING", "nullable": True},
        "interval": {"type": "STRING", "nullable": True},
        "meter": {"type": "STRING", "nullable": True},
    },
    "required": ["id", "name", "price_lists"],
}

_PRISM_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "products": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "STRING"},
                    "name": {"type": "STRING"},
                    "category": {"type": "STRING", "nullable": True},
                    "unit": {"type": "STRING", "nullable": True},
                    "kb_article": {"type": "STRING", "nullable": True},
                    "brand": {"type": "STRING", "nullable": True},
                    "kind": {"type": "STRING", "nullable": True},
                    "description": {"type": "STRING"},
                    "plans": {
                        "type": "ARRAY",
                        "items": _PLAN_SCHEMA,
                    },
                    "updated_at": {"type": "STRING"},
                },
                "required": ["id", "name", "description", "plans", "updated_at"],
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


def _source_payload(products: list[ProductRecord]) -> list[dict[str, Any]]:
    """Slim JSON for the prompt — drop validation metadata noise."""
    out: list[dict[str, Any]] = []
    for p in products:
        d = p.model_dump(mode="json", exclude={"status", "validation_errors", "warnings"})
        out.append(d)
    return out


def _call_gemini_structured(
    products_json: str, *, chunk_label: str
) -> list[RawProduct]:
    if not settings.gemini_api_key.strip():
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Set it in .env — required for "
            "Prism export (Gemini structured output)."
        )

    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.gemini_model}:generateContent"
    )

    user_content = (
        f"Convert the following validated catalog products to Prism RawProduct "
        f"shape ({chunk_label}).\n"
        f"Return JSON with a 'products' array.\n\n"
        f"{products_json}"
    )

    payload: dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_content}],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": _PRISM_RESPONSE_SCHEMA,
        },
    }

    with httpx.Client(timeout=120) as client:
        resp = client.post(
            endpoint, params={"key": settings.gemini_api_key}, json=payload
        )
    if resp.status_code >= 400:
        raise RuntimeError(
            f"Gemini prism-export request failed ({resp.status_code}): {resp.text[:400]}"
        )

    body = resp.json()
    raw_text = _extract_json_text(body)

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Gemini prism-export returned invalid JSON: {exc}"
        ) from exc

    parsed = PrismExport.model_validate(data)
    return parsed.products


def _stamp_ids_and_timestamps(
    products: list[RawProduct], job_id: str, *, now: datetime | None = None
) -> None:
    ts = (now or datetime.now(timezone.utc)).isoformat()
    for i, product in enumerate(products):
        product.id = f"{job_id}-{i:04d}"
        product.updated_at = ts
        for j, plan in enumerate(product.plans):
            plan.id = f"{product.id}-plan-{j:02d}"


def _norm_name(name: str | None) -> str:
    return (name or "").strip().lower()


def _norm_currency(value: str | None) -> str:
    return (value or "").strip().upper()


def _amounts_within_tolerance(a: float, b: float, tol: float) -> bool:
    if a <= 0 or b <= 0:
        return False
    return abs(a - b) / max(a, b) <= tol


def _numeric_validate(
    products: list[RawProduct],
    source_products: list[ProductRecord],
) -> list[PrismValidationIssue]:
    issues: list[PrismValidationIssue] = []
    tol = settings.prism_price_tolerance_pct

    source_by_name: dict[str, ProductRecord] = {}
    for src in source_products:
        key = _norm_name(src.name)
        if key and key not in source_by_name:
            source_by_name[key] = src

    if len(products) != len(source_products):
        issues.append(
            PrismValidationIssue(
                field="products",
                message=(
                    f"Exported product count ({len(products)}) differs from "
                    f"source validated count ({len(source_products)})"
                ),
                severity=PrismIssueSeverity.warning,
            )
        )

    for product in products:
        for plan in product.plans:
            for pl in plan.price_lists:
                issues.extend(
                    _validate_price_list(product, plan.name, pl)
                )

        src = source_by_name.get(_norm_name(product.name))
        if src is None or src.price is None:
            continue
        src_currency = _norm_currency(src.currency)
        src_price = float(src.price)
        same_currency_amounts: list[float] = []
        for plan in product.plans:
            for pl in plan.price_lists:
                if src_currency and _norm_currency(pl.currency) != src_currency:
                    continue
                same_currency_amounts.append(float(pl.amount))
        # Flag only when no same-currency amount is near the source selling price
        # (list/tier entries may legitimately differ).
        if same_currency_amounts and not any(
            _amounts_within_tolerance(amt, src_price, tol)
            for amt in same_currency_amounts
        ):
            issues.append(
                PrismValidationIssue(
                    product_id=product.id,
                    product_name=product.name,
                    field="price_lists.amount",
                    message=(
                        f"no price_list amount for currency {src_currency!r} "
                        f"is within {tol:.0%} of source price {src_price} "
                        f"(got {same_currency_amounts})"
                    ),
                    severity=PrismIssueSeverity.error,
                )
            )
    return issues


def _validate_price_list(
    product: RawProduct,
    plan_name: str,
    pl: RawPriceList,
) -> list[PrismValidationIssue]:
    issues: list[PrismValidationIssue] = []
    label = f"plan '{plan_name}' price_list '{pl.name}'"

    if not math.isfinite(pl.amount) or pl.amount <= 0:
        issues.append(
            PrismValidationIssue(
                product_id=product.id,
                product_name=product.name,
                field="price_lists.amount",
                message=f"{label}: amount must be a finite number > 0 (got {pl.amount})",
                severity=PrismIssueSeverity.error,
            )
        )

    if pl.discount is not None:
        if not math.isfinite(pl.discount) or pl.discount < 0 or pl.discount > pl.amount:
            issues.append(
                PrismValidationIssue(
                    product_id=product.id,
                    product_name=product.name,
                    field="price_lists.discount",
                    message=(
                        f"{label}: discount must satisfy 0 <= discount <= amount "
                        f"(got discount={pl.discount}, amount={pl.amount})"
                    ),
                    severity=PrismIssueSeverity.error,
                )
            )

    if not (pl.currency or "").strip():
        issues.append(
            PrismValidationIssue(
                product_id=product.id,
                product_name=product.name,
                field="price_lists.currency",
                message=f"{label}: currency is blank",
                severity=PrismIssueSeverity.warning,
            )
        )

    if not (pl.country or "").strip():
        issues.append(
            PrismValidationIssue(
                product_id=product.id,
                product_name=product.name,
                field="price_lists.country",
                message=f"{label}: country is blank",
                severity=PrismIssueSeverity.warning,
            )
        )

    return issues


def export_products_prism(
    products: list[ProductRecord],
    *,
    job_id: str,
    on_progress=None,
) -> PrismExportResult:
    """Convert validated products to Prism RawProduct shape via Gemini."""

    def _notify(step: str, message: str, percent: int | None = None, detail=None) -> None:
        if on_progress:
            on_progress("export_prism", step, message, percent=percent, detail=detail)

    validated = [p for p in products if p.status == ProductStatus.validated]
    source_count = len(validated)

    if source_count == 0:
        _notify("skip", "No validated products to export", percent=100)
        return PrismExportResult(
            job_id=job_id,
            products=[],
            issues=[],
            source_product_count=0,
            exported_product_count=0,
            valid=True,
        )

    batch_size = max(1, settings.prism_export_batch_size)
    batches = [
        validated[i : i + batch_size]
        for i in range(0, source_count, batch_size)
    ]
    batch_total = len(batches)
    all_exported: list[RawProduct] = []

    _notify(
        "start",
        f"Exporting {source_count} validated product(s) to Prism…",
        percent=5,
        detail={"source_product_count": source_count, "batches": batch_total},
    )

    for i, batch in enumerate(batches, start=1):
        pct = 10 + int((i - 1) / batch_total * 70)
        _notify(
            "llm_call",
            f"Gemini prism-export batch {i}/{batch_total}…",
            percent=pct,
            detail={"batch": i, "batch_total": batch_total, "size": len(batch)},
        )
        payload = json.dumps(_source_payload(batch), ensure_ascii=False, indent=2)
        chunk_label = f"batch {i}/{batch_total}, {len(batch)} product(s)"
        try:
            chunk_products = _call_gemini_structured(payload, chunk_label=chunk_label)
        except RuntimeError as exc:
            msg = str(exc)
            if "invalid JSON" in msg:
                logger.warning(
                    "Prism export batch %d/%d failed (%s); retrying once",
                    i,
                    batch_total,
                    msg,
                )
                chunk_products = _call_gemini_structured(
                    payload, chunk_label=chunk_label
                )
            else:
                raise
        all_exported.extend(chunk_products)
        logger.info(
            "Prism export batch %d/%d returned %d product(s)",
            i,
            batch_total,
            len(chunk_products),
        )

    _notify("stamp", "Stamping ids and timestamps…", percent=85)
    _stamp_ids_and_timestamps(all_exported, job_id)

    _notify("validate", "Running numeric validation…", percent=92)
    issues = _numeric_validate(all_exported, validated)
    valid = not any(iss.severity == PrismIssueSeverity.error for iss in issues)

    result = PrismExportResult(
        job_id=job_id,
        products=all_exported,
        issues=issues,
        source_product_count=source_count,
        exported_product_count=len(all_exported),
        valid=valid,
    )
    _notify(
        "done",
        f"Prism export complete — {len(all_exported)} product(s), valid={valid}",
        percent=100,
        detail={
            "exported_product_count": len(all_exported),
            "issue_count": len(issues),
            "valid": valid,
        },
    )
    return result
