"""Stage 3 — Validation.

Promotes CandidateProduct → ProductRecord with schema + business rules.
Only this stage produces final product records.
"""

from __future__ import annotations

import re
from collections import defaultdict

from app.models.document import ParsedDocument, TextBlock
from app.models.product import CandidateProduct, ProductRecord, ProductStatus

_CURRENCY_RE = re.compile(r"^[A-Za-z]{3}$")
_PRICE_CLEAN_RE = re.compile(r"[^\d.\-]")
_PRICE_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?")

_VARIANT_PRICE_SENTINELS = {
    "free",
    "included",
    "complimentary",
    "n/a",
    "na",
    "tbd",
    "on request",
    "-",
}


def _normalize_price(value: float | str | None) -> tuple[float | None, list[str]]:
    errors: list[str] = []
    if value is None or value == "":
        return None, errors
    if isinstance(value, (int, float)):
        return float(value), errors
    text = str(value).strip()
    cleaned = _PRICE_CLEAN_RE.sub("", text.replace(",", ""))
    if not cleaned or cleaned in {".", "-", "-."}:
        errors.append(f"Unparseable price: {value!r}")
        return None, errors
    try:
        return float(cleaned), errors
    except ValueError:
        errors.append(f"Unparseable price: {value!r}")
        return None, errors


def _normalize_currency(value: str | None) -> tuple[str | None, list[str]]:
    warnings: list[str] = []
    if value is None or not str(value).strip():
        return None, warnings
    code = str(value).strip().upper()
    if not _CURRENCY_RE.match(code):
        warnings.append(f"Currency not ISO-like: {value!r}")
        return code[:8], warnings
    return code, warnings


def _normalize_variant_price(
    value: float | str | None,
) -> tuple[float | None, str | None, str | None]:
    """Returns (price, available, warning). Sentinel/unparseable values never raise errors."""
    if value is None or value == "":
        return None, None, None
    if isinstance(value, (int, float)):
        return float(value), None, None
    text = str(value).strip()
    lowered = text.lower()
    if lowered in _VARIANT_PRICE_SENTINELS:
        return None, lowered, None
    cleaned = _PRICE_CLEAN_RE.sub("", text.replace(",", ""))
    if not cleaned or cleaned in {".", "-", "-."}:
        return None, None, f"Unparseable variant price: {value!r}"
    try:
        return float(cleaned), None, None
    except ValueError:
        return None, None, f"Unparseable variant price: {value!r}"


def _normalize_price_variants(candidate: CandidateProduct) -> tuple[list[dict] | None, list[str]]:
    """Returns (normalized_variants, product_level_errors). Variant price issues become
    per-variant warnings, never product-level errors."""
    product_level_errors: list[str] = []
    if not candidate.price_variants:
        return None, product_level_errors

    normalized: list[dict] = []
    for variant in candidate.price_variants:
        price, available, warning = _normalize_variant_price(variant.price)
        variant_warnings = list(variant.warnings)
        if warning:
            variant_warnings.append(warning)
        normalized.append(
            {
                "label": variant.label,
                "price": price,
                "details": (variant.details or "").strip() or None,
                "attribute_name": variant.attribute_name,
                "sku": variant.sku,
                "currency": variant.currency,
                "unit": variant.unit,
                "pack_size": variant.pack_size,
                "available": available if available is not None else variant.available,
                "is_default": variant.is_default,
                "warnings": variant_warnings,
            }
        )
    return normalized, product_level_errors


def _best_name_block(blocks: list[TextBlock], name: str) -> TextBlock | None:
    needle = name.strip().lower()
    if not needle:
        return None
    exact = [b for b in blocks if needle in b.text.lower()]
    if exact:
        return max(exact, key=lambda b: len(needle))
    return None


def _best_price_block(blocks: list[TextBlock], price_value: float | str | None) -> TextBlock | None:
    if price_value is None:
        return None
    if isinstance(price_value, float) and price_value.is_integer():
        price_text = str(int(price_value))
    else:
        price_text = str(price_value)
    clean = _PRICE_CLEAN_RE.sub("", price_text.replace(",", ""))
    if not clean:
        return None
    token_match = _PRICE_TOKEN_RE.search(clean)
    if not token_match:
        return None
    token = token_match.group(0)
    hits = [b for b in blocks if token in b.text]
    if hits:
        return hits[0]
    return None


def _flag_spatial_mismatch(
    candidate: CandidateProduct,
    warnings: list[str],
    page_blocks: dict[int, list[TextBlock]],
) -> None:
    if candidate.page is None or not candidate.source_snippet:
        return
    blocks = page_blocks.get(candidate.page, [])
    if not blocks:
        return
    name_block = _best_name_block(blocks, candidate.name)
    if not name_block or name_block.row_id is None:
        return

    price_block = _best_price_block(blocks, candidate.price)
    if price_block and price_block.row_id is not None:
        if name_block.row_id != price_block.row_id:
            warnings.append(
                "Spatial check: name and price token are not on the same OCR row (possible row-shift)"
            )

    for variant in candidate.price_variants or []:
        variant_block = _best_price_block(blocks, variant.price)
        if not variant_block or variant_block.row_id is None:
            continue
        if name_block.row_id != variant_block.row_id:
            label = variant.attribute_name or variant.label or "variant"
            warnings.append(
                f"Spatial check: variant '{label}' price token is not on the same OCR row "
                "as name (possible row-shift)"
            )


def _validate_one(
    candidate: CandidateProduct,
    page_blocks: dict[int, list[TextBlock]] | None = None,
) -> ProductRecord:
    errors: list[str] = []
    warnings: list[str] = []

    name = (candidate.name or "").strip()
    if not name:
        errors.append("Missing required field: name")

    price, price_errors = _normalize_price(candidate.price)
    errors.extend(price_errors)
    price_variants, variant_errors = _normalize_price_variants(candidate)
    errors.extend(variant_errors)
    for variant in price_variants or []:
        warnings.extend(variant.get("warnings", []))

    currency, currency_warnings = _normalize_currency(candidate.currency)
    warnings.extend(currency_warnings)

    confidence = candidate.confidence
    if confidence is not None and not (0.0 <= confidence <= 1.0):
        warnings.append(f"Confidence out of range, clamped: {confidence}")
        confidence = max(0.0, min(1.0, float(confidence)))

    if page_blocks:
        _flag_spatial_mismatch(candidate, warnings, page_blocks)

    # Empty shell: no name and nothing else useful
    if not name and not any(
        [
            candidate.brand,
            candidate.sku,
            candidate.product_code,
            candidate.price,
            candidate.description,
            candidate.price_variants,
        ]
    ):
        errors.append("Empty product shell")

    status = ProductStatus.validated if not errors else ProductStatus.rejected

    return ProductRecord(
        name=name or "(unnamed)",
        brand=(candidate.brand or None),
        sku=(candidate.sku or None),
        product_code=(candidate.product_code or None),
        price=price,
        currency=currency,
        unit=(candidate.unit or None),
        pack_size=(candidate.pack_size or None),
        description=(candidate.description or None),
        page=candidate.page,
        source_snippet=(candidate.source_snippet or None),
        confidence=confidence,
        price_variants=price_variants,
        status=status,
        validation_errors=errors,
        warnings=warnings,
    )


def _flag_duplicates(records: list[ProductRecord]) -> None:
    key_to_idxs: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        key = f"{rec.name.strip().lower()}|{rec.sku or ''}|{rec.product_code or ''}"
        key_to_idxs[key].append(i)
    for idxs in key_to_idxs.values():
        if len(idxs) > 1:
            for i in idxs:
                records[i].warnings.append("Possible duplicate product in document")


def validate_products(
    candidates: list[CandidateProduct],
    parsed: ParsedDocument | None = None,
    on_progress=None,
) -> list[ProductRecord]:
    """Stage 3 entrypoint."""

    def _notify(step: str, message: str, percent: int | None = None, detail=None) -> None:
        if on_progress:
            on_progress("validate", step, message, percent=percent, detail=detail)

    total = len(candidates)
    _notify(
        "validate_start",
        f"Validating {total} candidate(s)…",
        percent=88,
        detail={"candidate_count": total},
    )
    page_blocks: dict[int, list[TextBlock]] = {}
    if parsed is not None:
        for page in parsed.pages:
            page_blocks[page.page_number] = list(page.blocks)
    records = [_validate_one(c, page_blocks=page_blocks) for c in candidates]
    _flag_duplicates(records)
    validated = sum(1 for r in records if r.status == ProductStatus.validated)
    rejected = sum(1 for r in records if r.status == ProductStatus.rejected)
    _notify(
        "validate_done",
        f"{validated} validated, {rejected} rejected",
        percent=95,
        detail={"validated": validated, "rejected": rejected},
    )
    return records
