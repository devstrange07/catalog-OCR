"""Stage 3 — Validation.

Promotes CandidateProduct → ProductRecord with schema + business rules.
Only this stage produces final product records.
"""

from __future__ import annotations

import re
from collections import defaultdict

from app.models.document import ParsedDocument, TextBlock
from app.models.product import (
    Availability,
    CandidateProduct,
    Discount,
    ItemType,
    PriceTier,
    ProductRecord,
    ProductStatus,
    Promotion,
    Variant,
)

_CURRENCY_RE = re.compile(r"^[A-Za-z]{3}$")
_PRICE_CLEAN_RE = re.compile(r"[^\d.\-]")
_PRICE_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?")

_SYMBOL_TO_ISO: dict[str, str] = {
    "₹": "INR",
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₩": "KRW",
    "฿": "THB",
    "R$": "BRL",
    "zł": "PLN",
    "RM": "MYR",
    "S$": "SGD",
    "HK$": "HKD",
    "A$": "AUD",
    "C$": "CAD",
    "NZ$": "NZD",
}

_VARIANT_PRICE_SENTINELS = {
    "free",
    "included",
    "complimentary",
    "n/a",
    "na",
    "tbd",
    "on request",
    "on_request",
    "-",
}

_ITEM_TYPE_VALUES = {e.value for e in ItemType}
_AVAILABILITY_VALUES = {e.value for e in Availability}


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
    raw = str(value).strip()
    # Prefer longer multi-char symbols first (R$, HK$, etc.)
    for symbol, iso in sorted(_SYMBOL_TO_ISO.items(), key=lambda kv: -len(kv[0])):
        if raw == symbol or raw.startswith(symbol):
            return iso, warnings
    code = raw.upper()
    if _CURRENCY_RE.match(code):
        return code, warnings
    warnings.append(f"Currency not ISO-like: {value!r}")
    return code[:8], warnings


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


def _normalize_variants(
    candidate: CandidateProduct,
) -> tuple[list[Variant] | None, list[str]]:
    product_level_warnings: list[str] = []
    if not candidate.variants:
        return None, product_level_warnings

    normalized: list[Variant] = []
    for variant in candidate.variants:
        price, available, warning = _normalize_variant_price(variant.price)
        variant_warnings = list(variant.warnings)
        if warning:
            variant_warnings.append(warning)
            product_level_warnings.append(warning)
        currency, curr_warnings = _normalize_currency(variant.currency)
        variant_warnings.extend(curr_warnings)
        product_level_warnings.extend(curr_warnings)
        attribute = (variant.attribute or "").strip() or "Option"
        label = (variant.label or "").strip() or "Unnamed"
        normalized.append(
            Variant(
                attribute=attribute,
                label=label,
                price=price,
                currency=currency,
                unit=(variant.unit or None),
                pack_size=(variant.pack_size or None),
                sku=(variant.sku or None),
                description=(variant.description or "").strip() or None,
                available=available if available is not None else variant.available,
                is_default=bool(variant.is_default),
                warnings=variant_warnings,
            )
        )
    return normalized, product_level_warnings


def _normalize_price_tiers(
    candidate: CandidateProduct,
) -> tuple[list[PriceTier] | None, list[str]]:
    product_level_warnings: list[str] = []
    if not candidate.price_tiers:
        return None, product_level_warnings

    normalized: list[PriceTier] = []
    for tier in candidate.price_tiers:
        price, _available, warning = _normalize_variant_price(tier.price)
        tier_warnings = list(tier.warnings)
        if warning:
            # Rephrase for tiers (same underlying parser)
            warning = warning.replace("variant price", "tier price")
            tier_warnings.append(warning)
            product_level_warnings.append(warning)
        currency, curr_warnings = _normalize_currency(tier.currency)
        tier_warnings.extend(curr_warnings)
        product_level_warnings.extend(curr_warnings)
        label = (tier.label or "").strip() or "Unnamed"
        normalized.append(
            PriceTier(
                label=label,
                price=price,
                details=(tier.details or "").strip() or None,
                is_current=True if tier.is_current is None else bool(tier.is_current),
                currency=currency,
                billing_cycle=(tier.billing_cycle or None),
                warnings=tier_warnings,
            )
        )
    return normalized, product_level_warnings


def _normalize_discount(discount: Discount | None) -> Discount | None:
    if discount is None:
        return None
    amount = discount.amount
    percent = discount.percent
    description = (discount.description or "").strip() or None
    if amount is None and percent is None and description is None:
        return None
    return Discount(amount=amount, percent=percent, description=description)


def _normalize_promotion(promotion: Promotion | None) -> Promotion | None:
    if promotion is None:
        return None
    type_ = (promotion.type or "").strip() or None
    label = (promotion.label or "").strip() or None
    details = (promotion.details or "").strip() or None
    conditions = (promotion.conditions or "").strip() or None
    if not any([type_, label, details, conditions]):
        return None
    return Promotion(type=type_, label=label, details=details, conditions=conditions)


def _normalize_item_type(value: ItemType | str | None) -> tuple[ItemType, list[str]]:
    warnings: list[str] = []
    if value is None or value == "":
        return ItemType.product, warnings
    if isinstance(value, ItemType):
        return value, warnings
    text = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    # Common aliases
    aliases = {
        "combo": ItemType.bundle,
        "package": ItemType.bundle,
        "deal": ItemType.product,
        "offer": ItemType.product,
        "subscription": ItemType.plan,
        "membership": ItemType.plan,
        "ticket": ItemType.event,
        "extra": ItemType.addon,
        "add_on": ItemType.addon,
    }
    if text in aliases:
        return aliases[text], warnings
    if text in _ITEM_TYPE_VALUES:
        return ItemType(text), warnings
    warnings.append(f"Unknown item_type {value!r}; defaulting to 'other'")
    return ItemType.other, warnings


def _normalize_availability(
    value: Availability | str | None,
) -> tuple[Availability, list[str]]:
    warnings: list[str] = []
    if value is None or value == "":
        return Availability.available, warnings
    if isinstance(value, Availability):
        return value, warnings
    text = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "instock": Availability.available,
        "in_stock": Availability.available,
        "sold_out": Availability.out_of_stock,
        "outofstock": Availability.out_of_stock,
        "backorder": Availability.pre_order,
        "contact": Availability.on_request,
        "contact_sales": Availability.on_request,
        "call_for_pricing": Availability.on_request,
        "tba": Availability.coming_soon,
        "limited_stock": Availability.limited,
        "while_supplies_last": Availability.limited,
    }
    if text in aliases:
        return aliases[text], warnings
    if text in _AVAILABILITY_VALUES:
        return Availability(text), warnings
    warnings.append(f"Unknown availability {value!r}; defaulting to 'unknown'")
    return Availability.unknown, warnings


def _clean_str_list(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    cleaned = [str(v).strip() for v in values if str(v).strip()]
    return cleaned or None


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

    for variant in candidate.variants or []:
        variant_block = _best_price_block(blocks, variant.price)
        if not variant_block or variant_block.row_id is None:
            continue
        if name_block.row_id != variant_block.row_id:
            label = variant.label or variant.attribute or "variant"
            warnings.append(
                f"Spatial check: variant '{label}' price token is not on the same OCR row "
                "as name (possible row-shift)"
            )

    for tier in candidate.price_tiers or []:
        tier_block = _best_price_block(blocks, tier.price)
        if not tier_block or tier_block.row_id is None:
            continue
        if name_block.row_id != tier_block.row_id:
            label = tier.label or "tier"
            warnings.append(
                f"Spatial check: price tier '{label}' price token is not on the same OCR row "
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
    list_price, list_price_errors = _normalize_price(candidate.list_price)
    # list_price parse issues are warnings, not rejection
    for err in list_price_errors:
        warnings.append(err.replace("Unparseable price", "Unparseable list_price"))

    variants, variant_warnings = _normalize_variants(candidate)
    warnings.extend(variant_warnings)
    price_tiers, tier_warnings = _normalize_price_tiers(candidate)
    warnings.extend(tier_warnings)

    currency, currency_warnings = _normalize_currency(candidate.currency)
    warnings.extend(currency_warnings)

    item_type, item_type_warnings = _normalize_item_type(candidate.item_type)
    warnings.extend(item_type_warnings)
    availability, availability_warnings = _normalize_availability(candidate.availability)
    warnings.extend(availability_warnings)

    discount = _normalize_discount(candidate.discount)
    promotion = _normalize_promotion(candidate.promotion)

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
            candidate.list_price,
            candidate.description,
            candidate.variants,
            candidate.price_tiers,
            candidate.includes,
            candidate.promotion,
        ]
    ):
        errors.append("Empty product shell")

    status = ProductStatus.validated if not errors else ProductStatus.rejected

    return ProductRecord(
        name=name or "(unnamed)",
        brand=(candidate.brand or None),
        sku=(candidate.sku or None),
        product_code=(candidate.product_code or None),
        item_type=item_type,
        category=(candidate.category or None),
        project_name=(candidate.project_name or None),
        location=(candidate.location or None),
        price=price,
        list_price=list_price,
        currency=currency,
        price_qualifier=(candidate.price_qualifier or None),
        billing_cycle=(candidate.billing_cycle or None),
        discount=discount,
        unit=(candidate.unit or None),
        pack_size=(candidate.pack_size or None),
        quantity=candidate.quantity,
        promotion=promotion,
        validity=(candidate.validity or None),
        includes=_clean_str_list(candidate.includes),
        description=(candidate.description or None),
        conditions=_clean_str_list(candidate.conditions),
        minimum_order=(candidate.minimum_order or None),
        tags=_clean_str_list(candidate.tags),
        availability=availability,
        page=candidate.page,
        source_snippet=(candidate.source_snippet or None),
        confidence=confidence,
        price_tiers=price_tiers,
        variants=variants,
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


def _scope_bucket(record: ProductRecord) -> str:
    """Section identity used to detect cross-scope attribute bleed."""
    project = (record.project_name or "").strip().lower()
    category = (record.category or "").strip().lower()
    return f"{project}|{category}"


def _promotion_key(record: ProductRecord) -> str | None:
    promo = record.promotion
    if promo is None:
        return None
    parts = [
        (promo.type or "").strip().lower(),
        (promo.label or "").strip().lower(),
        (promo.details or "").strip().lower(),
        (promo.conditions or "").strip().lower(),
    ]
    key = "|".join(parts)
    return key if any(parts) else None


def _conditions_key(record: ProductRecord) -> str | None:
    if not record.conditions:
        return None
    normalized = sorted(c.strip().lower() for c in record.conditions if c and c.strip())
    return "||".join(normalized) if normalized else None


def _downgrade_confidence(record: ProductRecord, delta: float = 0.2) -> None:
    if record.confidence is None:
        return
    record.confidence = max(record.confidence - delta, 0.5)


def _flag_uniform_attribute_bleed(
    records: list[ProductRecord],
    *,
    attr_key_fn,
    warning: str,
) -> None:
    """Warn when the same attribute value spans >50% of items across distinct scopes."""
    total = len(records)
    if total < 2:
        return

    groups: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        key = attr_key_fn(rec)
        if key:
            groups[key].append(i)

    for idxs in groups.values():
        if len(idxs) / total <= 0.5:
            continue
        scopes = {_scope_bucket(records[i]) for i in idxs}
        if len(scopes) < 2:
            continue
        for i in idxs:
            if warning not in records[i].warnings:
                records[i].warnings.append(warning)
            _downgrade_confidence(records[i])


def _flag_scope_anomalies(records: list[ProductRecord]) -> None:
    """Soft checks for promotion/conditions broadcast across sections."""
    _flag_uniform_attribute_bleed(
        records,
        attr_key_fn=_promotion_key,
        warning="Promotion may be over-applied across sections",
    )
    _flag_uniform_attribute_bleed(
        records,
        attr_key_fn=_conditions_key,
        warning="Conditions may be over-applied from a single section",
    )


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
    _flag_scope_anomalies(records)
    validated = sum(1 for r in records if r.status == ProductStatus.validated)
    rejected = sum(1 for r in records if r.status == ProductStatus.rejected)
    _notify(
        "validate_done",
        f"{validated} validated, {rejected} rejected",
        percent=95,
        detail={"validated": validated, "rejected": rejected},
    )
    return records
