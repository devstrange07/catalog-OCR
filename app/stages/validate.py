"""Stage 3 — Validation.

Promotes CandidateProduct → ProductRecord with schema + business rules.
Only this stage produces final product records.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from statistics import median

from app.config import settings
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

logger = logging.getLogger(__name__)

_CURRENCY_RE = re.compile(r"^[A-Za-z]{3}$")
_PRICE_CLEAN_RE = re.compile(r"[^\d.\-]")
_PRICE_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?")
_SCALE_SUFFIX_RE = re.compile(
    r"(?P<num>[\d,]+(?:\.\d+)?)\s*"
    r"(?P<scale>crore|lakh|lac|million|thousand|mn|cr|k|m|l)\b",
    re.IGNORECASE,
)
_SCALE_MULTIPLIERS: dict[str, float] = {
    "crore": 10_000_000,
    "cr": 10_000_000,
    "lakh": 100_000,
    "lac": 100_000,
    "l": 100_000,
    "million": 1_000_000,
    "mn": 1_000_000,
    "m": 1_000_000,
    "thousand": 1_000,
    "k": 1_000,
}
_RATE_TIER_RE = re.compile(r"\b(rate|psf|p\.s\.f)\b|per\s|/\s*sq", re.IGNORECASE)
_FOOTNOTE_RE = re.compile(
    r"^(?:\*|†|‡|note\b|n\.b\.|disclaimer\b)",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(r"\d+(?:\.\d+)?\s*%")
_NUMBER_UNIT_RE = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:lakh|lac|crore|cr|sq\.?\s*ft|sqft|kg|g|ml|l|hrs?|min)\b",
    re.IGNORECASE,
)
_PERK_WORD_RE = re.compile(
    r"\b(free|included|complimentary|bonus|gift|waived|cashback)\b",
    re.IGNORECASE,
)
_HEADING_WORDS = frozenset(
    {"table", "comparison", "chart", "list", "snapshot", "overview", "matrix"}
)
_SCOPED_FIELDS = (
    "conditions",
    "promotion",
    "validity",
    "minimum_order",
    "category",
    "rate_tier",
)
# Footnote/header-style fields: under-population vs a similar table is suspicious.
# promotion/category are often legitimately section-local, so they are excluded.
_ASYMMETRIC_FIELDS = (
    "conditions",
    "validity",
    "minimum_order",
    "rate_tier",
)
_MAGNITUDE_RATIO = 20.0
_TABLE_JACCARD_MIN = 0.6
_TABLE_SIZE_RATIO_MIN = 0.5
_FIELD_HIGH_RATE = 0.8
_FIELD_LOW_RATE = 0.15

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


def _scale_from_text(text: str) -> tuple[float, str] | None:
    match = _SCALE_SUFFIX_RE.search(text)
    if not match:
        return None
    scale = match.group("scale").lower()
    multiplier = _SCALE_MULTIPLIERS.get(scale)
    if multiplier is None:
        return None
    raw_num = match.group("num").replace(",", "")
    try:
        return float(raw_num) * multiplier, match.group("scale")
    except ValueError:
        return None


def _normalize_price(
    value: float | str | None,
) -> tuple[float | None, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if value is None or value == "":
        return None, errors, warnings
    if isinstance(value, (int, float)):
        return float(value), errors, warnings
    text = str(value).strip()
    scaled = _scale_from_text(text)
    if scaled is not None:
        amount, suffix = scaled
        warnings.append(f"Applied scale suffix {suffix!r} to price {value!r}")
        return amount, errors, warnings
    cleaned = _PRICE_CLEAN_RE.sub("", text.replace(",", ""))
    if not cleaned or cleaned in {".", "-", "-."}:
        errors.append(f"Unparseable price: {value!r}")
        return None, errors, warnings
    try:
        return float(cleaned), errors, warnings
    except ValueError:
        errors.append(f"Unparseable price: {value!r}")
        return None, errors, warnings


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
    scaled = _scale_from_text(text)
    if scaled is not None:
        amount, suffix = scaled
        return amount, None, f"Applied scale suffix {suffix!r} to price {value!r}"
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

    price, price_errors, price_warnings = _normalize_price(candidate.price)
    errors.extend(price_errors)
    warnings.extend(price_warnings)
    list_price, list_price_errors, list_price_warnings = _normalize_price(
        candidate.list_price
    )
    # list_price parse issues are warnings, not rejection
    for err in list_price_errors:
        warnings.append(err.replace("Unparseable price", "Unparseable list_price"))
    warnings.extend(list_price_warnings)

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


def _has_scope(record: ProductRecord) -> bool:
    return bool((record.project_name or "").strip() or (record.category or "").strip())


def _has_rate_tier(record: ProductRecord) -> bool:
    for tier in record.price_tiers or []:
        if _RATE_TIER_RE.search(tier.label or ""):
            return True
    return False


def _field_populated(record: ProductRecord, field: str) -> bool:
    if field == "conditions":
        return bool(record.conditions)
    if field == "promotion":
        return record.promotion is not None
    if field == "validity":
        return bool((record.validity or "").strip())
    if field == "minimum_order":
        return bool((record.minimum_order or "").strip())
    if field == "category":
        return bool((record.category or "").strip())
    if field == "rate_tier":
        return _has_rate_tier(record)
    return False


def _populated_field_set(record: ProductRecord) -> set[str]:
    return {name for name in _SCOPED_FIELDS if _field_populated(record, name)}


def _structure_signature(record: ProductRecord) -> set[str]:
    """Fields used to decide whether two tables are structurally similar."""
    fields = _populated_field_set(record)
    if record.price is not None:
        fields.add("price")
    if (record.unit or "").strip():
        fields.add("unit")
    if (record.pack_size or "").strip():
        fields.add("pack_size")
    if record.price_tiers:
        fields.add("price_tiers")
    if (record.sku or "").strip():
        fields.add("sku")
    return fields


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _append_warning(record: ProductRecord, message: str) -> None:
    if message not in record.warnings:
        record.warnings.append(message)


def _table_groups(records: list[ProductRecord]) -> dict[tuple[int | None, str], list[int]]:
    groups: dict[tuple[int | None, str], list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        groups[(rec.page, _scope_bucket(rec))].append(i)
    return groups


def _flag_asymmetric_table_fields(records: list[ProductRecord]) -> None:
    groups = [
        (key, idxs)
        for key, idxs in _table_groups(records).items()
        if len(idxs) >= 2
    ]
    if len(groups) < 2:
        return
    signatures: list[tuple[tuple[int | None, str], list[int], set[str]]] = []
    for key, idxs in groups:
        union: set[str] = set()
        for i in idxs:
            union |= _structure_signature(records[i])
        signatures.append((key, idxs, union))

    for i, (key_a, idxs_a, fields_a) in enumerate(signatures):
        for key_b, idxs_b, fields_b in signatures[i + 1 :]:
            if key_a[0] != key_b[0]:
                continue
            size_ratio = min(len(idxs_a), len(idxs_b)) / max(len(idxs_a), len(idxs_b))
            if size_ratio < _TABLE_SIZE_RATIO_MIN:
                continue
            if _jaccard(fields_a, fields_b) < _TABLE_JACCARD_MIN:
                continue
            for field in _ASYMMETRIC_FIELDS:
                rate_a = sum(1 for idx in idxs_a if _field_populated(records[idx], field)) / len(
                    idxs_a
                )
                rate_b = sum(1 for idx in idxs_b if _field_populated(records[idx], field)) / len(
                    idxs_b
                )
                if rate_a >= _FIELD_HIGH_RATE and rate_b <= _FIELD_LOW_RATE:
                    msg = (
                        f"Similar table on this page populates '{field}' on most items; "
                        "this table has none — possible missed footnote"
                    )
                    for idx in idxs_b:
                        _append_warning(records[idx], msg)
                elif rate_b >= _FIELD_HIGH_RATE and rate_a <= _FIELD_LOW_RATE:
                    msg = (
                        f"Similar table on this page populates '{field}' on most items; "
                        "this table has none — possible missed footnote"
                    )
                    for idx in idxs_a:
                        _append_warning(records[idx], msg)


def _structured_blob(record: ProductRecord) -> str:
    parts: list[str] = [
        record.name or "",
        record.brand or "",
        record.sku or "",
        record.description or "",
        record.unit or "",
        record.pack_size or "",
        record.validity or "",
        record.minimum_order or "",
        record.category or "",
        record.project_name or "",
        record.location or "",
        str(record.price) if record.price is not None else "",
    ]
    parts.extend(record.includes or [])
    parts.extend(record.conditions or [])
    parts.extend(record.tags or [])
    promo = record.promotion
    if promo is not None:
        parts.extend(
            [
                promo.type or "",
                promo.label or "",
                promo.details or "",
                promo.conditions or "",
            ]
        )
    for tier in record.price_tiers or []:
        parts.extend([tier.label, tier.details or "", str(tier.price or "")])
    for variant in record.variants or []:
        parts.extend(
            [variant.attribute, variant.label, variant.description or "", str(variant.price or "")]
        )
    return " ".join(parts).lower()


def _interesting_snippet_tokens(snippet: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for match in list(_PERCENT_RE.finditer(snippet)) + list(_NUMBER_UNIT_RE.finditer(snippet)):
        token = match.group(0).strip()
        key = token.lower()
        if key not in seen:
            seen.add(key)
            tokens.append(token)
    for match in _PERK_WORD_RE.finditer(snippet):
        token = match.group(0).strip()
        key = token.lower()
        if key not in seen:
            seen.add(key)
            tokens.append(token)
    return tokens


def _record_numeric_values(record: ProductRecord) -> set[float]:
    values: set[float] = set()
    if record.price is not None:
        values.add(float(record.price))
    if record.list_price is not None:
        values.add(float(record.list_price))
    for tier in record.price_tiers or []:
        if isinstance(tier.price, (int, float)):
            values.add(float(tier.price))
    for variant in record.variants or []:
        if isinstance(variant.price, (int, float)):
            values.add(float(variant.price))
    return values


def _token_reflected_in_record(
    token: str,
    blob: str,
    numeric_values: set[float],
) -> bool:
    if token.lower() in blob:
        return True
    scaled = _scale_from_text(token)
    if scaled is None:
        return False
    amount, _suffix = scaled
    for value in numeric_values:
        if value <= 0:
            continue
        if abs(value - amount) <= 0.5:
            return True
        if abs(value - amount) / max(abs(value), abs(amount)) < 0.001:
            return True
    return False


def _flag_snippet_coverage(records: list[ProductRecord]) -> None:
    for rec in records:
        snippet = (rec.source_snippet or "").strip()
        if not snippet:
            continue
        blob = _structured_blob(rec)
        nums = _record_numeric_values(rec)
        for token in _interesting_snippet_tokens(snippet):
            if not _token_reflected_in_record(token, blob, nums):
                _append_warning(
                    rec,
                    f"Source snippet mentions {token!r} not reflected in any structured field",
                )


def _consumed_text(records: list[ProductRecord]) -> str:
    parts: list[str] = []
    for rec in records:
        parts.append(_structured_blob(rec))
        parts.append((rec.source_snippet or "").lower())
    return " ".join(parts)


def _flag_unclaimed_footnotes(
    records: list[ProductRecord],
    parsed: ParsedDocument | None,
) -> None:
    if parsed is None:
        return
    consumed = _consumed_text(records)
    page_records: dict[int, list[ProductRecord]] = defaultdict(list)
    for rec in records:
        if rec.page is not None:
            page_records[rec.page].append(rec)
    for page in parsed.pages:
        for block in page.blocks:
            text = (block.text or "").strip()
            if not text or not _FOOTNOTE_RE.search(text):
                continue
            if text.lower() in consumed or text.lower()[:40] in consumed:
                continue
            targets = page_records.get(page.page_number) or records
            msg = f"Block {block.block_id} looks like an unattached table footnote"
            for rec in targets:
                _append_warning(rec, msg)


def _pack_size_number(pack_size: str | None) -> float | None:
    if not pack_size:
        return None
    match = _PRICE_TOKEN_RE.search(pack_size.replace(",", ""))
    if not match:
        return None
    try:
        value = float(match.group(0))
    except ValueError:
        return None
    return value if value > 0 else None


def _flag_missing_derivable_rate(records: list[ProductRecord]) -> None:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        if _has_scope(rec):
            groups[_scope_bucket(rec)].append(i)
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        has_rate = any(_has_rate_tier(records[i]) for i in idxs)
        if not has_rate:
            continue
        for i in idxs:
            rec = records[i]
            if _has_rate_tier(rec):
                continue
            if rec.price is None or _pack_size_number(rec.pack_size) is None:
                continue
            _append_warning(
                rec,
                "Sibling records have an explicit rate tier; this one is computable but missing",
            )


def _flag_completeness_gaps(
    records: list[ProductRecord],
    parsed: ParsedDocument | None = None,
) -> None:
    _flag_asymmetric_table_fields(records)
    _flag_snippet_coverage(records)
    _flag_unclaimed_footnotes(records, parsed)
    _flag_missing_derivable_rate(records)


def _numeric_groups(records: list[ProductRecord]) -> dict[tuple[str, str], list[tuple[int, float, str]]]:
    groups: dict[tuple[str, str], list[tuple[int, float, str]]] = defaultdict(list)
    for i, rec in enumerate(records):
        if not _has_scope(rec):
            continue
        bucket = _scope_bucket(rec)
        if rec.price is not None:
            groups[(bucket, "price")].append((i, rec.price, "price"))
        for tier in rec.price_tiers or []:
            if isinstance(tier.price, (int, float)):
                label = (tier.label or "").strip().lower() or "tier"
                groups[(bucket, f"tier:{label}")].append((i, float(tier.price), f"tier '{tier.label}'"))
        for variant in rec.variants or []:
            if isinstance(variant.price, (int, float)):
                key = f"variant:{(variant.attribute or '').strip().lower()}|{(variant.label or '').strip().lower()}"
                groups[(bucket, key)].append(
                    (i, float(variant.price), f"variant '{variant.label}'")
                )
    return groups


def _flag_magnitude_inconsistency(records: list[ProductRecord]) -> None:
    for (_bucket, _series), items in _numeric_groups(records).items():
        if len(items) < 3:
            continue
        values = [v for _i, v, _label in items if v > 0]
        if len(values) < 3:
            continue
        mid = median(values)
        if mid <= 0:
            continue
        outliers = [
            (i, value, label)
            for i, value, label in items
            if value > 0 and max(value, mid) / min(value, mid) >= _MAGNITUDE_RATIO
        ]
        if not outliers:
            continue
        series_label = outliers[0][2]
        msg = (
            f"{series_label} values in this table span an implausible range "
            "— check for inconsistent scale"
        )
        for i, _value, _label in items:
            _append_warning(records[i], msg)


def _flag_currency_inconsistency(records: list[ProductRecord]) -> None:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        if _has_scope(rec) and rec.currency:
            groups[_scope_bucket(rec)].append(i)
    for idxs in groups.values():
        codes = {(records[i].currency or "").upper() for i in idxs}
        codes.discard("")
        if len(codes) <= 1:
            continue
        listed = ", ".join(sorted(codes))
        msg = f"Mixed currencies in the same section ({listed})"
        for i in idxs:
            _append_warning(records[i], msg)


def _looks_like_heading(value: str) -> bool:
    tokens = value.split()
    lowered = {t.strip(".,:;").lower() for t in tokens}
    if lowered & _HEADING_WORDS:
        return True
    return len(tokens) >= 5


def _flag_category_shape_drift(records: list[ProductRecord]) -> None:
    counts: dict[str, int] = defaultdict(int)
    for rec in records:
        value = (rec.category or "").strip()
        if value:
            counts[value] += 1
    if len(counts) < 2:
        return
    repeated = [value for value, n in counts.items() if n >= 3]
    if not repeated:
        return
    for rec in records:
        value = (rec.category or "").strip()
        if not value or counts[value] != 1:
            continue
        if not _looks_like_heading(value):
            continue
        _append_warning(
            rec,
            f"category value {value!r} looks like a section/table title, not a taxonomy label",
        )


def _flag_heading_category_bleed(
    records: list[ProductRecord],
    parsed: ParsedDocument | None,
) -> None:
    if parsed is None:
        return
    headings: set[str] = set()
    for page in parsed.pages:
        for block in page.blocks:
            text = (block.text or "").strip()
            if text and _looks_like_heading(text):
                headings.add(text.lower())
    if not headings:
        return
    for rec in records:
        value = (rec.category or "").strip()
        if value and value.lower() in headings:
            siblings = [
                other
                for other in records
                if (other.category or "").strip().lower() == value.lower()
            ]
            if len(siblings) <= 1:
                _append_warning(
                    rec,
                    f"category value {value!r} matches a section heading and is not shared as a taxonomy label",
                )


def _flag_semantic_plausibility(
    records: list[ProductRecord],
    parsed: ParsedDocument | None = None,
) -> None:
    _flag_magnitude_inconsistency(records)
    _flag_currency_inconsistency(records)
    _flag_category_shape_drift(records)
    _flag_heading_category_bleed(records, parsed)


def _maybe_repair_coverage_gaps(records: list[ProductRecord]) -> None:
    if not settings.validate_coverage_repair:
        return
    flagged = [
        rec
        for rec in records
        if any("Source snippet mentions" in w for w in rec.warnings)
    ]
    if not flagged:
        return
    try:
        from app.stages.coverage_repair import repair_coverage_gaps

        repair_coverage_gaps(flagged)
    except Exception:
        logger.warning("Coverage repair skipped", exc_info=True)


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
    _flag_completeness_gaps(records, parsed)
    _flag_semantic_plausibility(records, parsed)
    _maybe_repair_coverage_gaps(records)
    validated = sum(1 for r in records if r.status == ProductStatus.validated)
    rejected = sum(1 for r in records if r.status == ProductStatus.rejected)
    _notify(
        "validate_done",
        f"{validated} validated, {rejected} rejected",
        percent=95,
        detail={"validated": validated, "rejected": rejected},
    )
    return records
