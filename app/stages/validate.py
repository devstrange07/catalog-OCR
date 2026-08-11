"""Stage 3 — Validation.

Promotes CandidateProduct → ProductRecord with schema + business rules.
Only this stage produces final product records.
"""

from __future__ import annotations

import re
from collections import defaultdict

from app.models.product import CandidateProduct, ProductRecord, ProductStatus

_CURRENCY_RE = re.compile(r"^[A-Za-z]{3}$")
_PRICE_CLEAN_RE = re.compile(r"[^\d.\-]")


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


def _validate_one(candidate: CandidateProduct) -> ProductRecord:
    errors: list[str] = []
    warnings: list[str] = []

    name = (candidate.name or "").strip()
    if not name:
        errors.append("Missing required field: name")

    price, price_errors = _normalize_price(candidate.price)
    errors.extend(price_errors)

    currency, currency_warnings = _normalize_currency(candidate.currency)
    warnings.extend(currency_warnings)

    confidence = candidate.confidence
    if confidence is not None and not (0.0 <= confidence <= 1.0):
        warnings.append(f"Confidence out of range, clamped: {confidence}")
        confidence = max(0.0, min(1.0, float(confidence)))

    # Empty shell: no name and nothing else useful
    if not name and not any(
        [
            candidate.brand,
            candidate.sku,
            candidate.product_code,
            candidate.price,
            candidate.description,
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


def validate_products(candidates: list[CandidateProduct], on_progress=None) -> list[ProductRecord]:
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
    records = [_validate_one(c) for c in candidates]
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
