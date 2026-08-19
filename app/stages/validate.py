"""Stage 3 — Validation.

Promotes CandidateProduct → ProductRecord with schema + business rules, then
a layout-aware spatial association check, then cross-record semantic checks.
Only this stage produces final product records.
"""

from __future__ import annotations

import re
from collections import defaultdict

from app.models.document import LayoutRegion, ParsedDocument, TextBlock
from app.models.product import (
    CandidateProduct,
    FieldProvenance,
    IssueSeverity,
    ProductRecord,
    ProductStatus,
    ValidationIssue,
    dedupe_issues,
    derive_status,
)
from app.stages import semantics
from app.stages.layout import PageIndex, bbox_to_list, build_page_index, page_diagonal, proximity_confidence

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


def _price_token(value: float | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        text = str(int(value))
    else:
        text = str(value)
    clean = _PRICE_CLEAN_RE.sub("", text.replace(",", ""))
    if not clean:
        return None
    match = _PRICE_TOKEN_RE.search(clean)
    return match.group(0) if match else None


def _find_token_block(token: str, blocks: list[TextBlock]) -> TextBlock | None:
    for block in blocks:
        if token in block.text:
            return block
    return None


def _resolve_anchor_region(
    candidate: CandidateProduct,
    index: PageIndex,
) -> LayoutRegion | None:
    """Find the single candidate region a product's fields must come from.

    Prefers the LLM-declared region_id (validated against the page's actual
    regions); falls back to a geometric match on the product name. Returns
    None (rather than guessing) when the page has no layout data or the name
    resolves to more than one region.
    """
    page = candidate.page
    if page is None:
        return None

    if candidate.region_id:
        region = index.region_lookup.get((page, candidate.region_id))
        if region is not None:
            return region

    name = (candidate.name or "").strip().lower()
    if not name:
        return None
    blocks = index.blocks_by_page.get(page, [])
    matches = [b for b in blocks if b.region_id and name in b.text.lower()]
    region_ids = {b.region_id for b in matches}
    if len(region_ids) == 1:
        region_id = next(iter(region_ids))
        return index.region_lookup.get((page, region_id))
    return None


def _resolve_price_field(
    field_label: str,
    value: float | str | None,
    anchor: LayoutRegion,
    index: PageIndex,
    page: int | None,
) -> tuple[float | str | None, FieldProvenance | None, ValidationIssue | None]:
    """Confirm `value`'s OCR token lives inside the anchor region.

    If the token is found in a *different* region, the value is preserved as
    unresolved (set to None) rather than trusted across a region boundary —
    the caller keeps the flagged issue as evidence. If the token isn't found
    anywhere, the value is left untouched (conservative: OCR/LLM formatting
    can legitimately differ from the raw token).
    """
    if value is None or page is None:
        return value, None, None
    token = _price_token(value)
    if not token:
        return value, None, None

    in_region = index.blocks_by_region.get(anchor.region_id, [])
    hit = _find_token_block(token, in_region)
    if hit is not None:
        provenance = FieldProvenance(
            field=field_label,
            bbox=bbox_to_list(hit.bbox),
            region_id=anchor.region_id,
            block_ids=[hit.block_id],
            confidence=1.0,
        )
        return value, provenance, None

    page_blocks = index.blocks_by_page.get(page, [])
    other_hit = next(
        (b for b in page_blocks if b.region_id and b.region_id != anchor.region_id and token in b.text),
        None,
    )
    if other_hit is None:
        return value, None, None

    diagonal = page_diagonal(index.page_dims, page)
    confidence = proximity_confidence(anchor.bbox, other_hit.bbox, diagonal)
    issue = ValidationIssue(
        code="SPATIAL_ASSOCIATION_AMBIGUOUS",
        field=field_label,
        message=(
            f"{field_label} value {value!r} matches an OCR token found in region "
            f"'{other_hit.region_id}', not this product's own region "
            f"'{anchor.region_id}' (possible cross-column/row-shift contamination). "
            "Treating as unresolved instead of assigning it from a neighboring region."
        ),
        product_bbox=bbox_to_list(anchor.bbox),
        field_bbox=bbox_to_list(other_hit.bbox),
        region_id=anchor.region_id,
        confidence=confidence,
        severity=IssueSeverity.review,
    )
    return None, None, issue


def _resolve_spatial_fields(
    candidate: CandidateProduct,
    price: float | None,
    price_variants: list[dict] | None,
    index: PageIndex,
) -> tuple[float | None, list[dict] | None, str | None, list[FieldProvenance], list[ValidationIssue]]:
    provenance: list[FieldProvenance] = []
    issues: list[ValidationIssue] = []

    anchor = _resolve_anchor_region(candidate, index)
    if anchor is None:
        return price, price_variants, None, provenance, issues

    price, price_provenance, price_issue = _resolve_price_field(
        "price", price, anchor, index, candidate.page
    )
    if price_provenance:
        provenance.append(price_provenance)
    if price_issue:
        issues.append(price_issue)

    default_mirror = None
    if price_issue is None and price_provenance is not None:
        default_mirror = (price, price_provenance.bbox, price_provenance.region_id)

    for idx, variant in enumerate(price_variants or []):
        raw_price = variant.get("price")
        if raw_price is None:
            continue
        label = variant.get("attribute_name") or variant.get("label") or f"#{idx}"
        field_label = f"price_variants[{label}].price"

        if variant.get("is_default") and default_mirror is not None and raw_price == default_mirror[0]:
            # Mirrors the already-resolved top-level price/variant — do not
            # re-run resolution (and do not duplicate the issue) for it.
            provenance.append(
                FieldProvenance(
                    field=field_label,
                    bbox=default_mirror[1],
                    region_id=default_mirror[2],
                    block_ids=[],
                    confidence=1.0,
                )
            )
            continue

        resolved_price, v_provenance, v_issue = _resolve_price_field(
            field_label, raw_price, anchor, index, candidate.page
        )
        variant["price"] = resolved_price
        if v_provenance:
            provenance.append(v_provenance)
        if v_issue:
            issues.append(v_issue)

    return price, price_variants, anchor.region_id, provenance, issues


def _validate_one(candidate: CandidateProduct, index: PageIndex) -> ProductRecord:
    error_issues: list[ValidationIssue] = []
    warnings: list[str] = []

    name = (candidate.name or "").strip()
    if not name:
        error_issues.append(
            ValidationIssue(
                code="MISSING_REQUIRED_FIELD",
                field="name",
                message="Missing required field: name",
                severity=IssueSeverity.error,
            )
        )

    price, price_errors = _normalize_price(candidate.price)
    for msg in price_errors:
        error_issues.append(
            ValidationIssue(code="UNPARSEABLE_PRICE", field="price", message=msg, severity=IssueSeverity.error)
        )

    price_variants, variant_errors = _normalize_price_variants(candidate)
    for msg in variant_errors:
        error_issues.append(
            ValidationIssue(
                code="UNPARSEABLE_VARIANT_PRICE",
                field="price_variants",
                message=msg,
                severity=IssueSeverity.error,
            )
        )
    for variant in price_variants or []:
        warnings.extend(variant.get("warnings", []))

    currency, currency_warnings = _normalize_currency(candidate.currency)
    warnings.extend(currency_warnings)

    confidence = candidate.confidence
    if confidence is not None and not (0.0 <= confidence <= 1.0):
        warnings.append(f"Confidence out of range, clamped: {confidence}")
        confidence = max(0.0, min(1.0, float(confidence)))

    price, price_variants, region_id, provenance, spatial_issues = _resolve_spatial_fields(
        candidate, price, price_variants, index
    )
    review_issues = [i for i in dedupe_issues(spatial_issues) if i.severity == IssueSeverity.review]

    # Empty shell: no name and nothing else useful.
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
        error_issues.append(
            ValidationIssue(code="EMPTY_PRODUCT_SHELL", message="Empty product shell", severity=IssueSeverity.error)
        )

    schema_valid = not error_issues
    all_issues = error_issues + review_issues
    status, extraction_valid = derive_status(schema_valid, all_issues)

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
        region_id=region_id,
        provenance=provenance,
        schema_valid=schema_valid,
        extraction_valid=extraction_valid,
        status=status,
        validation_errors=all_issues,
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


def _enforce_status_invariant(record: ProductRecord) -> None:
    """Defensive final guard: a record can never be `validated` while it
    still carries an unresolved (review-severity) or hard (error-severity)
    issue. Recomputes from validation_errors and corrects status if a bug
    upstream ever leaves them inconsistent."""
    schema_valid = not any(i.severity == IssueSeverity.error for i in record.validation_errors)
    expected_status, expected_extraction_valid = derive_status(schema_valid, record.validation_errors)
    if record.status != expected_status:
        record.validation_errors.append(
            ValidationIssue(
                code="STATUS_INVARIANT_VIOLATION",
                message=(
                    f"Status '{record.status.value}' was inconsistent with outstanding "
                    f"validation issues; corrected to '{expected_status.value}'."
                ),
                severity=IssueSeverity.review,
            )
        )
    record.schema_valid = schema_valid
    record.extraction_valid = expected_extraction_valid
    record.status = expected_status


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

    index = build_page_index(parsed)
    records = [_validate_one(c, index) for c in candidates]
    _flag_duplicates(records)
    semantics.apply_semantic_checks(records, index)
    for record in records:
        _enforce_status_invariant(record)

    validated = sum(1 for r in records if r.status == ProductStatus.validated)
    needs_review = sum(1 for r in records if r.status == ProductStatus.needs_review)
    invalid = sum(1 for r in records if r.status == ProductStatus.invalid)
    _notify(
        "validate_done",
        f"{validated} validated, {needs_review} needs review, {invalid} invalid",
        percent=95,
        detail={"validated": validated, "needs_review": needs_review, "invalid": invalid},
    )
    return records
