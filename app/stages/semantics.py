"""Cross-record semantic integrity checks.

These run after per-candidate spatial resolution (`app.stages.validate`) and
look for signs of contamination that only show up when comparing records to
each other: a promo code or quantity phrase that actually belongs to a
neighboring card, two records citing the same OCR source, a "was"/MRP price
that doesn't make sense against the offer price, or a title that folds in a
separately laid-out size/attribute label.

Nothing here is hard-coded to any product name, brand, or price — every
check is driven by generic patterns (regex shape, region ownership,
provenance bbox equality, or block-concatenation geometry) so it applies
equally to flower catalogs, restaurant menus, or electronics flyers.
"""

from __future__ import annotations

import re
from collections import defaultdict

from app.models.product import IssueSeverity, ProductRecord, ValidationIssue, dedupe_issues
from app.stages.layout import PageIndex, bbox_to_list

_WAS_MRP_RE = re.compile(r"\b(?:was|mrp)\b[^0-9]{0,12}(\d[\d,]*\.?\d*)", re.IGNORECASE)
_PROMO_CODE_RE = re.compile(r"\b[A-Z]{3,}\d{1,4}\b")
_QUANTITY_RE = re.compile(
    r"\b\d{1,4}\s*(?:stems?|pcs?|pieces?|count|pack|flowers?|units?|ml|g|kg|oz)\b", re.IGNORECASE
)


def _region_bbox(index: PageIndex, page: int | None, region_id: str | None) -> list[float] | None:
    if page is None or not region_id:
        return None
    region = index.region_lookup.get((page, region_id))
    return bbox_to_list(region.bbox) if region else None


def _check_mrp_vs_price(record: ProductRecord) -> None:
    """A "was"/MRP price at or below the offer price is a contradiction —
    usually a sign the offer price actually belongs to a different card."""
    candidates: list[tuple[str, float | None]] = [(record.description or "", record.price)]
    for variant in record.price_variants or []:
        candidates.append((variant.details or "", variant.price if variant.price is not None else record.price))

    for text, compare_price in candidates:
        if not text or compare_price is None:
            continue
        match = _WAS_MRP_RE.search(text)
        if not match:
            continue
        try:
            was_value = float(match.group(1).replace(",", ""))
        except ValueError:
            continue
        if was_value <= compare_price:
            record.validation_errors.append(
                ValidationIssue(
                    code="MRP_NOT_ABOVE_PRICE",
                    field="price",
                    message=(
                        f"'Was'/MRP price {was_value} is not greater than the offer price "
                        f"{compare_price}; this contradicts a discount and suggests the offer "
                        "price may have been sourced from a different card."
                    ),
                    region_id=record.region_id,
                    confidence=0.6,
                    severity=IssueSeverity.review,
                )
            )
            return


def _check_shared_source_bbox(records: list[ProductRecord]) -> None:
    """Two records resolving a field to the exact same OCR bbox is a strong
    signal of a mis-split card or duplicate extraction."""
    owners: dict[tuple[str, tuple[float, ...]], list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        for prov in record.provenance:
            if not prov.bbox:
                continue
            owners[(prov.field, tuple(prov.bbox))].append(i)

    for (field_name, bbox), idxs in owners.items():
        unique_idxs = sorted(set(idxs))
        if len(unique_idxs) <= 1:
            continue
        for i in unique_idxs:
            record = records[i]
            record.validation_errors.append(
                ValidationIssue(
                    code="SHARED_SOURCE_BBOX",
                    field=field_name,
                    message=(
                        f"Field '{field_name}' resolves to the same OCR source as "
                        f"{len(unique_idxs) - 1} other record(s) on this page."
                    ),
                    field_bbox=list(bbox),
                    region_id=record.region_id,
                    confidence=0.5,
                    severity=IssueSeverity.review,
                )
            )


def _check_name_includes_separate_field(record: ProductRecord, index: PageIndex) -> None:
    """If the record's name is only formed by concatenating two adjacent
    blocks in its own region — one much shorter than the other — the layout
    likely represents the short one (e.g. a size/attribute badge) as its own
    field, not part of the title. Purely geometric: no word list involved."""
    if not record.region_id or not record.name:
        return
    blocks = sorted(
        (b for b in index.blocks_by_region.get(record.region_id, []) if b.bbox is not None),
        key=lambda b: (b.bbox.y0, b.bbox.x0),
    )
    if len(blocks) < 2:
        return
    name_norm = " ".join(record.name.split()).lower()

    for i in range(len(blocks)):
        for j in range(i + 2, min(i + 4, len(blocks)) + 1):
            combo = " ".join(b.text for b in blocks[i:j])
            if " ".join(combo.split()).lower() != name_norm:
                continue
            first_block = blocks[i]
            rest_text = " ".join(b.text for b in blocks[i + 1 : j])
            first_len = len(first_block.text.strip())
            rest_len = len(rest_text.strip())
            if not first_len or not rest_len:
                continue
            if first_len <= max(3, rest_len * 0.4) and len(first_block.text.split()) <= 2:
                record.validation_errors.append(
                    ValidationIssue(
                        code="NAME_INCLUDES_SEPARATE_FIELD",
                        field="name",
                        message=(
                            f"Name '{record.name}' concatenates a short separate block "
                            f"('{first_block.text.strip()}') with the title block; the layout "
                            "represents them as distinct fields (e.g. size/attribute vs. title)."
                        ),
                        field_bbox=bbox_to_list(first_block.bbox),
                        region_id=record.region_id,
                        confidence=0.6,
                        severity=IssueSeverity.review,
                    )
                )
                return


def _classify_clause(clause: str) -> str | None:
    if _PROMO_CODE_RE.search(clause):
        return "PROMO_CODE_OUT_OF_REGION"
    if _QUANTITY_RE.search(clause):
        return "QUANTITY_OUT_OF_REGION"
    if len(clause) >= 6:
        return "ATTRIBUTE_OUT_OF_REGION"
    return None


def _find_owning_region(index: PageIndex, page: int | None, needle: str) -> str | None:
    if page is None or not needle:
        return None
    for block in index.blocks_by_page.get(page, []):
        if block.region_id and needle in block.text.lower():
            return block.region_id
    return None


def _check_out_of_region_phrases(
    record: ProductRecord, region_owner: dict[str, ProductRecord], index: PageIndex
) -> None:
    """Split description text into clauses; if a clause (promo code,
    quantity phrase, or other descriptive attribute) actually lives in an
    OCR region owned by a *different* known product, flag contamination."""
    if not record.description or not record.region_id:
        return

    seen: set[str] = set()
    for clause in re.split(r"[,.;]\s*", record.description):
        clause = clause.strip()
        if not clause:
            continue
        code = _classify_clause(clause)
        if not code:
            continue
        owning_region = _find_owning_region(index, record.page, clause.lower())
        if not owning_region or owning_region == record.region_id:
            continue
        other_record = region_owner.get(owning_region)
        if other_record is None or other_record is record:
            continue
        dedupe_key = f"{code}:{owning_region}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        record.validation_errors.append(
            ValidationIssue(
                code=code,
                field="description",
                message=(
                    f"Description phrase '{clause}' matches OCR text in region "
                    f"'{owning_region}', which belongs to another product on this page — "
                    "likely cross-record contamination rather than this product's own attribute."
                ),
                product_bbox=_region_bbox(index, record.page, record.region_id),
                field_bbox=_region_bbox(index, record.page, owning_region),
                region_id=record.region_id,
                confidence=0.5,
                severity=IssueSeverity.review,
            )
        )


def apply_semantic_checks(records: list[ProductRecord], index: PageIndex) -> None:
    """Mutates `records` in place, appending structured ValidationIssue
    entries for cross-record contamination. Status is not recomputed here —
    the caller (validate_products) re-derives status/extraction_valid from
    the updated validation_errors afterward."""
    if not records:
        return

    region_owner: dict[str, ProductRecord] = {r.region_id: r for r in records if r.region_id}

    for record in records:
        _check_mrp_vs_price(record)
        _check_name_includes_separate_field(record, index)
        _check_out_of_region_phrases(record, region_owner, index)

    _check_shared_source_bbox(records)

    for record in records:
        record.validation_errors = dedupe_issues(record.validation_errors)
