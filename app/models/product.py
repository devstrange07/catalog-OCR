"""Product models — candidates (Stage 2) and records (Stage 3)."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ProductStatus(str, Enum):
    validated = "validated"
    needs_review = "needs_review"
    invalid = "invalid"


class IssueSeverity(str, Enum):
    error = "error"
    review = "review"


class ValidationIssue(BaseModel):
    """A structured, evidence-carrying validation finding.

    `severity="error"` means the record fails basic schema/business rules
    (`schema_valid=False`, status becomes `invalid`). `severity="review"`
    means the record is schema-valid but a field's spatial/semantic
    attribution is unresolved or contradicted (`extraction_valid=False`,
    status becomes `needs_review`).
    """

    code: str
    field: str | None = None
    message: str
    product_bbox: list[float] | None = None
    field_bbox: list[float] | None = None
    region_id: str | None = None
    confidence: float | None = None
    severity: IssueSeverity = IssueSeverity.review


class FieldProvenance(BaseModel):
    """Source bounding box / region for one extracted field, for audit."""

    field: str
    bbox: list[float] | None = None
    region_id: str | None = None
    block_ids: list[str] = Field(default_factory=list)
    confidence: float | None = None


class PriceVariant(BaseModel):
    label: str | None = None
    price: float | str | None = None
    details: str | None = None
    attribute_name: str | None = None
    sku: str | None = None
    currency: str | None = None
    unit: str | None = None
    pack_size: str | None = None
    available: bool | str | None = None
    is_default: bool = False
    warnings: list[str] = Field(default_factory=list)


class CandidateProduct(BaseModel):
    """LLM-proposed product via tool call. Not a final record."""

    name: str = ""
    brand: str | None = None
    sku: str | None = None
    product_code: str | None = None
    price: float | str | None = None
    currency: str | None = None
    unit: str | None = None
    pack_size: str | None = None
    description: str | None = None
    page: int | None = None
    source_snippet: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    price_variants: list[PriceVariant] | None = None
    region_id: str | None = None
    source_block_ids: list[str] | None = None


class ProductRecord(BaseModel):
    """Final product after Stage 3 validation."""

    name: str
    brand: str | None = None
    sku: str | None = None
    product_code: str | None = None
    price: float | None = None
    currency: str | None = None
    unit: str | None = None
    pack_size: str | None = None
    description: str | None = None
    page: int | None = None
    source_snippet: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    price_variants: list[PriceVariant] | None = None
    region_id: str | None = None
    provenance: list[FieldProvenance] = Field(default_factory=list)
    schema_valid: bool = True
    extraction_valid: bool = True
    status: ProductStatus
    validation_errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SubmitProductsArgs(BaseModel):
    """Schema for the enforced submit_products tool arguments."""

    products: list[CandidateProduct] = Field(default_factory=list)


def derive_status(schema_valid: bool, issues: list[ValidationIssue]) -> tuple[ProductStatus, bool]:
    """Derive (status, extraction_valid) from schema_valid + accumulated issues.

    A record is never `validated` merely because it is schema-valid — any
    outstanding review-severity issue (unresolved spatial/semantic
    association) forces `needs_review` instead.
    """
    if not schema_valid:
        return ProductStatus.invalid, False
    if any(issue.severity == IssueSeverity.review for issue in issues):
        return ProductStatus.needs_review, False
    return ProductStatus.validated, True


def dedupe_issues(issues: list[ValidationIssue]) -> list[ValidationIssue]:
    """Collapse equivalent review-severity issues (e.g. a variant-price issue
    that mirrors the product-level price issue) to one, keeping the first
    occurrence's evidence. Error-severity issues are never merged, since two
    distinct hard failures can legitimately share a (code, region, bbox) key
    of (code, None, None)."""
    errors = [i for i in issues if i.severity == IssueSeverity.error]
    reviews = [i for i in issues if i.severity == IssueSeverity.review]
    seen: set[tuple] = set()
    deduped: list[ValidationIssue] = []
    for issue in reviews:
        key = (issue.code, issue.region_id, tuple(issue.field_bbox) if issue.field_bbox else None)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(issue)
    return errors + deduped
