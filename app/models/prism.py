"""Prism (product-rag) ingestion schema mirror.

Mirrors backend/app/schemas.py in ../product-rag — RawPriceList, RawPlan,
RawProduct — which is the contract accepted by POST /products/raw
(see product-rag/backend/app/api/catalog.py).

If product-rag's schemas evolve, update this mirror to match.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RawPriceList(BaseModel):
    name: str
    country: str
    currency: str
    amount: float
    discount: float | None = None
    discount_strategy: str | None = None


class RawPlan(BaseModel):
    id: str
    name: str
    feature_text: str | None = None
    price_lists: list[RawPriceList] = Field(default_factory=list)
    billing_period: str | None = None
    pricing_type: str | None = None
    interval: str | None = None
    meter: str | None = None


class RawProduct(BaseModel):
    id: str
    name: str
    category: str | None = None
    unit: str | None = None
    kb_article: str | None = None
    brand: str | None = None
    kind: str | None = None
    description: str
    plans: list[RawPlan] = Field(default_factory=list)
    updated_at: str


class PrismExport(BaseModel):
    """Wire type for the Gemini prism-export call."""

    products: list[RawProduct] = Field(default_factory=list)


class PrismIssueSeverity(str, Enum):
    error = "error"
    warning = "warning"


class PrismValidationIssue(BaseModel):
    product_id: str | None = None
    product_name: str | None = None
    field: str
    message: str
    severity: PrismIssueSeverity


class PrismExportResult(BaseModel):
    """Converted + numerically validated prism export for a job."""

    job_id: str
    products: list[RawProduct] = Field(default_factory=list)
    issues: list[PrismValidationIssue] = Field(default_factory=list)
    source_product_count: int = 0
    exported_product_count: int = 0
    valid: bool = True
