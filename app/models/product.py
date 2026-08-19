"""Product models — candidates (Stage 2) and records (Stage 3)."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ProductStatus(str, Enum):
    validated = "validated"
    rejected = "rejected"


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
    status: ProductStatus
    validation_errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SubmitProductsArgs(BaseModel):
    """Schema for the enforced submit_products tool arguments."""

    products: list[CandidateProduct] = Field(default_factory=list)
