"""Product models — candidates (Stage 2) and records (Stage 3).

Domain-agnostic schema for retail catalogues, SaaS pricing, B2B parts lists,
restaurant menus, healthcare rate cards, real-estate listings, and similar sources.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ProductStatus(str, Enum):
    """Pipeline extraction quality — not market availability."""

    validated = "validated"
    rejected = "rejected"


class ItemType(str, Enum):
    product = "product"
    service = "service"
    bundle = "bundle"
    plan = "plan"
    event = "event"
    addon = "addon"
    listing = "listing"
    announcement = "announcement"
    other = "other"


class Availability(str, Enum):
    """Real-world market availability of the item."""

    available = "available"
    out_of_stock = "out_of_stock"
    pre_order = "pre_order"
    coming_soon = "coming_soon"
    discontinued = "discontinued"
    on_request = "on_request"
    waitlist = "waitlist"
    limited = "limited"
    unknown = "unknown"


class Discount(BaseModel):
    """Absolute or relative savings vs list/original price."""

    amount: float | None = None
    percent: float | None = None
    description: str | None = None


class Promotion(BaseModel):
    """Active promo/offer on this item (cross-domain)."""

    type: str | None = None
    label: str | None = None
    details: str | None = None
    conditions: str | None = None


class PriceTier(BaseModel):
    """Different price points for the SAME item configuration.

    Examples: MRP vs sale, monthly vs annual, volume bands, rack vs online rate.
    """

    label: str
    price: float | str | None = None
    details: str | None = None
    is_current: bool = True
    currency: str | None = None
    billing_cycle: str | None = None
    warnings: list[str] = Field(default_factory=list)


class Variant(BaseModel):
    """Genuinely different configurations of the same item.

    Examples: size/flavor, plan tier, portion, room type.
    """

    attribute: str
    label: str
    price: float | str | None = None
    currency: str | None = None
    unit: str | None = None
    pack_size: str | None = None
    sku: str | None = None
    description: str | None = None
    available: bool | str | None = True
    is_default: bool = False
    warnings: list[str] = Field(default_factory=list)


class CandidateProduct(BaseModel):
    """LLM-proposed product via tool call. Not a final record."""

    name: str = ""
    brand: str | None = None
    sku: str | None = None
    product_code: str | None = None
    item_type: ItemType | str | None = None
    category: str | None = None
    project_name: str | None = None
    location: str | None = None

    price: float | str | None = None
    list_price: float | str | None = None
    currency: str | None = None
    price_qualifier: str | None = None
    billing_cycle: str | None = None
    discount: Discount | None = None

    unit: str | None = None
    pack_size: str | None = None
    quantity: int | None = None

    promotion: Promotion | None = None
    validity: str | None = None
    includes: list[str] | None = None
    description: str | None = None
    conditions: list[str] | None = None
    minimum_order: str | None = None
    tags: list[str] | None = None
    availability: Availability | str | None = None

    page: int | None = None
    source_snippet: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    price_tiers: list[PriceTier] | None = None
    variants: list[Variant] | None = None


class ProductRecord(BaseModel):
    """Final product after Stage 3 validation."""

    name: str
    brand: str | None = None
    sku: str | None = None
    product_code: str | None = None
    item_type: ItemType = ItemType.product
    category: str | None = None
    project_name: str | None = None
    location: str | None = None

    price: float | None = None
    list_price: float | None = None
    currency: str | None = None
    price_qualifier: str | None = None
    billing_cycle: str | None = None
    discount: Discount | None = None

    unit: str | None = None
    pack_size: str | None = None
    quantity: int | None = None

    promotion: Promotion | None = None
    validity: str | None = None
    includes: list[str] | None = None
    description: str | None = None
    conditions: list[str] | None = None
    minimum_order: str | None = None
    tags: list[str] | None = None
    availability: Availability = Availability.available

    page: int | None = None
    source_snippet: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    price_tiers: list[PriceTier] | None = None
    variants: list[Variant] | None = None

    status: ProductStatus
    validation_errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SubmitProductsArgs(BaseModel):
    """Schema for the enforced submit_products tool arguments."""

    products: list[CandidateProduct] = Field(default_factory=list)
