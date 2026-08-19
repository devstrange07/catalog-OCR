"""Unit tests for app.stages.semantics cross-record integrity checks."""

from __future__ import annotations

from app.models.product import FieldProvenance, ProductRecord, ProductStatus
from app.stages import semantics
from app.stages.layout import PageIndex


def _record(**kwargs) -> ProductRecord:
    defaults = dict(name="X", status=ProductStatus.validated)
    defaults.update(kwargs)
    return ProductRecord(**defaults)


def test_mrp_not_above_price_flags_contradiction():
    record = _record(price=999, description="Limited time, was 799")
    semantics.apply_semantic_checks([record], PageIndex())
    assert any(i.code == "MRP_NOT_ABOVE_PRICE" for i in record.validation_errors)


def test_mrp_above_price_is_not_flagged():
    record = _record(price=799, description="Limited time, was 999")
    semantics.apply_semantic_checks([record], PageIndex())
    assert not any(i.code == "MRP_NOT_ABOVE_PRICE" for i in record.validation_errors)


def test_shared_source_bbox_flags_both_records():
    bbox = [10.0, 10.0, 50.0, 30.0]
    a = _record(name="A", provenance=[FieldProvenance(field="price", bbox=bbox)])
    b = _record(name="B", provenance=[FieldProvenance(field="price", bbox=bbox)])
    semantics.apply_semantic_checks([a, b], PageIndex())
    assert any(i.code == "SHARED_SOURCE_BBOX" for i in a.validation_errors)
    assert any(i.code == "SHARED_SOURCE_BBOX" for i in b.validation_errors)


def test_shared_source_bbox_not_flagged_for_distinct_bboxes():
    a = _record(name="A", provenance=[FieldProvenance(field="price", bbox=[0, 0, 10, 10])])
    b = _record(name="B", provenance=[FieldProvenance(field="price", bbox=[20, 20, 30, 30])])
    semantics.apply_semantic_checks([a, b], PageIndex())
    assert not any(i.code == "SHARED_SOURCE_BBOX" for i in a.validation_errors)
    assert not any(i.code == "SHARED_SOURCE_BBOX" for i in b.validation_errors)


def test_apply_semantic_checks_noop_on_empty_records():
    semantics.apply_semantic_checks([], PageIndex())  # must not raise
