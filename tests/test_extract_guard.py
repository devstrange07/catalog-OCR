"""Ensure Stage 2 refuses to invent products without a Gemini API key."""

from app.config import settings
from app.models.document import ParsedDocument, ParsedPage, TextBlock
from app.stages.extract_gemini import extract_products_gemini


def test_extract_gemini_requires_api_key(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "")
    doc = ParsedDocument(
        source_filename="x.png",
        source_path="x.png",
        page_count=1,
        pages=[
            ParsedPage(
                page_number=1,
                blocks=[TextBlock(block_id="p1-b0", text="Widget 9.99", reading_order=0)],
                full_text="Widget 9.99",
            )
        ],
    )
    try:
        extract_products_gemini(doc)
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "GEMINI_API_KEY" in str(exc)
