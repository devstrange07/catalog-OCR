"""Stage package."""

from app.stages.crawl import fetch_url, parsed_document_from_markdown
from app.stages.extract import extract_products
from app.stages.parse import parse_document
from app.stages.parse_gemini import parse_document_gemini
from app.stages.validate import validate_products

__all__ = [
    "parse_document",
    "parse_document_gemini",
    "extract_products",
    "validate_products",
    "fetch_url",
    "parsed_document_from_markdown",
]
