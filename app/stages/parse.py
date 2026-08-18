"""Stage 1 — Document Parsing.

PaddleOCR (primary text + boxes) + Docling (layout/structure).
Emits ParsedDocument only — never product records.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

# Work around PaddlePaddle 3.3.x + oneDNN/PIR crash on CPU (Windows/Linux).
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")

import numpy as np
from PIL import Image

from app.config import settings
from app.models.document import BBox, ParsedDocument, ParsedPage, TextBlock
from app.stages.spatial import group_into_rows

logger = logging.getLogger(__name__)

_ocr_engine = None


def _get_ocr():
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR

        _ocr_engine = PaddleOCR(
            lang=settings.ocr_lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )
    return _ocr_engine


def _rasterize(path: Path) -> list[Image.Image]:
    """Return RGB PIL images for each page."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(path))
        pages: list[Image.Image] = []
        try:
            for i in range(len(pdf)):
                page = pdf[i]
                bitmap = page.render(scale=2.0)
                pages.append(bitmap.to_pil().convert("RGB"))
        finally:
            pdf.close()
        return pages

    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
        return [Image.open(path).convert("RGB")]

    raise ValueError(f"Unsupported file type: {suffix}")


def _bbox_from_poly(poly) -> BBox | None:
    try:
        pts = np.asarray(poly, dtype=float).reshape(-1, 2)
        if pts.size == 0:
            return None
        return BBox(
            x0=float(pts[:, 0].min()),
            y0=float(pts[:, 1].min()),
            x1=float(pts[:, 0].max()),
            y1=float(pts[:, 1].max()),
        )
    except Exception:
        return None


def _as_dict(result_item) -> dict:
    if isinstance(result_item, dict):
        return result_item
    if hasattr(result_item, "json"):
        payload = result_item.json
        if isinstance(payload, dict):
            # PaddleX often nests under "res"
            if "res" in payload and isinstance(payload["res"], dict):
                return payload["res"]
            return payload
    if hasattr(result_item, "keys"):
        try:
            return dict(result_item)
        except Exception:
            pass
    return {}


def _lines_from_predict(result_item) -> list[tuple[list, str, float | None]]:
    """Normalize PaddleOCR 3.x predict output to (poly, text, conf) lines."""
    data = _as_dict(result_item)
    texts = data.get("rec_texts") or data.get("texts") or []
    scores = data.get("rec_scores") or data.get("scores") or []
    polys = data.get("rec_polys") or data.get("dt_polys") or data.get("polys") or []

    lines: list[tuple[list, str, float | None]] = []
    if texts:
        for i, text in enumerate(texts):
            text = (text or "").strip()
            if not text:
                continue
            poly = polys[i] if i < len(polys) else None
            conf = float(scores[i]) if i < len(scores) else None
            lines.append((poly if poly is not None else [], text, conf))
        return lines

    # Legacy 2.x shape: [[poly, (text, conf)], ...]
    if isinstance(result_item, list):
        for line in result_item:
            try:
                poly, (text, conf) = line[0], line[1]
                text = (text or "").strip()
                if text:
                    lines.append((poly, text, float(conf) if conf is not None else None))
            except Exception:
                continue
    return lines


def _ocr_page(image: Image.Image, page_number: int) -> ParsedPage:
    ocr = _get_ocr()
    arr = np.array(image)
    # Prefer predict (3.x); fall back to ocr() for older APIs
    try:
        result = ocr.predict(arr)
    except TypeError:
        result = ocr.ocr(arr)

    page_result = result[0] if result else None
    raw_lines = _lines_from_predict(page_result) if page_result is not None else []

    blocks: list[TextBlock] = []
    for idx, (poly, text, conf) in enumerate(raw_lines):
        block_id = f"p{page_number}-b{idx}"
        has_poly = poly is not None and len(poly) > 0
        blocks.append(
            TextBlock(
                block_id=block_id,
                text=text,
                bbox=_bbox_from_poly(poly) if has_poly else None,
                confidence=conf,
                reading_order=idx,
            )
        )
    blocks = group_into_rows(blocks)
    lines = [block.text for block in sorted(blocks, key=lambda b: b.reading_order)]

    return ParsedPage(
        page_number=page_number,
        width=image.width,
        height=image.height,
        blocks=blocks,
        full_text="\n".join(lines),
    )


def _docling_structure(path: Path) -> tuple[str | None, list[str]]:
    """Run Docling for layout/markdown structure. Failures become notes, not hard stops."""
    notes: list[str] = []
    # Avoid torch inductor needing a C++ compiler on Windows CI/dev machines.
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import PdfFormatOption

        # Prefer embedded PDF text/layout; do not rely on Docling's own OCR
        # because PaddleOCR is the primary text source.
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
        result = converter.convert(str(path))
        markdown = result.document.export_to_markdown()
        notes.append("Docling structure extraction succeeded.")
        try:
            texts = list(result.document.texts) if hasattr(result.document, "texts") else []
            notes.append(f"Docling text items: {len(texts)}")
        except Exception:
            pass
        return markdown, notes
    except Exception as exc:
        logger.warning("Docling parsing failed (OCR text still used): %s", exc)
        notes.append(f"Docling skipped/failed: {exc}")
        return None, notes


def _notify(
    on_progress,
    step: str,
    message: str,
    percent: int | None = None,
    detail: dict | None = None,
) -> None:
    if on_progress:
        on_progress("parse", step, message, percent=percent, detail=detail)


def parse_document(
    source_path: Path,
    source_filename: str | None = None,
    on_progress=None,
) -> ParsedDocument:
    """Stage 1 entrypoint: OCR + structure → ParsedDocument only."""
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(path)

    filename = source_filename or path.name

    _notify(on_progress, "loading_ocr", "Loading PaddleOCR models…", percent=3)
    _get_ocr()  # warm engine early so UI reflects model load

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        _notify(on_progress, "rasterize", "Rasterizing PDF pages…", percent=8)
    else:
        _notify(on_progress, "rasterize", "Loading image…", percent=8)

    raster_pages = _rasterize(path)
    total = len(raster_pages)
    _notify(
        on_progress,
        "rasterize",
        f"Loaded {total} page image(s)",
        percent=12,
        detail={"page_count": total},
    )

    pages: list[ParsedPage] = []
    for i, img in enumerate(raster_pages):
        page_num = i + 1
        # Parse band 12–45 for OCR pages
        pct = 12 + int((page_num / max(total, 1)) * 33)
        _notify(
            on_progress,
            "ocr_page",
            f"OCR page {page_num}/{total}",
            percent=pct,
            detail={"page": page_num, "total_pages": total},
        )
        pages.append(_ocr_page(img, page_num))

    _notify(on_progress, "docling", "Running Docling layout…", percent=48)
    markdown, notes = _docling_structure(path)
    notes.insert(0, "Primary text source: PaddleOCR (boxes + confidence).")
    notes.append("This stage emits document structure only — no product records.")

    block_count = sum(len(p.blocks) for p in pages)
    _notify(
        on_progress,
        "parse_done",
        f"Parsed {len(pages)} page(s), {block_count} text block(s)",
        percent=50,
        detail={"page_count": len(pages), "block_count": block_count},
    )

    return ParsedDocument(
        source_filename=filename,
        source_path=str(path.resolve()),
        page_count=len(pages),
        pages=pages,
        structure_notes=notes,
        raw_docling_markdown=markdown,
        meta={
            "ocr_engine": "paddleocr",
            "parser": "docling",
            "ocr_lang": settings.ocr_lang,
        },
    )
