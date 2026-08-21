"""Stage 1 — Document Parsing via Unlimited-OCR (baidu/Unlimited-OCR).

Runs the HF transformers model locally on CUDA GPU in gundam mode (per-page),
or remotely via UNLIMITED_OCR_REMOTE_URL (e.g. Colab + Cloudflare quick tunnel).
Produces per-page markdown; emits ParsedDocument only (no product fields).
"""

from __future__ import annotations

import io
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from app.config import settings
from app.models.document import ParsedDocument, ParsedPage

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None


def _notify(
    on_progress,
    step: str,
    message: str,
    percent: int | None = None,
    detail: dict | None = None,
) -> None:
    if on_progress:
        on_progress("parse", step, message, percent=percent, detail=detail)


def _remote_headers() -> dict[str, str]:
    key = settings.unlimited_ocr_remote_api_key.strip()
    return {"X-API-Key": key} if key else {}


def _remote_health_check(base_url: str) -> dict[str, Any]:
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(f"{base_url}/health", headers=_remote_headers())
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        raise RuntimeError(
            f"Colab Unlimited-OCR endpoint unreachable at {base_url}. "
            "Check the notebook is running and UNLIMITED_OCR_REMOTE_URL is up to date "
            "(the tunnel URL changes on every Colab restart)."
        ) from exc


def _infer_page_markdown_remote(image: Image.Image, base_url: str) -> tuple[str, dict[str, Any]]:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    with httpx.Client(timeout=settings.unlimited_ocr_remote_timeout_sec) as client:
        resp = client.post(
            f"{base_url}/infer",
            headers=_remote_headers(),
            files={"file": ("page.jpg", buf.getvalue(), "image/jpeg")},
        )
    resp.raise_for_status()
    body = resp.json()
    if not isinstance(body, dict):
        return "", {}
    return str(body.get("markdown") or ""), body


def _get_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required for parse_channel=unlimited_ocr. "
            "No CUDA device found — install a CUDA-enabled PyTorch build and ensure "
            "an NVIDIA GPU with sufficient VRAM is available. "
            "Or set UNLIMITED_OCR_REMOTE_URL to a Colab Cloudflare-tunnel endpoint."
        )

    from transformers import AutoModel, AutoTokenizer

    gpu_id = max(int(settings.unlimited_ocr_gpu_id), 0)
    torch.cuda.set_device(gpu_id)

    model_name = settings.unlimited_ocr_model
    logger.info("Loading Unlimited-OCR model %s on gpu:%s …", model_name, gpu_id)

    _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    _model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
    )
    _model = _model.eval().cuda()
    logger.info("Unlimited-OCR model loaded on gpu:%s", gpu_id)
    return _model, _tokenizer


def _rasterize(path: Path) -> list[Image.Image]:
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


def _infer_page_markdown(model, tokenizer, image: Image.Image) -> str:
    """Run model.infer on a single page image and return the cleaned markdown."""
    tmp_dir = tempfile.mkdtemp(prefix="uocr_page_")
    try:
        img_path = Path(tmp_dir) / "page.jpg"
        image.save(str(img_path), format="JPEG", quality=92)

        model.infer(
            tokenizer,
            prompt="<image>document parsing.",
            image_file=str(img_path),
            output_path=tmp_dir,
            base_size=settings.unlimited_ocr_base_size,
            image_size=settings.unlimited_ocr_image_size,
            crop_mode=True,
            max_length=settings.unlimited_ocr_max_length,
            no_repeat_ngram_size=settings.unlimited_ocr_no_repeat_ngram_size,
            ngram_window=settings.unlimited_ocr_ngram_window,
            save_results=True,
        )

        result_md_path = Path(tmp_dir) / "result.md"
        if result_md_path.exists():
            return result_md_path.read_text(encoding="utf-8").strip()

        logger.warning("result.md not found after model.infer; returning empty string")
        return ""
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def parse_document_unlimited_ocr(
    source_path: Path,
    source_filename: str | None = None,
    on_progress=None,
) -> ParsedDocument:
    """Stage 1 entrypoint: Unlimited-OCR per-page markdown → ParsedDocument."""
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(path)

    filename = source_filename or path.name
    remote_url = settings.unlimited_ocr_remote_url.strip().rstrip("/")
    model = None
    tokenizer = None

    if remote_url:
        _notify(on_progress, "remote_check", "Checking Colab Unlimited-OCR endpoint…", percent=3)
        health = _remote_health_check(remote_url)
        logger.info(
            "Colab endpoint healthy: device=%s vram=%s%%",
            health.get("device"),
            health.get("vram_pct"),
        )
        _notify(
            on_progress,
            "remote_ready",
            f"Colab endpoint ready (device={health.get('device')}, VRAM {health.get('vram_pct')}%)",
            percent=5,
        )
    else:
        _notify(on_progress, "loading_model", "Loading Unlimited-OCR model (CUDA)…", percent=3)
        model, tokenizer = _get_model()

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
        pct = 12 + int((page_num / max(total, 1)) * 33)
        _notify(
            on_progress,
            "unlimited_ocr_page",
            f"Unlimited-OCR page {page_num}/{total}",
            percent=pct,
            detail={"page": page_num, "total_pages": total},
        )
        if remote_url:
            page_md, meta = _infer_page_markdown_remote(img, remote_url)
            logger.info(
                "Remote OCR page %s/%s device=%s vram=%s%% elapsed=%ss",
                page_num,
                total,
                meta.get("device"),
                meta.get("vram_pct"),
                meta.get("elapsed_sec"),
            )
            if meta.get("device") == "cpu":
                _notify(
                    on_progress,
                    "cpu_offload",
                    f"Page {page_num}/{total} ran on CPU (Colab VRAM {meta.get('vram_pct')}% — slower)",
                    percent=pct,
                    detail=meta,
                )
        else:
            page_md = _infer_page_markdown(model, tokenizer, img)
        pages.append(
            ParsedPage(
                page_number=page_num,
                width=img.width,
                height=img.height,
                blocks=[],
                full_text=page_md,
            )
        )

    all_markdown_parts = []
    for page in pages:
        all_markdown_parts.append(f"--- Page {page.page_number} ---")
        all_markdown_parts.append(page.full_text)
    raw_markdown = "\n".join(all_markdown_parts)

    block_count = 0
    _notify(
        on_progress,
        "parse_done",
        f"Parsed {len(pages)} page(s), {block_count} text block(s) (markdown mode)",
        percent=50,
        detail={"page_count": len(pages), "block_count": block_count},
    )

    meta: dict[str, Any] = {
        "ocr_engine": "unlimited_ocr",
        "parser": "unlimited_ocr",
        "unlimited_ocr_model": settings.unlimited_ocr_model,
    }
    if remote_url:
        meta["unlimited_ocr_remote"] = True
        meta["unlimited_ocr_remote_url"] = remote_url

    structure_notes = [
        f"Primary text source: Unlimited-OCR markdown ({settings.unlimited_ocr_model}).",
        "Docling layout step skipped for Unlimited-OCR channel.",
        "This stage emits document structure only — no product records.",
    ]
    if remote_url:
        structure_notes.insert(1, f"Inference via remote endpoint: {remote_url}.")

    return ParsedDocument(
        source_filename=filename,
        source_path=str(path.resolve()),
        page_count=len(pages),
        pages=pages,
        structure_notes=structure_notes,
        raw_docling_markdown=raw_markdown,
        meta=meta,
    )
