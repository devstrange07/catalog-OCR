"""DeepSeek Files API transport — raw multipart upload, no base64 conversion."""

from __future__ import annotations

import logging
from pathlib import Path

from openai import OpenAI

from app.config import settings
from app.file_types import DEEPSEEK_VISION_IMAGE_SUFFIXES

logger = logging.getLogger(__name__)


def client() -> OpenAI:
    if not settings.deepseek_api_key.strip():
        raise RuntimeError(
            "DEEPSEEK_API_KEY is missing. Set it in .env for parse_channel=deepseek_direct."
        )
    return OpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
    )


def upload_image_file(oa_client: OpenAI, path: Path) -> str:
    """Upload raw image bytes via Files API; return file_id (file-api-...)."""
    suffix = path.suffix.lower()
    if suffix not in DEEPSEEK_VISION_IMAGE_SUFFIXES:
        raise ValueError(
            f"Unsupported file type '{suffix}' for DeepSeek Vision. "
            f"Allowed: {sorted(DEEPSEEK_VISION_IMAGE_SUFFIXES)}"
        )
    with path.open("rb") as f:
        uploaded = oa_client.files.create(file=f, purpose="user_data")
    file_id = getattr(uploaded, "id", None)
    if not file_id:
        raise RuntimeError("DeepSeek Files upload returned no file_id.")
    return str(file_id)


def delete_file(oa_client: OpenAI, file_id: str) -> None:
    """Best-effort delete of an uploaded Files API object."""
    if not file_id:
        return
    try:
        oa_client.files.delete(file_id)
    except Exception:
        logger.debug("Best-effort DeepSeek file delete failed for %s", file_id, exc_info=True)
