"""Shared Gemini file transport (inline vs Files API) and generate-response helpers."""

from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from app.config import settings
from app.observability.langsmith import trace_span
from app.file_types import mime_for_suffix

logger = logging.getLogger(__name__)

GENERATE_TIMEOUT = 180.0
FILES_POLL_TIMEOUT = 120.0
FILES_POLL_INTERVAL = 1.5

_FAILED_FINISH_REASONS = frozenset(
    {
        "MAX_TOKENS",
        "SAFETY",
        "RECITATION",
        "LANGUAGE",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "MALFORMED_FUNCTION_CALL",
        "OTHER",
    }
)


def mime_for_path(path: Path) -> str:
    return mime_for_suffix(path.suffix.lower())


def generate_content_url() -> str:
    return (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.gemini_model}:generateContent"
    )


def _notify(
    on_progress,
    stage: str,
    step: str,
    message: str,
    percent: int | None = None,
    detail: dict | None = None,
) -> None:
    if on_progress:
        on_progress(stage, step, message, percent=percent, detail=detail)


def upload_to_gemini_files(
    client: httpx.Client,
    path: Path,
    mime: str,
    data: bytes,
) -> str:
    """Upload via Files API; return file URI. Polls until ACTIVE."""
    key = settings.gemini_api_key
    meta = {"file": {"display_name": path.name}}
    with trace_span(
        "gemini.files.upload",
        run_type="tool",
        metadata={"filename": path.name, "mime": mime, "bytes": len(data)},
        tags=["gemini", "files_api"],
    ):
        resp = client.post(
            "https://generativelanguage.googleapis.com/upload/v1beta/files",
            params={"key": key},
            headers={"X-Goog-Upload-Protocol": "multipart"},
            files={
                "metadata": ("metadata.json", json.dumps(meta), "application/json"),
                "file": (path.name, data, mime),
            },
        )
    if resp.status_code >= 400:
        raise RuntimeError(f"Gemini Files upload failed ({resp.status_code}): {resp.text[:300]}")

    body = resp.json()
    file_obj = body.get("file") if isinstance(body.get("file"), dict) else body
    name = file_obj.get("name")
    uri = file_obj.get("uri")
    state = (file_obj.get("state") or "").upper()
    if not name:
        raise RuntimeError("Gemini Files upload returned no file name.")

    deadline = time.monotonic() + FILES_POLL_TIMEOUT
    while state in {"", "PROCESSING"}:
        if time.monotonic() > deadline:
            raise RuntimeError(f"Gemini Files upload timed out waiting for ACTIVE (state={state!r}).")
        time.sleep(FILES_POLL_INTERVAL)
        poll = client.get(
            f"https://generativelanguage.googleapis.com/v1beta/{name}",
            params={"key": key},
        )
        if poll.status_code >= 400:
            raise RuntimeError(f"Gemini Files poll failed ({poll.status_code}): {poll.text[:300]}")
        file_obj = poll.json()
        state = (file_obj.get("state") or "").upper()
        uri = file_obj.get("uri") or uri

    if state == "FAILED":
        raise RuntimeError("Gemini Files upload entered FAILED state.")
    if state != "ACTIVE" or not uri:
        raise RuntimeError(f"Gemini Files upload not usable (state={state!r}, uri={uri!r}).")
    return uri


def delete_gemini_file(client: httpx.Client, file_uri: str) -> None:
    """Best-effort delete of an uploaded Files API object."""
    name = file_uri.rstrip("/").split("/")[-1]
    if not name:
        return
    try:
        with trace_span(
            "gemini.files.delete",
            run_type="tool",
            metadata={"file_name": name},
            tags=["gemini", "files_api"],
        ):
            client.delete(
                f"https://generativelanguage.googleapis.com/v1beta/files/{name}",
                params={"key": settings.gemini_api_key},
            )
    except Exception:
        logger.debug("Best-effort Gemini file delete failed for %s", name, exc_info=True)


def build_file_part(
    client: httpx.Client,
    path: Path,
    *,
    on_progress=None,
    progress_stage: str = "parse",
) -> tuple[dict[str, Any], str, str | None, str, int]:
    """Return (file_part, transport, uploaded_file_uri_or_none, mime, size_bytes)."""
    mime = mime_for_path(path)
    data = path.read_bytes()
    size = len(data)

    if size <= settings.gemini_inline_max_bytes:
        with trace_span(
            "gemini.file_part.inline",
            run_type="tool",
            metadata={"mime": mime, "bytes": size, "threshold": settings.gemini_inline_max_bytes},
            tags=["gemini", "transport", "inline"],
        ):
            pass
        _notify(
            on_progress,
            progress_stage,
            "prepare_file",
            f"Sending document inline ({size} bytes)…",
            percent=12,
            detail={"transport": "inline", "bytes": size, "mime": mime},
        )
        part = {
            "inline_data": {
                "mime_type": mime,
                "data": base64.b64encode(data).decode("ascii"),
            }
        }
        return part, "inline", None, mime, size

    _notify(
        on_progress,
        progress_stage,
        "upload_file",
        f"Uploading document via Files API ({size} bytes)…",
        percent=10,
        detail={"transport": "files_api", "bytes": size, "mime": mime},
    )
    uri = upload_to_gemini_files(client, path, mime, data)
    _notify(
        on_progress,
        progress_stage,
        "upload_file",
        "Files API upload ready",
        percent=18,
        detail={"transport": "files_api", "file_uri": uri},
    )
    part = {"file_data": {"mime_type": mime, "file_uri": uri}}
    return part, "files_api", uri, mime, size


def extract_json_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini returned no candidates.")
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
    text = "\n".join(t for t in texts if t).strip()
    if not text:
        raise RuntimeError("Gemini returned empty content.")
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    return text


def assert_generate_ok(payload: dict[str, Any]) -> None:
    """Fail on safety blocks or truncated generation. Missing finishReason is allowed."""
    feedback = payload.get("promptFeedback") or {}
    block_reason = feedback.get("blockReason")
    candidates = payload.get("candidates") or []
    finish = ""
    if candidates:
        finish = str(candidates[0].get("finishReason") or "").upper()
    with trace_span(
        "gemini.generate.assert",
        run_type="tool",
        metadata={"finish_reason": finish or None, "block_reason": block_reason},
        tags=["gemini", "generate"],
    ):
        if block_reason:
            raise RuntimeError(f"Gemini blocked the prompt ({block_reason}).")
        if not candidates:
            return
        if finish in _FAILED_FINISH_REASONS:
            if finish == "MAX_TOKENS":
                raise RuntimeError(
                    f"Gemini generation finished unsuccessfully (finishReason=MAX_TOKENS). "
                    f"Raise EXTRACT_MAX_TOKENS (current={settings.extract_max_tokens})."
                )
            raise RuntimeError(f"Gemini generation finished unsuccessfully (finishReason={finish}).")
