"""URL fetch stage — hybrid binary download vs crawl4ai HTML.

Single URL only. Internal links are collected for a future deep-crawl pass;
they are never followed here.
"""

from __future__ import annotations

import logging
import mimetypes
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Callable, Literal
from urllib.parse import unquote, urlparse

import httpx

from app.config import settings
from app.file_types import GEMINI_TEXT_SUFFIXES, IMAGE_PDF_SUFFIXES, mime_for_suffix
from app.models.document import ParsedDocument, ParsedPage, TextBlock

logger = logging.getLogger(__name__)

BINARY_SUFFIXES = IMAGE_PDF_SUFFIXES | GEMINI_TEXT_SUFFIXES

SUFFIX_TO_CONTENT_TYPE = {
    suffix: mime_for_suffix(suffix) for suffix in BINARY_SUFFIXES
}

ProgressCallback = Callable[..., None]


class CrawlError(Exception):
    """Raised when URL validation or fetch fails."""


@dataclass
class BinaryFetch:
    kind: Literal["binary"] = "binary"
    data: bytes = b""
    filename: str = "download.bin"
    content_type: str = "application/octet-stream"
    source_url: str = ""
    final_url: str = ""


@dataclass
class HtmlFetch:
    kind: Literal["html"] = "html"
    markdown: str = ""
    title: str | None = None
    source_url: str = ""
    final_url: str = ""
    links_internal: list[str] = field(default_factory=list)
    content_type: str = "text/html"


FetchResult = BinaryFetch | HtmlFetch


def normalize_url(url: str) -> str:
    cleaned = (url or "").strip()
    if not cleaned:
        raise CrawlError("URL is required")
    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"}:
        raise CrawlError("URL must start with http:// or https://")
    if not parsed.netloc:
        raise CrawlError("URL is missing a host")
    return cleaned


def _path_suffix(url: str) -> str:
    path = unquote(urlparse(url).path or "")
    return PurePosixPath(path).suffix.lower()


def is_binary_suffix(url: str) -> bool:
    return _path_suffix(url) in BINARY_SUFFIXES


def is_binary_content_type(content_type: str | None) -> bool:
    if not content_type:
        return False
    main = content_type.split(";")[0].strip().lower()
    if main == "application/pdf":
        return True
    if main.startswith("image/"):
        return True
    return False


def _filename_from_url(url: str, content_type: str | None = None) -> str:
    path = unquote(urlparse(url).path or "")
    name = PurePosixPath(path).name
    if name and PurePosixPath(name).suffix:
        return name
    suffix = mimetypes.guess_extension((content_type or "").split(";")[0].strip()) or ".bin"
    if suffix == ".jpe":
        suffix = ".jpg"
    return f"download{suffix}"


def _filename_from_disposition(header: str | None, fallback: str) -> str:
    if not header:
        return fallback
    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', header, re.I)
    if not match:
        return fallback
    name = PurePosixPath(unquote(match.group(1).strip())).name
    return name or fallback


def detect_fetch_mode(url: str, timeout_sec: float | None = None) -> Literal["binary", "html"]:
    """Decide binary download vs HTML crawl without downloading the full body."""
    url = normalize_url(url)
    if is_binary_suffix(url):
        return "binary"

    timeout = timeout_sec if timeout_sec is not None else float(settings.crawl_timeout_sec)
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            try:
                head = client.head(url)
                if head.status_code < 400 and is_binary_content_type(head.headers.get("content-type")):
                    return "binary"
                if head.status_code < 400:
                    # Successful HEAD with non-binary type → HTML
                    ct = head.headers.get("content-type")
                    if ct and not is_binary_content_type(ct):
                        return "html"
            except httpx.HTTPError:
                pass

            # Fallback: small GET to inspect Content-Type
            with client.stream("GET", url) as resp:
                resp.raise_for_status()
                if is_binary_content_type(resp.headers.get("content-type")):
                    return "binary"
    except httpx.HTTPError as exc:
        logger.warning("Content-type probe failed for %s: %s — treating as HTML", url, exc)

    return "html"


def download_binary(url: str, timeout_sec: float | None = None) -> BinaryFetch:
    url = normalize_url(url)
    timeout = timeout_sec if timeout_sec is not None else float(settings.crawl_timeout_sec)
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        data = resp.content
        if not data:
            raise CrawlError("Downloaded file is empty")
        content_type = resp.headers.get("content-type") or SUFFIX_TO_CONTENT_TYPE.get(
            _path_suffix(str(resp.url)), "application/octet-stream"
        )
        fallback = _filename_from_url(str(resp.url), content_type)
        filename = _filename_from_disposition(resp.headers.get("content-disposition"), fallback)
        if PurePosixPath(filename).suffix.lower() not in BINARY_SUFFIXES:
            # Force a usable suffix from content-type / URL
            filename = _filename_from_url(str(resp.url), content_type)
        return BinaryFetch(
            data=data,
            filename=filename,
            content_type=content_type.split(";")[0].strip(),
            source_url=url,
            final_url=str(resp.url),
        )


def _markdown_from_result(result: Any) -> str:
    md = getattr(result, "markdown", None)
    if md is None:
        return ""
    if isinstance(md, str):
        return md.strip()
    fit = getattr(md, "fit_markdown", None) or ""
    raw = getattr(md, "raw_markdown", None) or ""
    text = (fit or raw or str(md)).strip()
    return text


def _internal_links(result: Any) -> list[str]:
    links = getattr(result, "links", None) or {}
    if not isinstance(links, dict):
        return []
    internal = links.get("internal") or []
    out: list[str] = []
    seen: set[str] = set()
    for item in internal:
        href = item.get("href") if isinstance(item, dict) else None
        if not href or href in seen:
            continue
        seen.add(href)
        out.append(href)
    return out


async def crawl_html(url: str, timeout_sec: float | None = None) -> HtmlFetch:
    url = normalize_url(url)
    timeout = timeout_sec if timeout_sec is not None else float(settings.crawl_timeout_sec)

    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

    browser_cfg = BrowserConfig(headless=settings.crawl_headless)
    run_cfg = CrawlerRunConfig(page_timeout=int(timeout * 1000))

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(url=url, config=run_cfg)

    if not getattr(result, "success", True):
        err = getattr(result, "error_message", None) or "crawl4ai failed"
        raise CrawlError(err)

    markdown = _markdown_from_result(result)
    if not markdown:
        raise CrawlError("Crawl returned empty markdown")

    meta = getattr(result, "metadata", None) or {}
    title = None
    if isinstance(meta, dict):
        title = meta.get("title")

    final_url = getattr(result, "url", None) or url
    return HtmlFetch(
        markdown=markdown,
        title=title,
        source_url=url,
        final_url=str(final_url),
        links_internal=_internal_links(result),
    )


def fetch_url(
    url: str,
    *,
    on_progress: ProgressCallback | None = None,
) -> FetchResult:
    """Synchronous entry: detect mode, then download or crawl HTML."""
    url = normalize_url(url)

    def progress(step: str, message: str, percent: int | None = None, detail: dict | None = None) -> None:
        if on_progress:
            on_progress("crawl", step, message, percent=percent, detail=detail)

    progress("detecting", "Detecting URL content type…", percent=2)
    mode = detect_fetch_mode(url)
    progress(
        "detected",
        f"Detected {'binary file' if mode == 'binary' else 'HTML page'}",
        percent=5,
        detail={"mode": mode},
    )

    if mode == "binary":
        progress("downloading", "Downloading file…", percent=8)
        result = download_binary(url)
        progress(
            "done",
            f"Downloaded {result.filename} ({len(result.data)} bytes)",
            percent=12,
            detail={"filename": result.filename, "bytes": len(result.data)},
        )
        return result

    progress("crawling_html", "Crawling page with crawl4ai…", percent=8)
    import asyncio

    try:
        result = asyncio.run(crawl_html(url))
    except RuntimeError:
        # Already inside an event loop (e.g. some test runners)
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(crawl_html(url))
        finally:
            loop.close()

    progress(
        "done",
        f"Crawled HTML — {len(result.markdown)} chars markdown",
        percent=12,
        detail={"final_url": result.final_url, "link_count": len(result.links_internal)},
    )
    return result


def parsed_document_from_markdown(
    markdown: str,
    *,
    source_url: str,
    source_filename: str | None = None,
    title: str | None = None,
) -> ParsedDocument:
    """Build a single-page ParsedDocument from crawled markdown (no OCR)."""
    text = (markdown or "").strip()
    if not text:
        raise CrawlError("Cannot build ParsedDocument from empty markdown")

    blocks: list[TextBlock] = []
    order = 0
    # Split on blank lines; treat heading lines as their own blocks
    chunks = re.split(r"\n\s*\n", text)
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        lines = chunk.splitlines()
        if len(lines) > 1 and re.match(r"^#{1,6}\s+\S", lines[0]):
            heading = lines[0].lstrip("#").strip()
            body = "\n".join(lines[1:]).strip()
            blocks.append(
                TextBlock(block_id=f"p1-b{order}", text=heading, reading_order=order)
            )
            order += 1
            if body:
                blocks.append(
                    TextBlock(block_id=f"p1-b{order}", text=body, reading_order=order)
                )
                order += 1
        else:
            blocks.append(
                TextBlock(block_id=f"p1-b{order}", text=chunk, reading_order=order)
            )
            order += 1

    if not blocks:
        blocks.append(TextBlock(block_id="p1-b0", text=text, reading_order=0))

    filename = source_filename or _filename_from_url(source_url) or "page.md"
    if not filename.endswith((".md", ".html", ".htm")):
        filename = f"{filename}.md"

    notes = ["Source: crawl4ai HTML markdown (OCR skipped)"]
    if title:
        notes.append(f"Page title: {title}")

    return ParsedDocument(
        source_filename=filename,
        source_path=source_url,
        page_count=1,
        pages=[
            ParsedPage(
                page_number=1,
                blocks=blocks,
                full_text=text,
            )
        ],
        structure_notes=notes,
        raw_docling_markdown=text,
        meta={
            "source": "crawl4ai",
            "source_url": source_url,
            "title": title,
        },
    )


def crawl_result_to_artifact(result: FetchResult) -> dict[str, Any]:
    """JSON-serializable crawl summary (links stored, not followed)."""
    if isinstance(result, BinaryFetch):
        return {
            "kind": "binary",
            "source_url": result.source_url,
            "final_url": result.final_url,
            "filename": result.filename,
            "content_type": result.content_type,
            "bytes": len(result.data),
            "links_internal": [],
        }
    return {
        "kind": "html",
        "source_url": result.source_url,
        "final_url": result.final_url,
        "title": result.title,
        "content_type": result.content_type,
        "markdown_chars": len(result.markdown),
        "markdown_preview": result.markdown[:2000],
        "links_internal": result.links_internal,
    }
