# Catalog Product Extraction POC

3-stage extractor for **catalogs and pamphlets**:

1. **Document Parsing** — selectable channel: PaddleOCR + Docling, or Gemini Vision (`gemini-2.5-flash`) → `ParsedDocument` only  
2. **Product Extraction** — `deepseek-v4-flash` with an **enforced** `submit_products` tool call → candidates  
3. **Validation** — Pydantic + business rules → final `ProductRecord`s  

**Hard rule:** OCR/parsing never emits the final product record. Products come only from the LLM tool call, then validation.

**URL ingest (single URL):** hybrid crawl — PDF/image URLs download into the OCR path; HTML pages use crawl4ai markdown (OCR skipped), then the same extract → validate stages. Deep crawl / URL queues are not enabled yet.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
crawl4ai-setup
copy .env.example .env   # or: cp .env.example .env
```

`crawl4ai-setup` installs Playwright browsers needed for HTML page crawls (one-time).

Set `DEEPSEEK_API_KEY` in `.env`.

Optional Gemini parse channel settings:
- `GEMINI_API_KEY` — required only when parse channel is `gemini`
- `GEMINI_MODEL` — defaults to `gemini-2.5-flash`
- `PARSE_CHANNEL_DEFAULT` — defaults to `paddleocr`

Optional crawl settings: `CRAWL_TIMEOUT_SEC`, `CRAWL_HEADLESS`.

## Run

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000).

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/jobs` | Upload PDF/image (`multipart` field `file`, optional `parse_channel`: `paddleocr`/`gemini`) |
| POST | `/api/jobs/from-url` | JSON `{ "url": "https://..." }`, optional `parse_channel` |
| GET | `/api/jobs/{id}` | Job status, stages, products |
| GET | `/api/jobs/{id}/artifacts/{name}` | `crawl.json` (URL jobs) / `parsed.json` / `candidates.json` / `products.json` |

### Hybrid URL behavior

1. Path suffix `.pdf` / image → download bytes → OCR parse  
2. Else probe `Content-Type`; `application/pdf` or `image/*` → download → OCR  
3. Otherwise crawl4ai HTML → markdown → `ParsedDocument` (no OCR)  
4. Internal links are stored on `crawl.json` / job meta for a future deep-crawl pass; they are **not** followed

## Artifacts

Each job writes under `outputs/{job_id}/`:

- `crawl.json` — URL jobs only (fetch kind, URLs, markdown preview / file meta, internal links)
- `parsed.json` — Stage 1 structure (no product schema)
- `candidates.json` — Stage 2 tool-call output
- `products.json` — Stage 3 validated/rejected records

## Samples

Drop your own catalog PDFs or images into `samples/`. Nothing is bundled (copyright).

## Notes

- First PaddleOCR run may download PaddleOCR models.
- On Windows CPU, PaddleOCR is initialized with `enable_mkldnn=False` to avoid a known oneDNN/PIR crash.
- Docling is used for PDF layout/markdown; its own OCR is disabled because PaddleOCR is primary. Image-only Docling failures are non-fatal — extraction still uses PaddleOCR text.
- Gemini channel uses a vision model to produce Stage-1 text blocks; if `GEMINI_API_KEY` is missing and channel is `gemini`, Stage-1 fails clearly.
- Missing `DEEPSEEK_API_KEY` fails Stage 2 clearly — it will not invent products from OCR.
- Unit tests: `python -m pytest tests -q`
