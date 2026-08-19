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
- `OCR_USE_GPU` — set `true` to request CUDA for PaddleOCR
- `OCR_GPU_ID` — CUDA device index (default `0`)

Optional crawl settings: `CRAWL_TIMEOUT_SEC`, `CRAWL_HEADLESS`.

## Optional: CUDA setup for PaddleOCR (Windows)

Use this only when you run Stage-1 with `parse_channel=paddleocr`. The Gemini parse channel is remote and does not use your local GPU.

CUDA here accelerates **PaddleOCR detection/recognition only**. Docling layout still runs on CPU (torch / RapidOCR) regardless of `OCR_USE_GPU`.

1. Install/update NVIDIA GPU driver.
2. Replace the CPU `paddlepaddle` wheel with a CUDA-compatible `paddlepaddle-gpu` wheel. An NVIDIA driver alone is not enough if `paddle.device.is_compiled_with_cuda()` is `False`. Confirmed working on this project (Python 3.13, CUDA 12.6, RTX 4060):

```bash
pip uninstall paddlepaddle
pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

3. Verify CUDA is visible (run this in a fresh interpreter, before importing other GPU libraries):

```bash
python -c "import paddle; print('compiled_with_cuda=', paddle.device.is_compiled_with_cuda()); print('device=', paddle.device.get_device())"
```

Expected: `compiled_with_cuda= True` and `device= gpu:0`.

4. Enable GPU in `.env`:

```env
OCR_USE_GPU=true
OCR_GPU_ID=0
```

If CUDA is unavailable (CPU-only Paddle wheel, driver/runtime mismatch, missing libs), this app logs a warning and automatically falls back to CPU OCR. Check `parsed.json` → `meta.ocr_runtime` (`cpu` vs `cuda`) to confirm the effective device. You may also see a cuDNN minor-version warning (for example Paddle built with 9.9 vs local 9.5); OCR can still run on GPU.

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
- `products.json` — Stage 3 records with `status` (`validated`/`needs_review`/`invalid`), each optionally carrying `price_variants` (see Notes)

## Samples

Drop your own catalog PDFs or images into `samples/`. Nothing is bundled (copyright).

## Notes

- First PaddleOCR run may download PaddleOCR models.
- On Windows CPU, PaddleOCR is initialized with `enable_mkldnn=False` to avoid a known oneDNN/PIR crash.
- OCR runtime metadata is written to `parsed.json` under `meta`:
  - `ocr_runtime`: `cpu` or `cuda`
  - `ocr_use_gpu_requested`: requested GPU mode from config
  - `ocr_gpu_id_requested`: requested GPU id
- Docling is used for PDF layout/markdown; its own OCR is disabled because PaddleOCR is primary. Image-only Docling failures are non-fatal — extraction still uses PaddleOCR text.
- Gemini channel uses a vision model to produce Stage-1 text blocks; if `GEMINI_API_KEY` is missing and channel is `gemini`, Stage-1 fails clearly.
- Missing `DEEPSEEK_API_KEY` fails Stage 2 clearly — it will not invent products from OCR.
- Product variants: a product's `price_variants` covers any distinct option of that product/service —
  price-column tables (age tiers, portion sizes), attribute-only options with no price difference
  (size, color, pack), and tiered/plan-based offerings. Each variant carries `attribute_name`,
  `label`, `price`, `currency`, `unit`, `pack_size`, `sku`, `available`, `is_default`, and `details`.
  Exactly one variant per product is marked `is_default`; the top-level `price`/`unit` mirror that
  variant for backward compatibility.
  - Non-numeric variant prices (free, included, complimentary, n/a, tbd, on request, `-`) are recorded
    in `available`, not treated as errors.
  - A variant with an unparseable price never invalidates the whole product — it adds a warning
    (on both the variant and the product) instead of a `validation_errors` entry. A product is only
    `invalid` when `name` is empty and no other identity/value field (brand, sku, product_code, price,
    description, price_variants) is present.
- Stage 2 candidates are grouped by `(name, page, sku)` before validation; duplicate-keyed candidates
    (e.g. a variant table split across a chunk boundary) have their `price_variants` merged rather than
    the later occurrence being dropped.

### Layout-aware spatial association and validation status

Multi-column flyers/catalogs previously let the raw OCR reading order interleave columns
(`name | price | promo`), so a name from one card could be paired with a price or promo code from an
adjacent card while still being marked `validated`. Stage 1 now segments each page into columns and
card-like **regions** (`app/stages/layout.py`) before Stage 2 ever sees the text; Stage 3 then only
associates a product's price/variant fields with OCR tokens found inside that same region.

- `ParsedPage.regions` — candidate visual cards/headings detected on the page (`region_id`, `column_id`,
  `bbox`, `block_ids`, `kind`). `TextBlock.region_id`/`column_id`/`line_id` tag which region/column/line
  each OCR block belongs to. Pages with no bbox data (e.g. the HTML/markdown crawl path) get no regions —
  spatial checks are skipped for those documents rather than producing false positives.
- The Stage 2 prompt text groups blocks under `[region <id>]` headers instead of joining a page-wide row
  with `" | "`; the model is instructed to keep every field of one product within a single region and may
  report `region_id`/`source_block_ids` per product.
- `ProductRecord` now separates:
  - `schema_valid` — the record's shape/types are valid (name present, prices parseable, not an empty shell).
  - `extraction_valid` — every field could be attributed to the product's own region (no unresolved spatial/semantic issue).
  - `status` — one of `validated`, `needs_review`, `invalid`. A record is `validated` only when both
    `schema_valid` and `extraction_valid` are true; `needs_review` means schema-valid but an association is
    unresolved; `invalid` means a schema/business rule failed.
  - `region_id` / `provenance` — the resolved anchor region and, per field, the source bbox/region/block ids
    used to resolve it, for audit.
  - `validation_errors` is now a list of structured `ValidationIssue` objects
    (`code`, `field`, `message`, `product_bbox`, `field_bbox`, `region_id`, `confidence`, `severity`), not
    plain strings. Free-text `warnings` (currency formatting, duplicate names, unparseable variant price)
    are unaffected and never change `status`.
- If a candidate's price (or a variant's price) matches an OCR token that lives in a *different* region than
  the product's own, Stage 3 does not guess — it nulls that field, records a `SPATIAL_ASSOCIATION_AMBIGUOUS`
  issue with both bboxes as evidence, and the record becomes `needs_review`.
- Cross-record semantic checks (`app/stages/semantics.py`) flag additional contamination patterns with no
  hard-coded product names/prices: `MRP_NOT_ABOVE_PRICE`, `SHARED_SOURCE_BBOX`, `NAME_INCLUDES_SEPARATE_FIELD`,
  `PROMO_CODE_OUT_OF_REGION`, `QUANTITY_OUT_OF_REGION`, `ATTRIBUTE_OUT_OF_REGION`.
- Equivalent issues at the product- and variant-price level are deduplicated (`app.models.product.dedupe_issues`)
  so the same contamination isn't reported twice.

Known limitation: Stage 2 is still an LLM call, so region-scoped prompting reduces but cannot prove the
absence of contamination on its own — the Stage 3 region-scoped resolver and semantic checks above are the
deterministic backstop and are what the automated tests assert against.

- Unit tests: `python -m pytest tests -q`

## Benchmark checklist (CPU vs CUDA)

1. Pick one representative multi-page PDF/image and keep all env/settings constant except OCR runtime.
2. Run with CPU:
   - `OCR_USE_GPU=false`
   - submit a job with `parse_channel=paddleocr`
   - record Stage-1 wall time from logs/job progress, split PaddleOCR vs Docling if possible.
3. Run with CUDA:
   - `OCR_USE_GPU=true`, `OCR_GPU_ID=0` (or target GPU)
   - submit the same file and record Stage-1 wall time.
4. Compare:
   - elapsed time and approximate pages/sec (`page_count / ocr_seconds`)
   - `parsed.json -> meta.ocr_runtime` should be `cpu` vs `cuda`.
5. Validate output parity:
   - `parsed.json` shape is valid for both runs
   - downstream candidate/product counts are materially consistent.

Worked example (`tours&travels.png`, 1 page, RTX 4060, paddlepaddle-gpu 3.3.0 / CUDA 12.6):

| Step | CPU (previous run) | CUDA |
|------|--------------------|------|
| PaddleOCR init + page OCR | ~118 s (remainder of Stage-1 after Docling) | 12.98 s init + 2.81 s OCR = **15.8 s** |
| Docling layout (still CPU) | 137 s | 146 s |
| Stage-1 total | 255 s | 162 s |
| Text blocks | 261 | 261 |
| `meta.ocr_runtime` | `cpu` (CPU wheel) | `cuda` |

CUDA sped up the PaddleOCR step by about **7x** on this image. Overall Stage-1 was about **1.6x** faster because Docling remains CPU-bound.
