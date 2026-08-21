# Catalog Product Extraction POC

3-stage extractor for **catalogs and pamphlets**:

1. **Document Parsing** — selectable channel: PaddleOCR + Docling, Gemini Vision (`gemini-2.5-flash`), or Unlimited-OCR (`baidu/Unlimited-OCR` via HF transformers, CUDA GPU) → `ParsedDocument` only  
2. **Product Extraction** — `gemini-2.5-flash` structured JSON (`products` array) → candidates (all parse channels)  
3. **Validation** — Pydantic + business rules → final `ProductRecord`s  

**Hard rule:** OCR/parsing never emits the final product record. Products come only from the LLM structured output, then validation.

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

Set `GEMINI_API_KEY` in `.env` (required for Stage 2 on all channels, and for Gemini Stage 1).

Optional Gemini settings:
- `GEMINI_API_KEY` — required for Stage-2 extraction always; also required when parse channel is `gemini` or `unlimited_ocr`
- `GEMINI_MODEL` — defaults to `gemini-2.5-flash`
- `PARSE_CHANNEL_DEFAULT` — defaults to `paddleocr` (options: `paddleocr`, `gemini`, `unlimited_ocr`)
- `OCR_USE_GPU` — set `true` to request CUDA for PaddleOCR
- `OCR_GPU_ID` — CUDA device index (default `0`)

Optional DeepSeek settings (legacy Stage-2 client still in tree; not used by the main pipeline):
- `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`

Optional Unlimited-OCR channel settings (local CUDA **or** Colab remote):
- `UNLIMITED_OCR_MODEL` — HF model id or local path (default `baidu/Unlimited-OCR`)
- `UNLIMITED_OCR_GPU_ID` — CUDA device index (default `0`)
- `UNLIMITED_OCR_MAX_LENGTH` — max generation length (default `32768`)
- `UNLIMITED_OCR_BASE_SIZE` / `UNLIMITED_OCR_IMAGE_SIZE` — gundam-mode resolution settings (defaults `1024`/`640`)
- `UNLIMITED_OCR_NO_REPEAT_NGRAM_SIZE` / `UNLIMITED_OCR_NGRAM_WINDOW` — repetition suppression (defaults `35`/`128`)
- `UNLIMITED_OCR_REMOTE_URL` — optional Colab-hosted base URL (e.g. via a Cloudflare quick tunnel); when set, Stage-1 calls the remote `/infer` API instead of loading the model locally
- `UNLIMITED_OCR_REMOTE_API_KEY` — optional shared secret sent as `X-API-Key` (must match Colab secret `UOCR_API_KEY`)
- `UNLIMITED_OCR_REMOTE_TIMEOUT_SEC` — per-page HTTP timeout (default `180`)

Optional crawl settings: `CRAWL_TIMEOUT_SEC`, `CRAWL_HEADLESS`.

## Optional: Unlimited-OCR via Colab (POC)

Use this when you do not have a local CUDA GPU. The notebook hosts `baidu/Unlimited-OCR` behind FastAPI and exposes it with a Cloudflare quick tunnel (no account or signup required).

1. Open [`colab/unlimited_ocr_colab_server.ipynb`](colab/unlimited_ocr_colab_server.ipynb) in Google Colab.
2. Runtime → Change runtime type → **GPU** (T4 is enough).
3. Colab Secrets (🔑 icon):
   - `UOCR_API_KEY` (optional shared secret)
4. Run all cells top-to-bottom. The last cell prints `PUBLIC URL: https://….trycloudflare.com`.
5. In local `.env`:

```env
UNLIMITED_OCR_REMOTE_URL=https://YOUR-SUBDOMAIN.trycloudflare.com
UNLIMITED_OCR_REMOTE_API_KEY=same-as-UOCR_API_KEY-if-set
```

6. Restart local `uvicorn` so settings reload, then submit a job with `parse_channel=unlimited_ocr`.

**Important:** Cloudflare quick tunnel URLs change every Colab restart (the free quick-tunnel mode has no persistent hostname). Re-copy the printed URL into `.env` and restart uvicorn after each restart. Colab free sessions also disconnect (~12h max / idle timeout) — re-run the notebook when that happens.

On the Colab side, if VRAM usage ≥ 80% the whole model is temporarily moved to CPU for that request (slower). The local UI shows a single `cpu_offload` progress line only when that happens.

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
| POST | `/api/jobs` | Upload PDF/image (`multipart` field `file`, optional `parse_channel`: `paddleocr`/`gemini`/`unlimited_ocr`) |
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
- `products.json` — Stage 3 validated/rejected records, each optionally carrying `price_variants` (see Notes)

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
- Gemini channel sends the **original** file to Gemini in one request (no local page rasterization). Files ≤ 15 MB go as `inline_data` with the correct MIME; larger files use the Gemini Files API (`file_data`). Besides PDF/images, Gemini Stage 1 also accepts text formats: `.csv`, `.txt`, `.md`, `.html`, `.htm`, `.xml`, `.json`, `.rtf` (text-as-text; not visual layout). These text types are **Gemini-channel only** — PaddleOCR/Unlimited-OCR uploads reject them. `.xlsx` / `.doc` / `.docx` are **not** supported (Gemini rejects Office Open XML MIME types). If `GEMINI_API_KEY` is missing and channel is `gemini`, Stage-1 fails clearly.
- Unlimited-OCR channel runs `baidu/Unlimited-OCR` via HF `transformers` (local CUDA) **or** a Colab-hosted endpoint (e.g. Cloudflare quick tunnel) when `UNLIMITED_OCR_REMOTE_URL` is set. It produces per-page markdown. A local CUDA GPU is required only when no remote URL is configured. `transformers>=4.57.0,<4.58.0` is pinned per the model card; this may conflict with Docling's own transformers dependency — use a fresh venv if pip resolution fails.
- Stage 2 always uses Gemini structured JSON (`GEMINI_MODEL`, default `gemini-2.5-flash`). Missing `GEMINI_API_KEY` fails Stage 2 clearly — it will not invent products from OCR.
- Product variants: a product's `price_variants` covers any distinct option of that product/service —
  price-column tables (age tiers, portion sizes), attribute-only options with no price difference
  (size, color, pack), and tiered/plan-based offerings. Each variant carries `attribute_name`,
  `label`, `price`, `currency`, `unit`, `pack_size`, `sku`, `available`, `is_default`, and `details`.
  Exactly one variant per product is marked `is_default`; the top-level `price`/`unit` mirror that
  variant for backward compatibility.
  - Non-numeric variant prices (free, included, complimentary, n/a, tbd, on request, `-`) are recorded
    in `available`, not treated as errors.
  - A variant with an unparseable price never rejects the whole product — it adds a warning
    (on both the variant and the product) instead of a `validation_errors` entry. A product is only
    `rejected` when `name` is empty and no other identity/value field (brand, sku, product_code, price,
    description, price_variants) is present.
  - Stage 2 candidates are grouped by `(name, page, sku)` before validation; duplicate-keyed candidates
    (e.g. a variant table split across a chunk boundary) have their `price_variants` merged rather than
    the later occurrence being dropped.
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
