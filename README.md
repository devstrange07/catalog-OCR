# Catalog Product Extraction POC

3-stage extractor for **catalogs and pamphlets**:

1. **Document Parsing** — PaddleOCR (primary text + boxes) + Docling (layout/structure) → `ParsedDocument` only  
2. **Product Extraction** — `deepseek-v4-flash` with an **enforced** `submit_products` tool call → candidates  
3. **Validation** — Pydantic + business rules → final `ProductRecord`s  

**Hard rule:** OCR/parsing never emits the final product record. Products come only from the LLM tool call, then validation.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # or: cp .env.example .env
```

Set `DEEPSEEK_API_KEY` in `.env`.

## Run

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000).

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/jobs` | Upload PDF/image (`multipart` field `file`) |
| GET | `/api/jobs/{id}` | Job status, stages, products |
| GET | `/api/jobs/{id}/artifacts/{name}` | `parsed.json` / `candidates.json` / `products.json` |

## Artifacts

Each job writes under `outputs/{job_id}/`:

- `parsed.json` — Stage 1 structure (no product schema)
- `candidates.json` — Stage 2 tool-call output
- `products.json` — Stage 3 validated/rejected records

## Samples

Drop your own catalog PDFs or images into `samples/`. Nothing is bundled (copyright).

## Notes

- First OCR run may download PaddleOCR models.
- On Windows CPU, PaddleOCR is initialized with `enable_mkldnn=False` to avoid a known oneDNN/PIR crash.
- Docling is used for PDF layout/markdown; its own OCR is disabled because PaddleOCR is primary. Image-only Docling failures are non-fatal — extraction still uses PaddleOCR text.
- Missing `DEEPSEEK_API_KEY` fails Stage 2 clearly — it will not invent products from OCR.
- Unit tests: `python -m pytest tests -q`
