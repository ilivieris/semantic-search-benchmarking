# Semantic Search Engine Comparison

> Drop documents into `data/`, build two independent semantic search engines, and compare their results side-by-side.

---

## Overview

This project provides a **plug-and-play framework** for building and benchmarking semantic search engines over any document corpus. Add your JSON documents to the `data/` folder, run the build script, and two independent engines spin up — one per embedding model — ready for side-by-side comparison through a dedicated UI and benchmark script.

Each document chunk is embedded into a dense vector space using multilingual sentence transformers. Queries are matched against this space using **FAISS** for fast approximate nearest-neighbour search.

> **Reference use-case:** the default schema and examples use the Greek **Government Gazette (ΦΕΚ — Φύλλο Εφημερίδας της Κυβερνήσεως)** as a real-world corpus. Adapt the chunk extractor in `app/builder.py` to fit your own document format.

---

## Architecture

```
Query (natural language)
        │
        ▼
  Sentence Transformer          ← same model used at index time
  (encode query → float32 vec)
        │
        ▼
  FAISS IndexFlatIP             ← exact cosine similarity search
  (L2-normalised inner product)
        │
        ▼
  chunk lookup  chunks.json     ← {int → metadata dict}
        │
        ▼
  JSON response  { rank, score, ...metadata }
```

**Index build pipeline**

```
data/*.json  ──►  chunk extraction  ──►  embedding (batch)  ──►  FAISS index  ──►  engines/<name>/
```

---

## Models

| Engine key | Model | Dim | Notes |
|---|---|---|---|
| `mpnet` | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 768 | Strong multilingual baseline |
| `novelcore` | `novelcore/embeddings-model` | — | Domain-tuned alternative |

> Any model from the [sentence-transformers library](https://www.sbert.net/docs/pretrained_models.html) can be swapped in via `.env`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | [sentence-transformers](https://www.sbert.net/) |
| Vector index | [FAISS](https://github.com/facebookresearch/faiss) (`IndexFlatIP`) |
| API | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn |
| Config | `python-dotenv` |
| Frontend | Vanilla HTML/CSS/JS (no framework) |

---

## Project Structure

```
search_engines/
├── .env                    ← runtime configuration (see below)
├── .env.example            ← configuration template
├── .gitignore
├── requirements.txt
├── README.md
│
├── app/                    ← Python package
│   ├── builder.py          ← chunk extraction + FAISS index builder
│   ├── searcher.py         ← SearchEngine class
│   └── api/
│       ├── main.py         ← FastAPI application entry point
│       └── routers/
│           └── search.py   ← POST /search/mpnet & /search/novelcore
│
├── scripts/
│   ├── build_all.py        ← builds all engines sequentially
│   └── compare.py          ← Hit@k accuracy benchmark
│
├── ui/
│   └── index.html          ← demo frontend (no build step)
│
├── engines/                ← auto-generated (git-ignored)
│   ├── mpnet/
│   │   ├── config.json
│   │   ├── chunks.json
│   │   └── faiss_index.bin
│   └── novelcore/
│       └── ...
│
└── data/                   ← raw JSON documents (git-ignored)
    └── **/*.json
```

---

## Installation

### Option A — Poetry *(recommended)*

```bash
# 1. Install dependencies (Poetry creates and manages the virtualenv automatically)
poetry install

# 2. Copy and edit configuration
cp .env.example .env
```

Activate the managed shell for subsequent commands:

```bash
poetry shell
```

> **Requires Poetry ≥ 1.8.** If Poetry is not installed, see the [official guide](https://python-poetry.org/docs/#installation).

---

### Option B — pip + venv

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and edit configuration
cp .env.example .env
```

---

## Configuration

All settings live in `.env`. The file is never committed to git.

| Variable | Default | Description |
|---|---|---|
| `DATA_PATH` | `data/` | Root folder containing raw JSON documents |
| `ENGINES_DIR` | `engines/` | Output folder for built search engines |
| `BATCH_SIZE` | `64` | Embedding batch size (lower for limited VRAM) |
| `TOP_K` | `5` | Default number of search results |

```ini
# .env
DATA_PATH="data/"
ENGINES_DIR="engines/"
BATCH_SIZE="16"
TOP_K="5"
```

---

## Usage

### 1 — Add your documents

Place JSON files under `data/`. The default extractor expects the ΦΕΚ schema (see [Data Format](#data-format) below). For a different schema, edit the `extract_chunks()` function in `app/builder.py`.

---

### 2 — Build the search engines

Processes all files in `data/`, generates embeddings, and writes the FAISS index for each model.

```bash
python scripts/build_all.py
```

Output per engine:

```
engines/mpnet/
├── config.json        ← { model, dim, num_chunks }
├── chunks.json        ← { "0": {...}, "1": {...}, ... }
└── faiss_index.bin    ← binary FAISS index
```

> **Tip:** building a single engine only:
> ```bash
> ENGINE_NAME=mpnet \
> EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2 \
> python -c "
> from dotenv import load_dotenv; load_dotenv()
> import os
> from app.builder import build_engine
> build_engine(os.getenv('DATA_PATH','data/'), f\"engines/{os.getenv('ENGINE_NAME')}\", os.getenv('EMBEDDING_MODEL'))
> "
> ```

---

### 3 — Start the API

```bash
uvicorn app.api.main:app --reload
```

The API loads both engines into memory at startup and keeps them warm for the lifetime of the process.

```
INFO:     Loading search engines …
[mpnet]     Loading chunks ...
[mpnet]     Loading FAISS index ...
[mpnet]     Loading model: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
[mpnet]     ✓ Ready — 45,231 vectors
[novelcore] ✓ Ready — 45,231 vectors
INFO:     Application startup complete.
```

---

### 4 — Open the demo UI

Open `ui/index.html` directly in a browser — no build step, no server required.

Features:
- Live search with 340 ms debounce
- Engine selector: **MPNET** / **NOVELCORE** / **⚡ COMPARE**
- Side-by-side comparison mode
- Score badges color-coded by confidence
- Expandable text previews
- Real-time API health indicator
- ⌘K / Ctrl+K to focus search

---

### 5 — Run the accuracy benchmark

```bash
python scripts/compare.py
```

Sample output:

```
══════════════════════════════════════════════════════════════════════
  Query 1/5: «αμοιβαία αναγνώριση ανταλλαγή αδειών οδήγησης»
  Expected doc: 554893
══════════════════════════════════════════════════════════════════════

  ── MPNET       [✓ HIT ]  top-1 score=0.9341
     [1] 0.9341  doc 554893  2020-01
     ...

  ── NOVELCORE   [✓ HIT ]  top-1 score=0.8812
     ...

══════════════════════════════════════════════════════════════════════
  SUMMARY  (Hit@5, n=4 queries with ground truth)
══════════════════════════════════════════════════════════════════════
  Engine          Hit@5        Accuracy   Avg Score@1
  ----------------------------------------------------
  mpnet           4/4            100%        0.8934
  novelcore       3/4             75%        0.8201
```

---

## API Reference

Base URL: `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs`

---

### `GET /health`

Returns status and index statistics for all loaded engines.

```json
{
  "status": "ok",
  "engines": {
    "mpnet":     { "ntotal": 45231, "model": "sentence-transformers/..." },
    "novelcore": { "ntotal": 45231, "model": "novelcore/embeddings-model" }
  }
}
```

---

### `POST /search/mpnet`

### `POST /search/novelcore`

**Request body**

```json
{
  "query": "your natural language query",
  "top_k": 5
}
```

| Field | Type | Required | Constraints |
|---|---|---|---|
| `query` | string | ✓ | min length 1 |
| `top_k` | integer | — | 1 – 50, default 5 |

**Response**

```json
{
  "engine": "mpnet",
  "query":  "your natural language query",
  "results": [
    {
      "rank":   1,
      "score":  0.934123,
      "...":    "document-specific metadata fields"
    }
  ]
}
```

**Score interpretation**

| Range | Meaning |
|---|---|
| ≥ 0.80 | High confidence — strong semantic match |
| 0.65 – 0.79 | Medium — related content |
| 0.50 – 0.64 | Low — loosely related |
| < 0.50 | Poor — likely irrelevant |

---

## How It Works

1. **Chunk extraction** — each JSON document is parsed into smaller chunks (e.g. article-level). Each chunk carries the full metadata of its parent document.

2. **Embedding** — the text fed to the model combines metadata and content into a single string, so the model can match queries that reference both context and content.

3. **Indexing** — embeddings are L2-normalised before insertion into `faiss.IndexFlatIP`. Inner product on normalised vectors equals cosine similarity — scores are in `[0, 1]`.

4. **Serving** — at API startup, both engines (index + chunks + model) are loaded into memory once and shared across all requests via `app.state.engines`.

---

## Data Format

The default extractor in `app/builder.py` expects the following schema (ΦΕΚ use-case):

```json
{
  "fek_id":       "554893",
  "issue":        "A",
  "sheet_number": "18",
  "year":         "2020",
  "month":        "01",
  "kads": [
    {
      "header": "ΝΟΜΟΣ ΥΠ' ΑΡΙΘΜ. 4656",
      "title":  "Κύρωση...",
      "sections": {
        "consider": { "text": "...", "is_valid": true },
        "body": [
          { "header": "Άρθρο 1", "title": "...", "text": "...", "is_valid": true }
        ]
      }
    }
  ]
}
```

To use a different document format, implement your own `extract_chunks(doc: dict) -> list[dict]` in `app/builder.py`. Each returned chunk dict must contain at minimum a `"text"` key; all other keys are passed through as metadata and returned in search results.
