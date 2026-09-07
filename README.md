# RAG Practice

A modular RAG (Retrieval-Augmented Generation) pipeline built entirely with open source tools. Demonstrates the full RAG lifecycle — chunking a PDF, generating vector embeddings, similarity search, and LLM-backed answer generation — with clear separation between stages so each can be tested and understood independently. The project also doubles as a reference for good Python dev practices: isolated environments, linting, type checking, automated tests with coverage gates, and CI.

## Pipeline Overview

```
PDF Document ─[chunking]─> JSON Chunks ─[embeddings]─> FAISS Index ─[retrieval]─> Top-K Chunks ─[generation]─> LLM Answer
```

Four isolated stages, each producing an artifact consumed by the next:

| Stage | Input | Output | Command |
|-------|-------|--------|---------|
| **Chunking** | PDF file | `generated_chunks.json` | `uv run chunk_pdf` |
| **Embeddings** | `generated_chunks.json` | `index.faiss` | `uv run build_embeddings` |
| **Retrieval** | FAISS index + chunks JSON | Terminal results | `uv run python -m src.pipeline.retrieval --prompt "..."` |
| **Generation** | Retrieved chunks + prompt | LLM-generated answer | `uv run python -m src.pipeline.generation --prompt "..."` |

Retrieval and generation are also exposed over HTTP via a small FastAPI server — see [Local Server](#local-server-fastapi) — and through a browser UI, see [Streamlit UI](#streamlit-ui).

## Why `uv`

https://github.com/astral-sh/uv is a fast Python package and environment manager. This project uses uv to manage dependencies, create isolated environments, and run commands reproducibly without requiring global package installations.

## Automation with Nox

[Nox](https://nox.thea.codes/) is used to automate environment setup, linting, typing, tests, and the pipeline itself in isolated, reproducible environments.

### Sessions

Defined in [`noxfile.py`](noxfile.py):

- **`dev`** — Creates/updates the project virtualenv with all dependencies (`uv sync --all-extras --all-groups`). Run this first.
- **`etl_pipeline`** — Runs `chunk_pdf` followed by `build_embeddings`, producing the chunks JSON and the FAISS index in one go.
- **`web_server`** — Serves the FastAPI application locally.
- **`ui`** — Serves the Streamlit UI locally.
- **`format`** — Checks import sorting and formatting with Ruff.
- **`lint`** — Runs Ruff lint checks and writes a JUnit/HTML report to `.reports/linter/`.
- **`typing`** — Runs `mypy` over `src`.
- **`test`** — Runs the test suite with `pytest` + `coverage`, writing reports to `.reports/pytest/` and `.reports/coverage/`.
- **`lock`** — Upgrades `uv.lock`.

Each nox session runs in its own isolated virtual environment under `.nox/`, ensuring a clean and reproducible execution environment. The project's main virtual environment is managed separately in `.venv`, created by the `dev` session.

Running `nox` without arguments runs every session in the file above in order. In practice you'll usually run sessions individually:

```bash
nox -s dev           # sync environment (run this first)
nox -s etl_pipeline  # chunk the PDF and build the FAISS index
nox -s web_server    # serve the FastAPI endpoints
nox -s ui            # serve the Streamlit UI
nox -s format        # check formatting/import order
nox -s lint          # lint with Ruff
nox -s typing        # type-check with mypy
nox -s test          # run tests with coverage
```

These same `format` / `lint` / `typing` / `test` sessions run in CI on every pull request — see [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Folder Structure

```
├── pyproject.toml               # Project config, dependencies, CLI entry points
├── noxfile.py                   # Automation sessions
├── uv.lock                      # Locked dependency tree
├── .python-version              # Python version pinning
├── .github/workflows/ci.yml     # CI: format, lint, typing, test on every PR
├── src/
│   ├── pipeline/                # Thin CLI entry points that wire up the func builders
│   │   ├── chunking.py          #   PDF -> JSON chunks
│   │   ├── embedding.py         #   chunks -> FAISS index
│   │   ├── retrieval.py         #   query the FAISS index
│   │   └── generation.py        #   retrieve + generate an LLM answer
│   ├── func/                    # Reusable builder classes containing all core logic
│   │   ├── chunking_builder.py  #   PDF text extraction + word-level chunking
│   │   ├── embedding_builder.py #   embedding generation + FAISS index building
│   │   ├── retriever_builder.py #   similarity search against the index
│   │   └── generator_builder.py #   RAG prompt assembly + local LLM generation
│   ├── tools/
│   │   └── file_finder.py       # Locates the PDF/JSON/FAISS files in supporting_files/
│   ├── local_server/
│   │   └── web_server.py        # FastAPI app exposing /rag/retrieval and /rag/generate
│   ├── local_ui/
│   │   ├── app.py               # Streamlit front end (upload a PDF, ask a question)
│   │   └── pipeline.py          # Streamlit-free orchestration: ingest PDF, answer query
│   └── tests/                   # pytest suite (unit tests for every module above)
├── supporting_files/            # Place your PDF here; outputs land here too
└── .nox/                        # Per-session virtualenvs (auto-generated by nox)
```

### Architecture

- **`src/pipeline/`** — Thin CLI entry points that wire up the builders. `chunk_pdf` and `build_embeddings` are registered as console scripts in `pyproject.toml`; `retrieval` and `generation` are run as modules (`python -m ...`).
- **`src/func/`** — Reusable builder classes containing all core logic. Importable by tests, the pipeline scripts, or the local server.
- **`src/tools/`** — Small shared utilities, currently `file_finder.py`, which locates the single PDF/JSON/FAISS file in `supporting_files/` so no paths are hardcoded.
- **`src/local_server/`** — Optional FastAPI server that exposes retrieval and generation over HTTP.
- **`src/local_ui/`** — Optional Streamlit front end. `pipeline.py` holds the orchestration logic (kept free of Streamlit imports so it stays unit-testable), and `app.py` is the thin UI layer.
- **`supporting_files/`** — The single data directory. Source PDF, generated chunk JSON, and the FAISS index all live here; each `file_finder` lookup expects exactly one file of its type in this folder.

## How to Use

### 1. Prerequisites

Install `uv` (one-time) — see the [official install instructions](https://docs.astral.sh/uv/getting-started/installation/), e.g. on macOS/Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Add Your PDF

Place a single PDF file in `supporting_files/`. `file_finder.get_pdf_file` expects exactly one `*.pdf` in that folder — it raises if none or more than one is found.

### 3. Sync the Environment

```bash
nox -s dev
```

### 4. Run Chunking and Embeddings

```bash
uv run chunk_pdf          # -> supporting_files/generated_chunks.json
uv run build_embeddings   # -> supporting_files/index.faiss
```

Or run both via nox: `nox -s etl_pipeline`.

### 5. Retrieve

```bash
uv run python -m src.pipeline.retrieval --prompt "write your prompt here"
```

Prints the top-3 most similar chunks (top-k is currently fixed via `RetrieverBuilder.top_k`, not a CLI flag).

### 6. Generate an Answer

```bash
uv run python -m src.pipeline.generation --prompt "write your prompt here"
```

Retrieves context, builds a chat prompt, and generates an answer locally with `microsoft/Phi-3-mini-4k-instruct` via `transformers`. The model is downloaded and loaded into memory on first run, so expect a slower first invocation and a GPU is used automatically if `torch.cuda.is_available()`.

## Local Server (FastAPI)

`src/local_server/web_server.py` exposes the same retrieval and generation logic over HTTP:

```bash
uv run local_server
```

Starts a Uvicorn server at `http://127.0.0.1:8011` with two endpoints:

- `POST /rag/retrieval` — body `{"prompt": "..."}`, returns the ranked list of retrieved chunks.
- `POST /rag/generate` — body `{"prompt": "..."}`, returns `{"generated_answer": "..."}`.

## Streamlit UI

`src/local_ui/app.py` wraps the whole pipeline in a browser UI so you can upload a PDF and ask questions about it without touching the CLI:

```bash
nox -s ui
```

Equivalently, `uv run streamlit run src/local_ui/app.py`. Streamlit opens at `http://localhost:8501`.

**Flow:**

1. **Upload a PDF** — the file is written to `supporting_files/`, then chunked and embedded automatically. Because `file_finder` expects exactly one file of each type in that folder, uploading a new PDF first clears any existing `*.pdf`, `*.json`, and `*.faiss` artifacts. Uploading replaces the previous document rather than adding to it.
2. **Ask a question** — the question box and *Ask* button stay disabled until a PDF has been indexed.
3. **Read the results** — the retrieved passages are listed first (each expandable, with page number and similarity score), followed by the generated answer.

**Notes:**

- The Phi-3 generation model is imported lazily and cached with `@st.cache_resource`, so only the first question in a session pays the model-loading cost.
- Retrieval runs once per question and its output is fed straight into the generator, avoiding a duplicate FAISS search.
- The Streamlit-free logic in `src/local_ui/pipeline.py` is covered by `src/tests/unit/test_local_ui_pipeline.py`.

## FAISS — Vector Similarity Search

[FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search) is an open source library for efficient similarity search and clustering of dense vectors.

In this pipeline:

- **Index type**: `IndexFlatIP` (inner product). Since embeddings are L2-normalized, inner product is equivalent to cosine similarity, giving a natural similarity score between 0 and 1.
- **Storage**: The index is serialized to disk as a single `.faiss` binary file — no database server needed.
- **Search**: At query time, the user's prompt is embedded with the same `SentenceTransformer` model (`BAAI/bge-small-en-v1.5`), then FAISS finds the top-K nearest vectors in the index by brute-force exact search.

## Testing & Quality Gates

The test suite lives in `src/tests/` (unit tests under `src/tests/unit/`) and covers every builder, pipeline entry point, `file_finder`, and the FastAPI server. Run it with:

```bash
nox -s test
```

Coverage is measured over `src` with a minimum threshold of 80% (`[tool.coverage.report] fail_under = 80` in `pyproject.toml`); `nox -s lint` and `nox -s typing` enforce Ruff and mypy respectively. All four checks (`format`, `lint`, `typing`, `test`) run automatically in CI on every pull request via [`.github/workflows/ci.yml`](.github/workflows/ci.yml).
