# 📄 Document Analyser RAG

> A modular AI pipeline for extracting structured data from PDFs using RAG with OpenAI, LangChain, and Chroma — with pluggable retrieval strategies, an evaluation harness, and full cost/latency observability.

[![CI](https://github.com/pedroberiotto/document-analyzer-with-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/pedroberiotto/document-analyzer-with-rag/actions/workflows/ci.yml)
![App](https://img.shields.io/badge/App-Streamlit-ff4b4b)
![Backend](https://img.shields.io/badge/API-FastAPI-009688)
![LLM](https://img.shields.io/badge/LLM-OpenAI-412991)
![RAG](https://img.shields.io/badge/Pattern-RAG-1f6feb)
![Vector%20Store](https://img.shields.io/badge/Vector%20Store-Chroma-00c853)
![Orchestration](https://img.shields.io/badge/Orchestration-LangChain-1a73e8)
![License](https://img.shields.io/badge/License-MIT-orange)

---

## ✨ Features

- 🔌 **Bring-your-own schema** — define custom fields in JSON (name + description + type).
- 📚 **RAG over a single document** — chunk, embed, and index each PDF in its own Chroma collection.
- 📑 **Table-aware ingestion** — tables are extracted (pdfplumber) and indexed as whole chunks, so table-heavy financial/legal docs stay searchable.
- 🛡️ **Robustness** — typed errors (invalid PDF → 422, unknown doc → 404, LLM failure → 502), request timeouts, exponential-backoff retries, and embedding cache by `document_id` (no double-billing).
- 🔀 **Pluggable retrieval strategies**
  - `dense` — vector similarity
  - `hybrid` — BM25 + dense fused with Reciprocal Rank Fusion (RRF)
  - `agentic` — iterative retrieval that re-queries when context confidence is low
  - optional **cross-encoder reranking** on top of any strategy
- 🧠 **LLM field extraction** — structured outputs (Pydantic) with `confidence` + evidence `sources[]`.
- 🔌 **Pluggable providers** — OpenAI or **Ollama** (local/free) via one env var.
- 📊 **Observability** — structured logs (structlog) plus per-field / per-document **tokens, USD cost, and latency** persisted to SQLite.
- 🧪 **Evaluation harness** — run a strategy over an annotated dataset and compare all strategies (field accuracy, retrieval precision, faithfulness, latency, cost).
- 🖥️ **Streamlit UI** + ⚙️ **FastAPI** sharing one orchestration service.

---

## 📂 Project Structure

```text
document-analyzer-with-rag/
├── app/
│   ├── config.py                # Central Settings (pydantic-settings): models, chunking, k, paths, logging
│   ├── ingestion.py             # PDF → chunks → embeddings → Chroma
│   ├── retrieval/               # Retrieval strategies (the "R" in RAG)
│   │   ├── base.py              #   RetrieverBase interface
│   │   ├── dense.py             #   vector similarity
│   │   ├── hybrid.py            #   BM25 + dense (RRF)
│   │   ├── reranker.py          #   cross-encoder reranking wrapper
│   │   ├── agentic.py           #   iterative re-querying
│   │   └── factory.py           #   build_retriever() + strategy/reranker enums
│   ├── extraction/              # The "AG" — answer generation
│   │   ├── llm.py               #   LLM, prompt, FieldAnswer schema
│   │   └── pipeline.py          #   field-by-field RAG loop + telemetry
│   ├── telemetry/               # Cross-cutting observability
│   │   ├── logging.py           #   structlog setup (console | json)
│   │   ├── pricing.py           #   cost from config/pricing.json
│   │   └── tracker.py           #   per-field/-doc logging + SQLite + in-memory buffer
│   ├── models/                  # Pydantic models (schema, result, telemetry)
│   ├── service.py               # ExtractionService — orchestration shared by API + UI
│   └── main.py                  # FastAPI app
├── config/
│   └── pricing.json             # USD price table per model
├── eval/                        # Evaluation harness
│   ├── dataset/                 #   PDFs to evaluate (gitignored content)
│   ├── ground_truth.json        #   hand-annotated expected values
│   ├── metrics.py               #   field match, retrieval precision, faithfulness
│   ├── run_eval.py              #   run ONE strategy → metrics + eval.db
│   ├── compare.py               #   run ALL strategies → comparison.csv/json
│   └── db.py                    #   SQLite store of eval results
├── data/                        # Runtime artifacts (gitignored)
│   ├── uploads/  chroma/  telemetry.db
├── streamlit_app.py             # Streamlit UI entrypoint
└── requirements.txt
```

The package layout mirrors the three canonical RAG stages — **ingestion → retrieval → extraction** — with `service.py` tying them together so the UI and API never duplicate wiring.

---

## 🔌 Model providers (run it for free)

The LLM and embeddings are **pluggable** — pick a provider via env, no code changes:

| Provider | Use case | Cost | Setup |
|----------|----------|------|-------|
| **Ollama** | local dev / offline | **free** | `ollama serve` + `ollama pull llama3.2:3b nomic-embed-text` |
| **OpenAI** | best quality | paid | `OPENAI_API_KEY` |

```bash
LLM_PROVIDER=ollama EMBEDDING_PROVIDER=ollama   # 100% local, $0
LLM_PROVIDER=openai                             # gpt-4.1-mini (OPENAI_API_KEY)
```

> ⚠️ Embeddings must match between indexing and querying a document — re-index after switching `EMBEDDING_PROVIDER`.

## 🚀 Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                 # core deps (add ".[dev]" for tests/lint, ".[reranker]" for cross-encoder)

cp .env.example .env             # then edit: choose a provider + add any keys

# Free local route (recommended for trying it out):
ollama serve &
ollama pull llama3.2:3b && ollama pull nomic-embed-text
# .env -> LLM_PROVIDER=ollama, EMBEDDING_PROVIDER=ollama
```

### Docker

```bash
docker compose up --build                 # API on :8000, UI on :8501
docker compose --profile local up --build # also start a local Ollama service
```

### Streamlit UI

```bash
streamlit run streamlit_app.py
```

Upload a PDF, pick a retrieval strategy in the sidebar, paste a schema, and run.
The UI shows each field's value, confidence, evidence snippets, and **per-field cost / tokens / latency**.

### REST API

```bash
uvicorn app.main:app --reload
```

| Method | Path                | Description                          |
|--------|---------------------|--------------------------------------|
| GET    | `/healthz`          | liveness probe                       |
| POST   | `/documents/upload` | upload + index a PDF                 |
| GET    | `/documents`        | list indexed document ids            |
| POST   | `/schemas`          | register an extraction schema        |
| GET    | `/schemas`          | list registered schemas              |
| POST   | `/extract`          | run extraction (returns result + telemetry) |

```bash
# 1. upload
curl -F "file=@invoice.pdf" localhost:8000/documents/upload
# 2. register a schema (JSON body = ExtractionSchema)
curl -X POST localhost:8000/schemas -H 'Content-Type: application/json' -d @schema.json
# 3. extract
curl -X POST "localhost:8000/extract?document_id=<id>&schema_name=basic_invoice&retriever_strategy=hybrid&reranker=cross_encoder"
```

---

## ⚙️ Configuration

All tunables live in `app/config.py` (`Settings`) and can be overridden via environment variables or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | `openai` \| `ollama` |
| `EMBEDDING_PROVIDER` | `openai` | `openai` \| `ollama` |
| `OPENAI_API_KEY` | – | OpenAI credentials |
| `LLM_MODEL` / `OLLAMA_MODEL` | `gpt-4.1-mini` / `llama3.2:3b` | model per provider |
| `EMBEDDING_MODEL` / `OLLAMA_EMBEDDING_MODEL` | `text-embedding-3-small` / `nomic-embed-text` | embedding model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | splitter settings |
| `EXTRACT_TABLES` | `true` | also index tables via pdfplumber |
| `REINDEX_IF_EXISTS` | `false` | re-embed even if the doc is already cached |
| `DEFAULT_STRATEGY` / `DEFAULT_RERANKER` | `dense` / `none` | UI defaults |
| `LLM_TIMEOUT` / `LLM_MAX_RETRIES` | `60` / `3` | request timeout + backoff retries |
| `RETRIEVAL_K` | `5` | chunks retrieved per query |
| `RRF_K` | `60` | RRF constant (hybrid) |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | reranker |
| `AGENTIC_CONFIDENCE_THRESHOLD` / `AGENTIC_MAX_ITERATIONS` | `0.6` / `3` | agentic loop |
| `LOG_LEVEL` / `LOG_FORMAT` | `INFO` / `console` | logging (`console`\|`json`) |

```bash
LLM_MODEL=gpt-4.1 RETRIEVAL_K=8 LOG_FORMAT=json uvicorn app.main:app
```

---

## 📊 Observability

Every extraction emits structured logs and persists telemetry to `data/telemetry.db`
(`field_extractions` + `document_runs` tables): tokens, USD cost (from `config/pricing.json`),
and latency, broken down per field and per document. Switch to JSON logs with `LOG_FORMAT=json`.

---

## 🧪 Evaluation

1. Drop PDFs in `eval/dataset/` and fill in their expected values in `eval/ground_truth.json`.
2. Run a single strategy or compare them all:

```bash
# one strategy
python -m eval.run_eval --strategy hybrid --reranker cross_encoder --k 8

# all strategies × rerankers → eval/results/comparison.csv + .json
python -m eval.compare

# re-print the latest stored results without re-running
python -m eval.compare --from-db
```

**Metrics:** field match (normalized), retrieval precision (is the answer in a retrieved snippet?),
faithfulness (is the answer grounded in context?), average confidence, latency, tokens, and cost.
Runs are deterministic (`temperature=0`, fixed `seed`) and stored in `eval/results/eval.db`.

---

## 📈 Example benchmark

Strategies × providers, evaluated on **3 Brazilian capital-markets documents** (75–238 pages each):
a real-estate securitization (CRI), a debenture indenture (Debêntures) and an agribusiness
securitization (CRA). Each is annotated with **9 fields** — issuer, issuer CNPJ, instrument type,
HQ city, issue/series numbers, fiduciary agent + CNPJ, issue date.
*The PDFs are not committed (`eval/dataset/*.pdf` is gitignored); they are referenced for reproducibility.*

**OpenAI** — `gpt-4.1-mini` + `text-embedding-3-small` (k = 8)

| Strategy | Accuracy | Retrieval prec. | Faithfulness | Latency | Cost |
|----------|---------:|----------------:|-------------:|--------:|-----:|
| dense    | 70% | 63% | 89% | 17.3s | $0.029 |
| **hybrid** | **93%** | **78%** | 85% | 20.5s | $0.031 |
| agentic  | 78% | 67% | 89% | 39.8s | $0.055 |

**Ollama** (local, free) — `llama3.2:3b` + `nomic-embed-text` (k = 8)

| Strategy | Accuracy | Retrieval prec. | Faithfulness | Latency | Cost |
|----------|---------:|----------------:|-------------:|--------:|-----:|
| dense    | 22% | 74% | 93% | 79.0s  | $0.00 |
| hybrid   | 30% | 78% | 81% | 74.8s  | $0.00 |
| agentic  | 22% | 74% | 93% | 121.8s | $0.00 |

### Finding 1 — retrieval, not prompt tuning, moved the needle

On long legal documents the first attempt scored poorly. Making the field *descriptions* more
detailed barely helped; **decoupling the retrieval query from the extraction prompt** (a short
natural-language question in the document's language) plus raising `k` from 5 → 8 transformed it.
OpenAI accuracy (field match) across iterations:

| Strategy | v1 (basic prompt) | v2 (detailed prompt) | v3 (+ retrieval query, k=8) |
|----------|------------------:|---------------------:|----------------------------:|
| dense    | 48% | 52% | **70%** |
| hybrid   | 56% | 52% | **93%** |
| agentic  | 56% | 67% | **78%** |

> Lesson: in RAG, *what you retrieve* often matters more than *how you prompt*. A long, detailed
> field description is great for the generator but a poor embedding query — keep them separate.

### Finding 2 — why hybrid wins

Hybrid recovered **6 fields that dense missed — all of them exact-token fields** (3 CNPJs,
3 legal entity names), and cut dense's "not found" answers from 6/27 to 1/27. Reason: a CNPJ
(`08.769.451/0001-08`) or a rare name (`Pentágono`) carries little *semantic* signal, so dense
embeddings blur it among look-alike chunks; **BM25 matches the literal tokens** in the
qualification clause and ranks it first. RRF fuses lexical precision with dense recall.
(The `agentic` retriever wraps the *dense* base, so it inherits this blind spot — an
"agentic-over-hybrid" base is promising future work.)

### Finding 3 — retrieval is solved locally; the small generator is the ceiling

With the improved queries, `nomic-embed-text` retrieves **as well as OpenAI** (retrieval precision
~74–78% on both). Yet local accuracy stays low: in **14 of 21 fields the right chunk was retrieved
but `llama3.2:3b` still failed** — returning `null`, the wrong instrument type, or even echoing the
instruction instead of the value. The bottleneck is purely the generator; a larger local model
(e.g. `qwen2.5:7b`) would close most of the gap.

> Illustrative benchmark (3 documents). Accuracy = exact-match on normalized values;
> faithfulness = answer grounded in the retrieved context (note it can be high while accuracy is
> low — "faithful but wrong").

**Reproduce** (drop your PDFs in `eval/dataset/` and annotate `eval/ground_truth.json`):

```bash
LLM_PROVIDER=openai EMBEDDING_PROVIDER=openai python -m eval.compare --strategies dense hybrid agentic --rerankers none --k 8
LLM_PROVIDER=ollama EMBEDDING_PROVIDER=ollama python -m eval.compare --strategies dense hybrid agentic --rerankers none --k 8
```

---

## 🛠️ Development

```bash
pip install -e ".[dev]"

ruff check .            # lint
ruff format .           # format
pytest                  # unit + integration tests (fully offline, LLM mocked)
```

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs ruff + pytest on every push and PR.
Dependencies are pinned in [`pyproject.toml`](pyproject.toml); the cross-encoder reranker
(`sentence-transformers`) lives behind the `reranker` extra to keep the base install lean.

---

## 📝 License

MIT — see [LICENSE](LICENSE).
