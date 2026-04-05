Oto gotowy kod Markdown. Wystarczy, że klikniesz przycisk "Kopiuj" w prawym górnym rogu poniższego bloku i wkleisz całość bezpośrednio do swojego pliku `README.md`:

```markdown
# TSL-RAG — EU Transport & Logistics Legal Assistant

A Retrieval-Augmented Generation (RAG) system designed to navigate, query, and cite European and Polish road transport laws. 

## The Problem
Transport law spans multiple overlapping regulations (e.g., EC 561/2006, AETR, Dir. 2002/15, and national penalty tariffs). A compliance officer asking *"Can a driver extend their daily rest if they're on a ferry?"* requires an answer that is legally accurate, cites the correct article, and respects document hierarchy. 

Generic RAG systems relying solely on flat vector search often fail to retrieve exact legal boundaries. This project implements a custom hybrid search and reranking pipeline to address these domain-specific challenges.

## Architecture

```text
     PDF Documents
           │
           ▼
┌─────────────────────────────────────────────┐
│              Ingestion Pipeline             │
│  LegalPDFParser → LegalChunker → Embedder   │
│  • pdfplumber (tables) + pymupdf (text)     │
│  • Article-aware chunking (400 tok max)     │
│  • Tables → atomic chunks (never split)     │
│  • nomic-embed-text → 768d vectors          │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
              PostgreSQL + pgvector
              (HNSW index, m=16)
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
     Dense Search   BM25 Search  Metadata
     (cosine sim)  (rank-bm25)   Filters
          │            │
          └─────┬──────┘
                ▼
         RRF Fusion
    (Reciprocal Rank Fusion)
                │
                ▼
     Cross-Encoder Reranker
  (ms-marco-MiniLM-L-6-v2, CPU)
                │
                ▼
        RAG Generator
   (Ollama llama3.1 8B / mistral 7B)
   Polish system prompt + citation enforcement
                │
                ▼
     FastAPI  ←→  Streamlit UI
```

## Tech Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| **Embedding** | `nomic-embed-text` (768d) | Free, local, strong multilingual support. |
| **Vector DB** | PostgreSQL + `pgvector` | Supports SQL joins, metadata filtering, and HNSW indexing. |
| **Keyword search** | `rank-bm25` (in-memory) | Low latency for smaller, highly specific legal corpora. |
| **Reranker** | `ms-marco-MiniLM-L-6-v2` | Lightweight cross-encoder (~90MB), runs efficiently on CPU. |
| **LLM** | `llama3.1:8b-instruct` | Runs locally via Ollama (~5GB VRAM required). |
| **API** | FastAPI + Uvicorn | Async handling, automatic OpenAPI documentation. |
| **UI** | Streamlit | Rapid prototyping with built-in retrieval debug panels. |
| **Eval** | Custom harness + Gemini Flash | LLM-as-a-Judge for semantic answer scoring. |

## Legal Corpus
The system is currently loaded with the following core documents:

| Document | Type | Description |
|----------|------|-------------|
| **EC_561_2006** | EU Regulation | Drivers' hours — the primary source. |
| **EU_2020_1054** | EU Regulation | Mobility Package amendments to EC 561/2006. |
| **DIRECTIVE_2002_15** | EU Directive | Working time for mobile road transport workers. |
| **DIRECTIVE_2020_1057** | EU Directive | Posting of drivers in the road transport sector. |
| **EU_165_2014** | EU Regulation | Rules regarding tachographs in road transport. |
| **EU_1071_2009** | EU Regulation | Conditions to pursue the occupation of road transport operator. |
| **EU_1072_2009** | EU Regulation | Common rules for access to the international road haulage market (Cabotage). |
| **AETR** | Int. Agreement | European Agreement concerning the Work of Crews of Vehicles engaged in International Road Transport. |
| **PL_DRIVER_HOURS_ACT** | National Law | Polish Act on the working time of drivers. |
| **TARIFF_DRIVER_2022** | Penalty Tariff | Fines for driver violations (PL). |
| **TARIFF_COMPANY_2022** | Penalty Tariff | Fines for transport company violations (PL). |
| **TARIFF_MANAGER_2022** | Penalty Tariff | Fines for transport manager violations (PL). |

## Evaluation
The system includes a custom evaluation harness running a golden dataset of 15 domain-specific questions. It supports two evaluation modes:

1. **Keyword Match (Offline):** Fast evaluation checking for the presence of exact expected strings (e.g., "9 godzin", "Art. 6").
2. **LLM-as-a-Judge (Semantic):** Uses Google's Gemini API to evaluate whether the generated answer conceptually matches the reference facts, even if phrased differently.

```bash
# Run keyword-based evaluation
uv run python -m evals.run_evals --output evals/results/run_001.json

# Run semantic evaluation (requires GEMINI_API_KEY)
uv run python -m evals.run_evals --use-judge --output evals/results/run_001_judge.json
```

## Quick Start

**Prerequisites:** Docker, Ollama, Python 3.11+, `uv`.

1. **Pull local models:**
   ```bash
   ollama pull llama3.1:8b-instruct-q4_K_M
   ollama pull nomic-embed-text
   ```

2. **Clone & install:**
   ```bash
   git clone [https://github.com/yourname/tsl-rag](https://github.com/yourname/tsl-rag)
   cd tsl-rag
   uv sync
   ```

3. **Configure environment:**
   ```bash
   cp env.example .env
   # Edit .env to add your GEMINI_API_KEY (optional, for evals only)
   ```

4. **Start the database:**
   ```bash
   docker compose up -d
   ```

5. **Ingest the documents:**
   *(Ensure PDF files are placed in `data/raw/` first)*
   ```bash
   uv run python -m tsl_rag.ingestion.cli ingest-all data/raw/
   ```

6. **Run the application:**
   Terminal 1 (Backend API):
   ```bash
   uv run python main.py
   ```
   Terminal 2 (Frontend UI):
   ```bash
   uv run streamlit run ui.py
   ```
   The UI will be available at `http://localhost:8501`.

## Design Decisions
* **Custom Pipeline vs. Orchestration Frameworks:** Frameworks like LangChain or LlamaIndex often obscure the retrieval mechanics. This project implements hybrid search, Reciprocal Rank Fusion (RRF), and cross-encoder reranking explicitly using standard Python libraries to maintain full control and testability.
* **Article-Boundary Chunking:** Legal text loses its meaning when split arbitrarily. The custom `LegalChunker` treats article boundaries as hard limits and avoids splitting tabular data (like penalty tariffs), ensuring the LLM receives complete, coherent legal thoughts.
* **PostgreSQL + pgvector:** Using `pgvector` within Docker removes the need for external SaaS vector databases, while allowing standard SQL joins and metadata filtering (e.g., filtering by `document_type`).

## Roadmap
- [ ] OCR pre-processing integration for scanned penalty tariff PDFs.
- [ ] Streaming generation responses in FastAPI and Streamlit.
- [ ] Integration tests with `pytest-asyncio` against a temporary test database.

## License
MIT
```
