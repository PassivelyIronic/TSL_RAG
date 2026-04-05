TSL-RAG — EU Transport & Logistics Legal AssistantA Retrieval-Augmented Generation (RAG) system designed to navigate, query, and cite European and Polish road transport laws.The ProblemTransport law spans multiple overlapping regulations (e.g., EC 561/2006, AETR, Dir. 2002/15, and national penalty tariffs). A compliance officer asking "Can a driver extend their daily rest if they're on a ferry?" requires an answer that is legally accurate, cites the correct article, and respects document hierarchy.Generic RAG systems relying solely on flat vector search often fail to retrieve exact legal boundaries. This project implements a custom hybrid search and reranking pipeline to address these domain-specific challenges.ArchitecturePlaintext     PDF Documents
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
Tech StackLayerTechnologyRationaleEmbeddingnomic-embed-text (768d)Free, local, strong multilingual support.Vector DBPostgreSQL + pgvectorSupports SQL joins, metadata filtering, and HNSW indexing.Keyword searchrank-bm25 (in-memory)Low latency for smaller, highly specific legal corpora.Rerankerms-marco-MiniLM-L-6-v2Lightweight cross-encoder (~90MB), runs efficiently on CPU.LLMllama3.1:8b-instructRuns locally via Ollama (~5GB VRAM required).APIFastAPI + UvicornAsync handling, automatic OpenAPI documentation.UIStreamlitRapid prototyping with built-in retrieval debug panels.EvalCustom harness + Gemini FlashLLM-as-a-Judge for semantic answer scoring.Legal CorpusThe system is currently loaded with the following core documents:DocumentTypeDescriptionEC_561_2006EU RegulationDrivers' hours — the primary source.EU_2020_1054EU RegulationMobility Package amendments to EC 561/2006.DIRECTIVE_2002_15EU DirectiveWorking time for mobile road transport workers.DIRECTIVE_2020_1057EU DirectivePosting of drivers in the road transport sector.EU_165_2014EU RegulationRules regarding tachographs in road transport.EU_1071_2009EU RegulationConditions to pursue the occupation of road transport operator.EU_1072_2009EU RegulationCommon rules for access to the international road haulage market (Cabotage).AETRInt. AgreementEuropean Agreement concerning the Work of Crews of Vehicles engaged in International Road Transport.PL_DRIVER_HOURS_ACTNational LawPolish Act on the working time of drivers.TARIFF_DRIVER_2022Penalty TariffFines for driver violations (PL).TARIFF_COMPANY_2022Penalty TariffFines for transport company violations (PL).TARIFF_MANAGER_2022Penalty TariffFines for transport manager violations (PL).EvaluationThe system includes a custom evaluation harness running a golden dataset of 15 domain-specific questions. It supports two evaluation modes:Keyword Match (Offline): Fast evaluation checking for the presence of exact expected strings (e.g., "9 godzin", "Art. 6").LLM-as-a-Judge (Semantic): Uses Google's Gemini API to evaluate whether the generated answer conceptually matches the reference facts, even if phrased differently.Bash# Run keyword-based evaluation
uv run python -m evals.run_evals --output evals/results/run_001.json

# Run semantic evaluation (requires GEMINI_API_KEY)
uv run python -m evals.run_evals --use-judge --output evals/results/run_001_judge.json
Quick StartPrerequisites: Docker, Ollama, Python 3.11+, uv.Pull local models:Bashollama pull llama3.1:8b-instruct-q4_K_M
ollama pull nomic-embed-text
Clone & install:Bashgit clone https://github.com/yourname/tsl-rag
cd tsl-rag
uv sync
Configure environment:Bashcp env.example .env
# Edit .env to add your GEMINI_API_KEY (optional, for evals only)
Start the database:Bashdocker compose up -d
Ingest the documents:(Ensure PDF files are placed in data/raw/ first)Bashuv run python -m tsl_rag.ingestion.cli ingest-all data/raw/
Run the application:Terminal 1 (Backend API):Bashuv run python main.py
Terminal 2 (Frontend UI):Bashuv run streamlit run ui.py
The UI will be available at http://localhost:8501.
