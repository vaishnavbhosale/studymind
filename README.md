# StudyMind 🧠
### A RAG-Based Personal Knowledge Agent for Students

[![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Local_Vector_DB-orange?style=for-the-badge)](https://www.trychroma.com)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.3-blueviolet?style=for-the-badge)](https://groq.com)

---

## The Problem

As a student, I had lecture notes, PDFs, and slides scattered across multiple folders — but no way to query across them intelligently.

When I needed to revise a concept the night before an exam, I had two bad options:

- Search through 20 files manually
- Ask ChatGPT — which would hallucinate answers not grounded in my actual notes

This is the same problem enterprises face with their data — scattered across warehouses, dashboards, and pipelines with no unified, trusted context layer.

**Atlan solves this for enterprises. StudyMind solves this for students.**

---

## The Parallel to Enterprise AI

| Student Problem | Enterprise Problem (Atlan) |
|---|---|
| Notes scattered across PDFs | Data scattered across warehouses |
| "Which note explains this concept?" | "Which table contains revenue data?" |
| AI hallucinating wrong answers | AI agents reasoning on bad/ungoverned data |
| Need grounded, cited answers | Need trusted, governed AI outputs |
| Gap detection vs syllabus | Data coverage vs business requirements |

---

## What It Does

StudyMind is a RAG-based AI agent that:

- Ingests lecture notes and PDFs into a local vector database
- Answers questions grounded **only** in your actual notes — never the internet
- Cites which source the answer came from (like data lineage)
- Detects hallucinations using a multi-metric LLM-as-judge evaluator
- Identifies gaps between your notes and your syllabus topics
- Exposes the full pipeline via a **FastAPI backend** for scalable access

---

## Architecture

```
PDF Notes
    ↓
[ingest.py]  →  Extract text → Chunk with overlap → Batch embed → Store in ChromaDB
    ↓
[ChromaDB]   →  Local vector database (semantic similarity search)
    ↓
[services/rag_service.py]
    ├── semantic retrieval
    ├── prompt orchestration
    ├── answer generation (Groq + Llama 3.3)
    └── LLM-as-judge evaluation
    ↓
[Presentation Layers]
    ├── app.py      →  Streamlit web UI
    ├── main.py     →  Terminal CLI
    └── api.py      →  FastAPI backend (REST)
```

The key architectural decision: **all RAG logic lives in `rag_service.py`**. The three presentation layers (UI, CLI, API) all call the same service — no duplicated logic, no tight coupling.

---

## Architecture Evolution

StudyMind started as a tightly coupled Streamlit prototype — retrieval, prompting, evaluation, and UI logic all lived in a single file. As the system grew, this became unmaintainable.

The architecture was refactored in three phases:

**Phase 1 — Extract the service layer**
`rag_service.py` was created as the single orchestration layer. All retrieval, prompt building, generation, and evaluation logic moved here. `agent.py` became a pure CLI presentation layer.

**Phase 2 — Add a REST API layer**
`api.py` exposed the RAG pipeline via FastAPI, making the system accessible beyond Streamlit — from scripts, other services, or future frontends.

**Phase 3 — Decouple providers**
Embeddings (Gemini) and generation (Groq + Llama 3.3) were separated into distinct responsibilities. Swapping LLM providers now requires changing one variable, not rewriting prompt logic.

**What this refactor improved:**
- Eliminated circular import bugs caused by tight coupling
- Single source of truth for RAG logic — fix once, works everywhere
- Multi-interface reuse — same pipeline, three entry points
- Provider flexibility — Gemini for embeddings, Groq for generation
- Maintainability — each file has one clear responsibility

---

## Key Engineering Decisions

### 1. Sliding Window Chunking with Overlap
```python
chunk_size=500, overlap=100
```
Concepts that fall on chunk boundaries aren't lost — they appear in both the ending chunk and the start of the next one. Ensures retrieval always finds complete context.

### 2. Batch Embedding
Instead of one API call per chunk (N round trips), chunks are embedded in batches of 50. Reduces API calls by ~98% for large PDFs — critical for staying within free-tier rate limits.

### 3. Idempotent Ingestion
Using `collection.upsert()` instead of `collection.add()` — re-ingesting the same PDF never crashes or creates duplicates. The pipeline is safe to re-run at any time.

### 4. Content Hash Deduplication
Overlapping chunks can return near-identical results. Deduplicated using MD5 hashes of full chunk content — more reliable than prefix matching or similarity thresholds.

### 5. Multi-Metric LLM Judge
Three-dimensional evaluation instead of a simple yes/no confidence score:
- **Faithfulness** — is every claim supported by the retrieved context?
- **Relevance** — does the answer actually address the question asked?
- **Completeness** — are all key points from the context covered?

### 6. Distance Threshold for Gap Detection
ChromaDB always returns *something* even when nothing relevant exists. A cosine distance threshold of `0.8` filters out false positives in gap detection — without it, every topic would appear "covered."

### 7. Separated Embedding and Generation Providers
Gemini handles embeddings (`gemini-embedding-001`) and Groq handles generation (`llama-3.3-70b`). This separates cost, latency, and rate limit concerns — and means either provider can be swapped independently.

---

## Challenges Faced

These were the real engineering problems — not just "it worked first try."

**Circular imports** — When `api.py` and `app.py` both imported from `agent.py`, and `agent.py` imported from `ingest.py`, Python circular import errors broke the startup. Fixed by extracting all shared logic into `rag_service.py` — a clean dependency direction with no cycles.

**Quota exhaustion** — Embedding large PDFs with one API call per chunk hit Gemini's free-tier rate limits instantly. Fixed with batch embedding (50 chunks per call) and exponential backoff on retries.

**Vector DB drift** — During development, re-ingesting modified PDFs created duplicate entries in ChromaDB, causing retrieval to return stale or conflicting chunks. Fixed with `upsert()` and a persistent DB path with a manual reset flag.

**Chunk overlap duplication** — Overlapping chunks caused the same sentence to appear twice in retrieved context, inflating the prompt and confusing the evaluator. Fixed with MD5 content-hash deduplication before passing context to the LLM.

**Evaluation cost tradeoffs** — Running the LLM judge on every query tripled API costs. Added a `--no-eval` flag to skip evaluation in CLI mode for quick lookups, while keeping full evaluation in the Streamlit UI.

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Core language |
| Google Gemini (`gemini-embedding-001`) | Text embeddings |
| Groq + Llama 3.3 70B | Answer generation (fast inference) |
| ChromaDB | Local vector database |
| FastAPI | REST API backend |
| Streamlit | Web UI |
| PyPDF2 | PDF text extraction |
| Rich | Terminal UI formatting |

---

## Project Structure

```
studymind/
├── services/
│   └── rag_service.py     # Core RAG pipeline — retrieval, generation, evaluation
├── shared/
│   ├── __init__.py
│   └── embeddings.py      # Shared embedding logic (single source of truth)
├── notes/                 # Drop your PDFs here
├── db/                    # ChromaDB storage (auto-created)
├── agent.py               # CLI presentation layer
├── api.py                 # FastAPI backend
├── app.py                 # Streamlit web UI
├── evals.py               # LLM-as-judge evaluation system
├── ingest.py              # PDF ingestion pipeline
├── main.py                # Terminal entry point
├── .env                   # API keys (never committed)
└── requirements.txt
```

---

## How to Run

### 1. Clone the repo
```bash
git clone https://github.com/vaishnavbhosale/studymind.git
cd studymind
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your API keys
Create a `.env` file in the root:
```
GEMINI_API_KEY=your_gemini_key_here
GROQ_API_KEY=your_groq_key_here
```

### 5. Add your PDFs
Drop any PDF notes into the `/notes` folder.

### 6. Ingest your notes
```bash
python ingest.py
```

### 7. Run — Web UI (recommended)
```bash
streamlit run app.py
```

### 7. Run — FastAPI Backend
```bash
uvicorn api:app --reload
```
Interactive Swagger docs: `http://127.0.0.1:8000/docs`

### 7. Run — Terminal CLI
```bash
python main.py
```

---

## Features

| Feature | Description |
|---|---|
| 📥 Ingest Notes | Upload PDFs and index them into ChromaDB |
| 💬 Ask Questions | Get answers grounded only in your notes |
| 🔍 Find Gaps | Check which syllabus topics are missing from your notes |
| 📊 Confidence Score | Every answer comes with an LLM judge evaluation (Faithfulness + Relevance + Completeness) |
| 📄 Source Citation | Always shows which note the answer came from |
| 🔌 REST API | FastAPI backend exposes the full pipeline programmatically |

---

## Potential Improvements

- **OCR support** — PyPDF2 can't read scanned PDFs; add `pytesseract` as fallback
- **Hybrid search** — combine vector similarity with BM25 keyword search for better retrieval
- **Reranking** — add a cross-encoder reranker after initial retrieval for higher precision
- **Streaming responses** — show answers word by word instead of waiting for full generation
- **Persistent cache** — cache embeddings so re-ingesting unchanged files skips API calls
- **Cloud scale** — swap ChromaDB for Pinecone/Weaviate for multi-user, cloud-hosted deployment

---

## Built By

**Vaishnav Bhosale** — Built as part of preparation for Atlan's AI Native Builder Internship (June 2026)

> *"The same way Atlan gives AI agents a trusted, governed context layer over enterprise data — StudyMind gives AI a trusted, grounded context layer over student notes."*
