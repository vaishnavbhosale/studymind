# StudyMind 🧠
### A RAG-Based Personal Knowledge Agent for Students

---

## The Problem

As a student, I had lecture notes, PDFs, and slides scattered across multiple folders — but no way to query across them intelligently.

When I needed to revise a concept the night before an exam, I had two bad options:

- Search through 20 files manually
- Ask ChatGPT — which would hallucinate answers not grounded in my actual notes

This is the same problem enterprises face with their data — scattered across warehouses, dashboards, and pipelines with no unified, trusted context layer.

**Atlan solves this for enterprises. StudyMind solves this for students.**

---

## What It Does

StudyMind is a RAG-based AI agent that:

- Ingests your lecture notes and PDFs into a local vector database
- Answers questions grounded **only** in your actual notes — never the internet
- Cites which source the answer came from (like data lineage)
- Detects hallucinations using a multi-metric LLM-as-judge evaluator
- Identifies gaps between your notes and your syllabus topics

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

## Architecture

```
PDF Notes
    ↓
[ingest.py]  →  Extract text → Chunk with overlap → Batch embed → Store in ChromaDB
    ↓
[ChromaDB]   →  Local vector database (semantic similarity search)
    ↓
[agent.py]   →  Embed question → Retrieve top chunks → Build prompt → Ask Gemini
    ↓
[evals.py]   →  LLM Judge (Faithfulness + Relevance + Completeness) → Confidence score
    ↓
Answer grounded in your notes, with source citation
```

---

## Key Engineering Decisions

### 1. Sliding Window Chunking with Overlap
```python
chunk_size=500, overlap=100
```
Concepts that fall on chunk boundaries aren't lost — they appear in both the ending chunk and the start of the next one. This ensures retrieval always finds complete context.

### 2. Batch Embedding
Instead of one API call per chunk (N round trips), chunks are embedded in batches of 50. Reduces API calls by 98% for large PDFs.

### 3. Idempotent Ingestion
Using `collection.upsert()` instead of `collection.add()` — re-ingesting the same PDF never crashes or creates duplicates.

### 4. Content Hash Deduplication
Overlapping chunks can return near-identical results. We deduplicate using MD5 hashes of full chunk content — more reliable than prefix matching.

### 5. Multi-Metric LLM Judge
Three-dimensional evaluation instead of a simple yes/no:
- **Faithfulness** — is every claim supported by the context?
- **Relevance** — does the answer address the question?
- **Completeness** — are key points covered?

### 6. Distance Threshold for Gap Detection
ChromaDB always returns *something* even when nothing is relevant. A cosine distance threshold of `0.8` prevents false positives in gap detection.

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core language |
| Google Gemini API (`gemini-2.5-flash`) | LLM for generation |
| Google Gemini API (`gemini-embedding-001`) | Text embeddings |
| ChromaDB | Local vector database |
| PyPDF2 | PDF text extraction |
| Streamlit | Web UI |
| Rich | Terminal UI |

---

## Project Structure

```
studymind/
├── shared/
│   ├── __init__.py
│   └── embeddings.py      # Shared embedding logic (single source of truth)
├── notes/                 # Drop your PDFs here
├── db/                    # ChromaDB storage (auto-created)
├── agent.py               # RAG pipeline — search + generate + evaluate
├── app.py                 # Streamlit web UI
├── evals.py               # LLM-as-judge evaluation system
├── ingest.py              # PDF ingestion pipeline
├── main.py                # Terminal UI
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

### 4. Add your Gemini API key
Create a `.env` file in the root:
```
GEMINI_API_KEY=your_key_here
```

### 5. Add your PDFs
Drop any PDF notes into the `/notes` folder.

### 6. Run — Web UI (recommended)
```bash
streamlit run app.py
```

### 6. Run — Terminal UI
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
| 📊 Confidence Score | Every answer comes with an LLM judge evaluation |
| 📄 Source Citation | Always shows which note the answer came from |

---

## Potential Improvements

- **OCR support** — PyPDF2 can't read scanned PDFs; add `pytesseract` as fallback
- **Hybrid search** — combine vector similarity with BM25 keyword search for better retrieval
- **Reranking** — add a cross-encoder reranker after initial retrieval for higher precision
- **Streaming responses** — show answers word by word instead of waiting for full response
- **Persistent cache** — cache embeddings so re-ingesting unchanged files skips API calls
- **Cloud scale** — swap ChromaDB for Pinecone/Weaviate + FastAPI backend for multi-user support

---

## Built By

**Vaishnav Bhosale** — Built as part of preparation for Atlan's AI Native Builder Internship (June 2026)

> *"The same way Atlan gives AI agents a trusted, governed context layer over enterprise data — StudyMind gives AI a trusted, grounded context layer over student notes."*
