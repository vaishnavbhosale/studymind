# StudyMind 🧠
### A Personal Knowledge Agent for Students

## The Problem
As a student, I had lecture notes, PDFs, and slides scattered across 
multiple folders — but no way to query across them. When I needed to 
revise a concept at midnight, I had two bad options:
- Search through 20 files manually
- Ask ChatGPT — which would hallucinate answers not from my actual notes

This is the same problem enterprises face with their data — scattered 
across warehouses, dashboards, and pipelines with no unified context layer.
Atlan solves this for enterprises. StudyMind solves this for students.

## What It Does
StudyMind is a RAG-based AI agent that:
- Ingests your lecture notes and PDFs into a local vector database
- Answers your questions grounded ONLY in your actual notes
- Cites which note/source the answer came from — like data lineage
- Identifies gaps in your notes vs your syllabus topics

## Why This Matters
Every answer is grounded in your actual content — not the internet.
This eliminates hallucination and builds trust in the AI's responses.
Just like Atlan's context layer gives AI agents trusted, governed data
to reason over — StudyMind gives the AI only your verified notes.

## The Parallel to Enterprise AI
| Student Problem | Enterprise Problem (Atlan) |
|---|---|
| Notes scattered across PDFs | Data scattered across warehouses |
| "Which note explains revenue?" | "Which table contains revenue data?" |
| AI hallucinating wrong answers | AI agents reasoning on bad data |
| Need grounded, cited answers | Need trusted, governed AI outputs |

## Tech Stack
- **Python** — core language
- **Google Gemini API** — LLM for answering + embeddings
- **ChromaDB** — local vector database for storing note embeddings
- **PyPDF2** — PDF text extraction
- **Rich** — terminal UI

## How It Works
1. You drop PDFs into the `/notes` folder
2. `ingest.py` extracts text, splits into chunks, converts to embeddings,
   stores in ChromaDB
3. When you ask a question, it's converted to an embedding and matched
   against your notes using vector similarity search
4. The top matching chunks are sent to Gemini as context
5. Gemini answers ONLY from that context — grounded, cited, trusted

## How to Run

### 1. Clone the repo
git clone https://github.com/vaishnavbhosale/studymind.git
cd studymind

### 2. Install dependencies
pip install -r requirements.txt

### 3. Add your Gemini API key
Create a .env file:
GEMINI_API_KEY=your_key_here

### 4. Add your PDFs
Drop any PDF notes into the /notes folder

### 5. Run
python main.py

## Features
- Option 1: Index your notes
- Option 2: Ask any question — answered from your notes only
- Option 3: Find gaps — check which syllabus topics are missing from notes

## Built By
Vaishnavv — Built as part of preparation for Atlan's AI Native 
Builder Internship (June 2026)
