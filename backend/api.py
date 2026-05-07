"""
api.py

FastAPI REST API for StudyMind.

RENAMED from backend/main.py to api.py for clarity.

Previously there were two files named main.py:
  - backend/main.py  (this file — FastAPI)
  - main.py          (CLI)
Running the wrong one started the wrong app silently.
Now the distinction is explicit: `uvicorn api:app` for the API server.

Run with:
    uvicorn api:app --reload
"""

import os
import sys

# ─── Path fix ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import FastAPI
from pydantic import BaseModel

from services.rag_service import answer_question

app = FastAPI(title="StudyMind API")


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "StudyMind API is running"}


@app.post("/ask")
def ask_question(payload: QuestionRequest):
    """
    Accepts a question and returns the full RAG result including
    answer, sources, confidence, and LLM judge verdict.
    """
    result = answer_question(payload.question)
    return result