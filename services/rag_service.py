"""
services/rag_service.py

Core RAG (Retrieval-Augmented Generation) service layer.

This module owns:
  - search_notes()     — vector search against ChromaDB
  - answer_question()  — full RAG pipeline (retrieve → prompt → judge)

WHY search_notes() MOVED HERE:
Previously search_notes() lived in agent.py, and rag_service.py
imported it from agent.py. But agent.py also imported answer_question()
from rag_service.py. That created a circular import that crashed Python
at startup before any code ran.

The fix: move search_notes() here so the dependency chain is linear:
  app.py / agent.py / main.py
       → services.rag_service
           → shared.embeddings
           → evals

No module in this chain imports back up the chain.
"""

import os
import sys
import logging

from google import genai

from google import genai

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
# ─────────────────────────────────────────────────────────────────────────────

import chromadb
from dotenv import load_dotenv
from groq import Groq
from evals import keyword_score, llm_judge
from shared.embeddings import get_embedding, content_fingerprint
ENABLE_EVALS = False

logging.basicConfig(
    filename=os.path.join(BASE_DIR, "studymind.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"), override=True)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Add it to your .env file.")

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)
groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# All modules use the same absolute DB path so they connect to the
# same ChromaDB instance regardless of working directory.
DB_PATH = os.path.join(BASE_DIR, "db")
db_client = chromadb.PersistentClient(path=DB_PATH)
collection = db_client.get_or_create_collection(name="studymind")

GAP_DISTANCE_THRESHOLD = 0.8


def search_notes(query: str, n_results: int = 3) -> dict:
    """
    Converts the query into an embedding, finds the most
    semantically similar chunks stored in ChromaDB, and
    deduplicates overlapping results before returning.

    Deduplication handles the sliding-window chunking overlap:
    adjacent chunks share words, so retrieval can return near-
    identical content twice. We use a content hash fingerprint
    (more reliable than first-50-chars) to deduplicate.

    MOVED FROM agent.py to break the circular import:
    agent.py was importing this, and rag_service.py was importing
    answer_question from agent.py — a cycle Python cannot resolve.
    """
    embedding = get_embedding(client, query)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    seen = set()
    unique_chunks = []
    unique_metadatas = []
    unique_distances = []

    for doc, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        fingerprint = content_fingerprint(doc)
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique_chunks.append(doc)
            unique_metadatas.append(metadata)
            unique_distances.append(distance)

    results["documents"][0] = unique_chunks
    results["metadatas"][0] = unique_metadatas
    results["distances"][0] = unique_distances

    return results

def answer_question(question: str):

    results = search_notes(question)

    if not results["documents"][0]:
        return {
            "success": False,
            "error": "No relevant content found in your notes."
        }

    context_parts = []
    sources = []

    for i, (doc, metadata) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0]
    )):
        context_parts.append(
            f"[Excerpt {i+1} from {metadata['source']}]\n{doc}"
        )
        source = metadata["source"]
        if source not in sources:
            sources.append(source)

    context = "\n\n".join(context_parts)

    prompt = f"""You are a helpful study assistant.

ONLY answer using the provided excerpts.

If the answer is not found, say exactly:
"I couldn't find this in your notes. You may need to check other sources."

Always mention the source file name when answering.

--- NOTES ---
{context}
--- END NOTES ---

Question:
{question}

Answer:"""

    response = groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

    answer = response.choices[0].message.content

    if "email" in question.lower():
        expected_keywords = ["@", ".com"]
    elif "link" in question.lower() or "url" in question.lower():
        expected_keywords = ["http"]
    else:
        expected_keywords = []

    score = keyword_score(answer, expected_keywords)

    if ENABLE_EVALS:

        judge_result = llm_judge(
            client,
            question,
            answer,
            context
        )

        if not expected_keywords:
            confidence = "HIGH" if judge_result else "LOW"

        else:
            confidence = (
                "HIGH"
                if score > 0.7 and judge_result
                else "LOW"
            )

    else:

        judge_result = True
        confidence = "HIGH"

    logging.info(
        f"Q: {question[:80]} | Judge: {judge_result} | "
        f"Confidence: {confidence} | Sources: {sources}"
    )

    return {
        "success": True,
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "judge_result": judge_result,
        "score": score
    }