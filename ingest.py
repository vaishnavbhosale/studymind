"""
ingest.py

PDF ingestion pipeline: reads PDFs, chunks text, embeds chunks,
and upserts into ChromaDB.

All paths (./db, ./notes) are resolved relative to this file's
directory — NOT the current working directory. Previously, using
"./db" caused two separate ChromaDB instances to be created
depending on where you launched the script from, meaning queries
would return empty results if you ran from the wrong folder.
"""

import os
import time
import PyPDF2
from google import genai
from rich import print
from rich.progress import track
from dotenv import load_dotenv
import chromadb

from shared.embeddings import get_embeddings_batch

# Resolve paths relative to this file so they work regardless of
# which directory the user launches the app from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"), override=True)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Add it to your .env file.")

client = genai.Client(api_key=api_key)

# Single shared DB path — all modules resolve to the same location
DB_PATH = os.path.join(BASE_DIR, "db")
db_client = chromadb.PersistentClient(path=DB_PATH)
collection = db_client.get_or_create_collection(name="studymind")


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file page by page.

    Note: PyPDF2 only handles text-layer PDFs.
    Scanned documents will return empty strings per page —
    consider adding an OCR fallback (e.g. pytesseract) for those.
    """
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n[Page {page_num + 1}]\n{page_text}"
    return text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Splits text into overlapping word-based chunks.

    chunk_size = how many words per chunk
    overlap    = how many words repeated between chunks

    Why overlap exists:
    If a concept is defined at word 490 of chunk 1,
    that definition also appears at the start of chunk 2.
    Retrieval always finds complete context regardless of
    which chunk it pulls.
    """
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += step
    return chunks


def ingest_pdf(pdf_path: str, display_name: str = None):
    """
    Ingests a single PDF into ChromaDB.

    display_name parameter added to fix a bug in app.py where
    Streamlit uploads to a temp path like /tmp/mynotes_abc123.pdf.
    Without this, os.path.basename(pdf_path) would store the garbled
    temp filename as the source metadata, making answers show wrong
    source names. Now callers can pass the original filename explicitly.
    """
    # Use display_name if provided, otherwise fall back to the actual filename.
    # This fixes the Streamlit temp-file naming bug.
    filename = display_name or os.path.basename(pdf_path)

    print(f"[bold green]Reading:[/bold green] {filename}")

    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print(f"[red]No text found in {filename}. It may be a scanned PDF.[/red]")
        return

    chunks = chunk_text(text)
    print(f"[blue]Split into {len(chunks)} chunks — embedding in batches...[/blue]")

    BATCH_SIZE = 50
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds

    all_embeddings = []

    for batch_start in track(
        range(0, len(chunks), BATCH_SIZE),
        description=f"Embedding {filename}..."
    ):
        batch = chunks[batch_start:batch_start + BATCH_SIZE]

        # Retry logic added: previously a single API blip would silently
        # drop entire chunks from the vector DB with no recovery attempt.
        # Now we retry up to MAX_RETRIES times before giving up on a batch.
        for attempt in range(MAX_RETRIES):
            try:
                batch_embeddings = get_embeddings_batch(client, batch)
                all_embeddings.extend(batch_embeddings)
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"[yellow]Retry {attempt + 1}/{MAX_RETRIES} for batch at chunk {batch_start}: {e}[/yellow]")
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"[red]Embedding failed after {MAX_RETRIES} attempts for batch at chunk {batch_start}: {e}[/red]")
                    all_embeddings.extend([None] * len(batch))

    for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
        if embedding is None:
            print(f"[red]Skipping chunk {i} due to embedding failure[/red]")
            continue

        collection.upsert(
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{"source": filename, "chunk": i}],
            ids=[f"{filename}_chunk_{i}"]
        )

    print(f"[green]✓ Ingested {filename}[/green]")


def ingest_all_notes():
    """
    Ingests all PDFs from the ./notes folder.
    Path is resolved relative to this file to avoid working-directory issues.
    """
    notes_folder = os.path.join(BASE_DIR, "notes")

    if not os.path.exists(notes_folder):
        print(f"[red]Folder '{notes_folder}' not found. Create it and add PDF files.[/red]")
        return

    pdf_files = [f for f in os.listdir(notes_folder) if f.endswith(".pdf")]

    if not pdf_files:
        print("[red]No PDF files found in /notes folder[/red]")
        return

    print(f"[bold]Found {len(pdf_files)} PDF file(s)[/bold]")
    for pdf_file in pdf_files:
        ingest_pdf(os.path.join(notes_folder, pdf_file))

    print("\n[bold green]All notes ingested! Ready to query.[/bold green]")


if __name__ == "__main__":
    ingest_all_notes()