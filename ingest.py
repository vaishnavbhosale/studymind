import os
import PyPDF2
from google import genai
from rich import print
from rich.progress import track
from dotenv import load_dotenv
import chromadb

from shared.embeddings import get_embeddings_batch

load_dotenv(dotenv_path=".env", override=True)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("API key not found in ingest.py")

client = genai.Client(api_key=api_key)

db_client = chromadb.PersistentClient(path="./db")
collection = db_client.get_or_create_collection(name="studymind")


def extract_text_from_pdf(pdf_path):
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


def chunk_text(text, chunk_size=500, overlap=100):
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


def ingest_pdf(pdf_path):
    filename = os.path.basename(pdf_path)
    print(f"[bold green]Reading:[/bold green] {filename}")

    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print(f"[red]No text found in {filename}. It may be a scanned PDF.[/red]")
        return

    chunks = chunk_text(text)
    print(f"[blue]Split into {len(chunks)} chunks — embedding in batches...[/blue]")

    # --- Batch embedding ---
    # One API call for all chunks instead of N sequential calls.
    # Dramatically faster for large PDFs.
    BATCH_SIZE = 50  # stay within API limits
    all_embeddings = []

    for batch_start in track(
        range(0, len(chunks), BATCH_SIZE),
        description=f"Embedding {filename}..."
    ):
        batch = chunks[batch_start:batch_start + BATCH_SIZE]
        try:
            batch_embeddings = get_embeddings_batch(client, batch)
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"[red]Embedding failed for batch starting at chunk {batch_start}: {e}[/red]")
            # Fill with None so indices stay aligned with chunks
            all_embeddings.extend([None] * len(batch))

    # --- Upsert instead of add ---
    # collection.add() throws if a chunk ID already exists (e.g. re-ingesting the same file).
    # collection.upsert() safely overwrites — making ingest idempotent.
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
    notes_folder = "./notes"

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