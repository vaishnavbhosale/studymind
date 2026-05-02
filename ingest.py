import os
import chromadb
import PyPDF2
from google import genai
from rich import print
from dotenv import load_dotenv

# 🔥 FIXED loading
load_dotenv(dotenv_path=".env", override=True)

api_key = os.getenv("GEMINI_API_KEY")
print("INGEST KEY:", api_key)

if not api_key:
    raise ValueError("API key not found in ingest.py")

client = genai.Client(api_key=api_key)

# Setup ChromaDB
db_client = chromadb.PersistentClient(path="./db")
collection = db_client.get_or_create_collection(name="studymind")

def get_embedding(text):
    result = client.models.embed_content(
       model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n[Page {page_num + 1}]\n{page_text}"
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0

    for word in words:
        current_chunk.append(word)
        current_size += 1
        if current_size >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_size = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def ingest_pdf(pdf_path):
    filename = os.path.basename(pdf_path)
    print(f"[bold green]Reading:[/bold green] {filename}")

    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print(f"[red]No text found in {filename}[/red]")
        return

    chunks = chunk_text(text)
    print(f"[blue]Split into {len(chunks)} chunks[/blue]")

    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)

        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{"source": filename, "chunk": i}],
            ids=[f"{filename}_chunk_{i}"]
        )

    print(f"[bold green]Done! {filename} ingested successfully.[/bold green]")

def ingest_all_notes():
    notes_folder = "./notes"
    pdf_files = [f for f in os.listdir(notes_folder) if f.endswith(".pdf")]

    if not pdf_files:
        print("[red]No PDF files found in /notes folder[/red]")
        return

    print(f"[bold]Found {len(pdf_files)} PDF files[/bold]")
    for pdf_file in pdf_files:
        ingest_pdf(os.path.join(notes_folder, pdf_file))

    print("\n[bold green]All notes ingested! Ready to query.[/bold green]")

if __name__ == "__main__":
    ingest_all_notes()