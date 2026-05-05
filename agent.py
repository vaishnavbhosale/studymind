import os
import chromadb
from rich.panel import Panel
from rich import print
from dotenv import load_dotenv
from google import genai
from evals import keyword_score, llm_judge


load_dotenv(dotenv_path=".env", override=True)

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("API key not found in agent.py")

client = genai.Client(api_key=api_key)

db_client = chromadb.PersistentClient(path="./db")
collection = db_client.get_or_create_collection(name="studymind")


def get_embedding(text):
    """
    Converts text into a list of numbers (embedding)
    that represents its meaning mathematically.
    Same model used for both storing and searching
    so similarity comparisons are consistent.
    """
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values


def search_notes(query, n_results=3):
    """
    Converts the question into an embedding,
    then finds the most semantically similar chunks
    stored in ChromaDB.

    Deduplication step added here to handle
    overlapping chunks from sliding window chunking.
    Without this, the same sentence could appear
    twice in the context sent to Gemini —
    wasting context window space and potentially
    confusing the model.
    """
    embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results
    )

    # --- Deduplication for sliding window overlap ---
    # Overlapping chunks share their opening words.
    # We use the first 50 characters as a fingerprint.
    # If two chunks start the same way, they're duplicates
    # and we only keep the first one.
    seen = set()
    unique_chunks = []
    unique_metadatas = []

    for doc, metadata in zip(
        results["documents"][0],
        results["metadatas"][0]
    ):
        fingerprint = doc[:50].strip()
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique_chunks.append(doc)
            unique_metadatas.append(metadata)

    # Replace results with deduplicated versions
    # so the rest of the code works exactly as before
    results["documents"][0] = unique_chunks
    results["metadatas"][0] = unique_metadatas

    return results


def ask(question):
    """
    Full RAG pipeline:
    1. Search notes for relevant chunks
    2. Build context from top chunks
    3. Send context + question to Gemini
    4. Evaluate the answer
    5. Display result with confidence score
    """
    print("\n[bold blue]Searching your notes...[/bold blue]")

    results = search_notes(question)

    if not results["documents"][0]:
        print("[red]No relevant content found in your notes.[/red]")
        return

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

    prompt = f"""You are a helpful study assistant. Answer the student's question 
ONLY using the provided excerpts from their notes. 

If the answer is not in the notes, say exactly: 
"I couldn't find this in your notes. You may need to check other sources."

Always mention which note/source your answer came from.

--- NOTES EXCERPTS ---
{context}
--- END OF EXCERPTS ---

Student's question: {question}

Answer:"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    answer = response.text

    # Keyword scoring — only used for specific question types
    # where exact tokens matter (emails, URLs).
    # For general questions expected_keywords is empty
    # so we rely entirely on the LLM judge.
    if "email" in question.lower():
        expected_keywords = ["@", ".com"]
    elif "link" in question.lower() or "url" in question.lower():
        expected_keywords = ["http"]
    else:
        expected_keywords = []

    score = keyword_score(answer, expected_keywords)
    judge_result = llm_judge(client, question, answer, context)

    # Confidence is HIGH only if LLM judge passes.
    # Keyword score is a secondary signal used only
    # when we have specific tokens to check for.
    if not expected_keywords:
        confidence = "HIGH" if judge_result else "LOW"
    else:
        confidence = "HIGH" if score > 0.7 and judge_result else "LOW"

    print(Panel.fit(
        f"{answer}\n\n"
        f"[bold yellow]Eval Score:[/bold yellow] {score}\n"
        f"[bold yellow]LLM Judge:[/bold yellow] {'PASS ✅' if judge_result else 'FAIL ❌'}\n"
        f"[bold cyan]Confidence:[/bold cyan] {confidence}",
        title="🤖 Answer from your notes",
        border_style="green"
    ))

    print(f"\n[dim]Sources used: {', '.join(sources)}[/dim]")


def find_gaps(syllabus_topics):
    """
    Checks which topics from your syllabus
    are covered in your notes and which are missing.
    Runs a search for each topic — if no chunks
    are returned, that topic is not in your notes.
    """
    print("\n[bold yellow]Checking your notes for gaps...[/bold yellow]\n")

    missing = []
    covered = []

    for topic in syllabus_topics:
        results = search_notes(topic, n_results=1)
        if results["documents"][0]:
            covered.append(topic)
        else:
            missing.append(topic)

    print(Panel(
        "\n".join([f"[green]✓ {t}[/green]" for t in covered]),
        title="[bold green]Topics Covered in Your Notes[/bold green]",
        border_style="green"
    ))

    print(Panel(
        "\n".join([f"[red]✗ {t}[/red]" for t in missing]),
        title="[bold red]Topics Missing from Your Notes[/bold red]",
        border_style="red"
    ))