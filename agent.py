import os
import logging
import chromadb
from rich.panel import Panel
from rich import print
from dotenv import load_dotenv
from google import genai
from evals import keyword_score, llm_judge
from shared.embeddings import get_embedding, content_fingerprint

logging.basicConfig(
    filename="studymind.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

load_dotenv(dotenv_path=".env", override=True)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("API key not found in agent.py")

client = genai.Client(api_key=api_key)

db_client = chromadb.PersistentClient(path="./db")
collection = db_client.get_or_create_collection(name="studymind")

GAP_DISTANCE_THRESHOLD = 0.8


def search_notes(query, n_results=3):
    """
    Converts the query into an embedding, finds the most
    semantically similar chunks stored in ChromaDB, and
    deduplicates overlapping results before returning.

    Deduplication handles the sliding-window chunking overlap:
    adjacent chunks share words, so retrieval can return near-
    identical content twice. We use a content hash fingerprint
    (more reliable than first-50-chars) to deduplicate.
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

    logging.info(
        f"Query: {question} | "
        f"Chunks retrieved: {len(results['documents'][0])} | "
        f"Distances: {results['distances'][0]}"
    )

    if not results["documents"][0]:
        logging.warning(f"No chunks found for query: {question}")
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


    if "email" in question.lower():
        expected_keywords = ["@", ".com"]
    elif "link" in question.lower() or "url" in question.lower():
        expected_keywords = ["http"]
    else:
        expected_keywords = []

    score = keyword_score(answer, expected_keywords)
    judge_result = llm_judge(client, question, answer, context)


    if not expected_keywords:
        confidence = "HIGH" if judge_result else "LOW"
    else:
        confidence = "HIGH" if score > 0.7 and judge_result else "LOW"

    logging.info(
        f"Answer generated | "
        f"Judge: {'PASS' if judge_result else 'FAIL'} | "
        f"Confidence: {confidence} | "
        f"Sources: {', '.join(sources)}"
    )

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
    Checks which topics from your syllabus are covered in
    your notes and which are missing.

    Uses a distance threshold to distinguish a genuine match
    from ChromaDB returning a result just because it has to —
    even when no chunk is actually about the topic.
    """
    print("\n[bold yellow]Checking your notes for gaps...[/bold yellow]\n")

    missing = []
    covered = []

    for topic in syllabus_topics:
        results = search_notes(topic, n_results=1)

        is_covered = (
            results["documents"][0]
            and results["distances"][0][0] < GAP_DISTANCE_THRESHOLD
        )

      
        logging.info(
            f"Gap check | Topic: {topic} | "
            f"Covered: {is_covered} | "
            f"Distance: {results['distances'][0][0] if results['documents'][0] else 'N/A'}"
        )

        if is_covered:
            covered.append(topic)
        else:
            missing.append(topic)

    print(Panel(
        "\n".join([f"[green]✓ {t}[/green]" for t in covered]) or "[dim]None[/dim]",
        title="[bold green]Topics Covered in Your Notes[/bold green]",
        border_style="green"
    ))

    print(Panel(
        "\n".join([f"[red]✗ {t}[/red]" for t in missing]) or "[dim]None[/dim]",
        title="[bold red]Topics Missing from Your Notes[/bold red]",
        border_style="red"
    ))