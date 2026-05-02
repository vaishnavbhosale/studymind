import os
import chromadb
from rich.panel import Panel
from rich import print
from dotenv import load_dotenv
from google import genai
from evals import keyword_score, llm_judge  # ✅ evals import

# 🔥 Load environment
load_dotenv(dotenv_path=".env", override=True)

api_key = os.getenv("GEMINI_API_KEY")
print("AGENT KEY:", api_key)

if not api_key:
    raise ValueError("API key not found in agent.py")

client = genai.Client(api_key=api_key)

# Setup ChromaDB
db_client = chromadb.PersistentClient(path="./db")
collection = db_client.get_or_create_collection(name="studymind")


def get_embedding(text):
    """Get embedding using Gemini"""
    result = client.models.embed_content(
        model="gemini-embedding-001",  # or gemini-embedding-2
        contents=text
    )
    return result.embeddings[0].values


def search_notes(query, n_results=3):
    """Search your notes for relevant chunks"""
    embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results
    )
    return results


def ask(question):
    """Ask a question and get answer from your notes"""
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

    # 🤖 Generate answer
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    answer = response.text

    # 🧠 Decide eval keywords dynamically
    if "email" in question.lower():
        expected_keywords = ["@", ".com"]
    elif "link" in question.lower() or "url" in question.lower():
        expected_keywords = ["http"]
    else:
        expected_keywords = []

    # ✅ Run evals
    score = keyword_score(answer, expected_keywords)
    judge_result = llm_judge(client, question, answer, context)

    # 🔥 Confidence logic
    confidence = "HIGH" if score > 0.7 and judge_result else "LOW"

    # 🎯 Print final output
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
    """Find what topics are missing from your notes"""
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