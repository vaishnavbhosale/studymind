"""
agent.py

CLI presentation layer — handles only terminal output formatting.
All business logic (search, RAG, evaluation) lives in services/rag_service.py.

WHY THIS FILE WAS REFACTORED:
Previously agent.py defined search_notes() and imported answer_question()
from rag_service.py. But rag_service.py imported search_notes() from here.
That circular import crashed Python before any function ever ran.

Fix: search_notes() moved to rag_service.py. agent.py now only imports
from rag_service (one-way dependency), and adds the CLI presentation on top.
"""

import logging
from rich.panel import Panel
from rich import print

from services.rag_service import answer_question, search_notes

GAP_DISTANCE_THRESHOLD = 0.8

logging.basicConfig(
    filename="studymind.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


def ask(question: str):
    """
    CLI wrapper around the reusable RAG service layer.
    Handles only terminal presentation logic — no RAG code here.
    """
    print("\n[bold blue]Searching your notes...[/bold blue]")

    result = answer_question(question)

    if not result["success"]:
        print(f"[red]{result['error']}[/red]")
        return

    print(Panel.fit(
        f"{result['answer']}\n\n"
        f"[bold yellow]LLM Judge:[/bold yellow] "
        f"{'PASS ✅' if result['judge_result'] else 'FAIL ❌'}\n"
        f"[bold cyan]Confidence:[/bold cyan] "
        f"{result['confidence']}",
        title="🤖 Answer from your notes",
        border_style="green"
    ))

    print(
        f"\n[dim]Sources used: "
        f"{', '.join(result['sources'])}[/dim]"
    )


def find_gaps(syllabus_topics: list[str]):
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