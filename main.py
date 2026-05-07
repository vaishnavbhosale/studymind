"""
main.py

CLI entrypoint for StudyMind.

Previously there were TWO main.py files:
  - backend/main.py  (FastAPI)
  - main.py          (CLI, this file)

The FastAPI app has been moved to api.py to eliminate the ambiguity.
Running the wrong main.py silently started the wrong app with no error,
which was confusing. Now:
  - python main.py          → CLI interface
  - uvicorn api:app         → FastAPI server
"""

import os
import sys

# ─── Path fix ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
# ─────────────────────────────────────────────────────────────────────────────

from rich import print
from rich.panel import Panel
from rich.prompt import Prompt

from agent import ask, find_gaps
from ingest import ingest_all_notes
from dotenv import load_dotenv

load_dotenv(override=True)


def show_welcome():
    print(Panel(
        "[bold white]StudyMind — Your Personal Knowledge Agent[/bold white]\n"
        "[dim]Ask questions. Get answers from YOUR notes only.[/dim]",
        border_style="blue"
    ))


def main():
    show_welcome()

    while True:
        print("\n[bold]What do you want to do?[/bold]")
        print("[1] Upload and index my notes")
        print("[2] Ask a question")
        print("[3] Find gaps in my notes")
        print("[4] Exit")

        choice = Prompt.ask("\nChoose", choices=["1", "2", "3", "4"])

        if choice == "1":
            print("\n[yellow]Put your PDF files in the /notes folder first.[/yellow]")
            confirm = Prompt.ask("Ready?", choices=["yes", "no"])
            if confirm == "yes":
                ingest_all_notes()

        elif choice == "2":
            question = Prompt.ask("\n[bold blue]Your question[/bold blue]")
            ask(question)

        elif choice == "3":
            print("\n[yellow]Enter your syllabus topics one by one.[/yellow]")
            print("[dim]Press Enter with empty input when done.[/dim]\n")
            topics = []
            while True:
                topic = Prompt.ask("Topic (or press Enter to finish)")
                if not topic:
                    break
                topics.append(topic)
            if topics:
                find_gaps(topics)
            else:
                print("[yellow]No topics entered.[/yellow]")

        elif choice == "4":
            print("\n[bold green]Goodbye! Good luck with your studies.[/bold green]")
            break


if __name__ == "__main__":
    main()