"""
main.py
────────
CLI entrypoint for the Hybrid RAG pipeline.

Commands:
    ingest  — Load docs, chunk, embed, index into ChromaDB + BM25
    query   — Run a single question through the hybrid RAG pipeline
    chat    — Interactive REPL mode
"""

import argparse
import logging
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from config.settings import settings
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunker import RecursiveChunker
from src.ingestion.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.rag_pipeline import RAGPipeline

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
console = Console()


# ── Helper: build the full pipeline ──────────────────────────────────────────

def build_pipeline() -> RAGPipeline:
    embedder = Embedder()
    vector_store = VectorStore(embedder)
    bm25 = BM25Retriever()
    bm25.load()          # no-op if index doesn't exist yet
    retriever = HybridRetriever(vector_store, bm25)
    return RAGPipeline(retriever)


# ── Command: ingest ───────────────────────────────────────────────────────────

def cmd_ingest(args: argparse.Namespace) -> None:
    console.rule("[bold green]📥 Ingestion")

    # Load
    console.print(f"Loading documents from [cyan]{args.docs_dir}[/cyan]…")
    loader = DocumentLoader(args.docs_dir)
    documents = loader.load()
    console.print(f"✅ Loaded [bold]{len(documents)}[/bold] document(s).")

    # Chunk
    chunker = RecursiveChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = chunker.split(documents)
    console.print(f"✅ Created [bold]{len(chunks)}[/bold] chunks.")

    # Embed + Vector Store
    console.print("Embedding and indexing into ChromaDB…")
    embedder = Embedder()
    vector_store = VectorStore(embedder)
    if args.reset:
        vector_store.reset()
    vector_store.add_chunks(chunks)
    console.print(f"✅ VectorStore has [bold]{vector_store.count()}[/bold] docs.")

    # BM25
    console.print("Building BM25 index…")
    bm25 = BM25Retriever()
    bm25.build(chunks)
    console.print("✅ BM25 index ready.")

    console.rule("[bold green]✅ Ingestion Complete")


# ── Command: query ────────────────────────────────────────────────────────────

def cmd_query(args: argparse.Namespace) -> None:
    pipeline = build_pipeline()
    _run_query(pipeline, args.question)


def _run_query(pipeline: RAGPipeline, question: str) -> None:
    console.rule(f"[bold blue]🔍 Query")
    console.print(f"[bold]Q:[/bold] {question}\n")

    with console.status("Retrieving and generating…"):
        response = pipeline.run(question)

    # Answer panel
    console.print(Panel(
        response.answer,
        title="[bold green]Answer",
        border_style="green",
        padding=(1, 2),
    ))

    # Sources table
    if response.sources:
        table = Table(title="Retrieved Sources", show_lines=True)
        table.add_column("#", style="dim", width=4)
        table.add_column("Source", style="cyan")
        table.add_column("RRF Score", justify="right", style="yellow")

        for i, chunk in enumerate(response.chunks, start=1):
            table.add_row(
                str(i),
                chunk.metadata.get("source", "unknown"),
                f"{chunk.score:.4f}",
            )
        console.print(table)


# ── Command: chat ─────────────────────────────────────────────────────────────

def cmd_chat(args: argparse.Namespace) -> None:
    console.rule("[bold magenta]💬 Hybrid RAG Chat")
    console.print("Type your question and press Enter. Type [bold]exit[/bold] to quit.\n")

    pipeline = build_pipeline()

    while True:
        try:
            question = console.input("[bold yellow]You:[/bold yellow] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            console.print("[dim]Goodbye![/dim]")
            break

        _run_query(pipeline, question)
        console.print()


# ── CLI Wiring ────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hybrid-rag",
        description="Hybrid RAG pipeline: vector search + BM25 + RRF",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest documents into the knowledge base")
    p_ingest.add_argument(
        "--docs-dir", default="data/sample_docs",
        help="Directory containing documents to ingest (default: data/sample_docs)"
    )
    p_ingest.add_argument(
        "--reset", action="store_true",
        help="Wipe existing vector store before ingesting"
    )

    # query
    p_query = sub.add_parser("query", help="Ask a single question")
    p_query.add_argument("--question", "-q", required=True, help="Question to answer")

    # chat
    sub.add_parser("chat", help="Interactive chat mode")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "ingest": cmd_ingest,
        "query": cmd_query,
        "chat": cmd_chat,
    }
    dispatch[args.command](args)
