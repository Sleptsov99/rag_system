"""
CLI entry-point for the RAG system.

Commands:
  ingest  — index documents from data/documents/
  query   — ask a question (interactive or single)
  clear   — wipe the vector store

Usage:
  python main.py ingest
  python main.py query "Что такое RAG?"
  python main.py query          # interactive mode
  python main.py clear
"""

import argparse
import sys

from config import config
from src.pipeline import RAGPipeline


def cmd_ingest(args, pipeline: RAGPipeline) -> None:
    directory = args.dir or config.DATA_DIR
    n = pipeline.ingest(directory)
    print(f"\nDone. {n} chunks indexed. Total in store: {pipeline.document_count}")


def cmd_query(args, pipeline: RAGPipeline) -> None:
    if pipeline.document_count == 0:
        print("Index is empty. Run:  python main.py ingest")
        sys.exit(1)

    if args.question:
        _ask(pipeline, args.question, args.mmr, args.sources, args.top_k)
    else:
        print(f"\nRAG system ready ({pipeline.document_count} chunks indexed).")
        print("Type your question and press Enter. Type 'exit' to quit.\n")
        while True:
            try:
                q = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye.")
                break
            if not q:
                continue
            if q.lower() in ("exit", "quit", "q"):
                break
            _ask(pipeline, q, args.mmr, args.sources, args.top_k)
            print()


def _ask(pipeline, question, use_mmr, show_sources, top_k):
    result = pipeline.query(
        question,
        top_k=top_k,
        use_mmr=use_mmr,
        return_sources=show_sources,
    )
    if show_sources and isinstance(result, dict):
        print(f"\nAnswer:\n{result['answer']}")
        print("\nSources:")
        for s in result["sources"]:
            print(f"  [{s['score']:.3f}] {s['source']}  (chunk {s['chunk_id']})")
    else:
        print(f"\nAnswer:\n{result}")


def cmd_clear(_, pipeline: RAGPipeline) -> None:
    confirm = input("Clear ALL indexed documents? [y/N] ").strip().lower()
    if confirm == "y":
        pipeline.clear_index()
        print("Index cleared.")
    else:
        print("Cancelled.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG System — Retrieval-Augmented Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--provider", choices=["ollama", "openai", "dummy"],
        help="LLM provider (overrides config.py)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- ingest ---
    p_ingest = sub.add_parser("ingest", help="Index documents")
    p_ingest.add_argument("--dir", help="Directory with documents (default: data/documents)")

    # --- query ---
    p_query = sub.add_parser("query", help="Ask a question")
    p_query.add_argument("question", nargs="?", help="Question (omit for interactive mode)")
    p_query.add_argument("--mmr", action="store_true", help="Use MMR re-ranking")
    p_query.add_argument("--sources", action="store_true", help="Show source chunks")
    p_query.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve")

    # --- clear ---
    sub.add_parser("clear", help="Wipe the vector store")

    args = parser.parse_args()

    pipeline = RAGPipeline(llm_provider=args.provider)

    dispatch = {"ingest": cmd_ingest, "query": cmd_query, "clear": cmd_clear}
    dispatch[args.command](args, pipeline)


if __name__ == "__main__":
    main()
