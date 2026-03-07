"""RAG pipeline: ingest documents and answer questions."""

from __future__ import annotations

from pathlib import Path

from config import Config, config as default_config
from src.document_processor import DocumentLoader, TextChunker, TextPreprocessor
from src.embeddings import EmbeddingModel
from src.generator import BaseGenerator, get_generator
from src.retriever import Retriever
from src.vector_store import VectorStore


class RAGPipeline:

    def __init__(self, cfg: Config | None = None, llm_provider: str | None = None):
        cfg = cfg or default_config

        self._embedding = EmbeddingModel(cfg.EMBEDDING_MODEL)

        self._loader = DocumentLoader()
        self._preprocessor = TextPreprocessor(language="russian")
        self._chunker = TextChunker(
            chunk_size=cfg.CHUNK_SIZE,
            chunk_overlap=cfg.CHUNK_OVERLAP,
        )

        self._store = VectorStore(
            embedding_model=self._embedding,
            persist_dir=cfg.CHROMA_DIR,
            collection_name=cfg.COLLECTION_NAME,
        )
        self._retriever = Retriever(self._store)

        self._generator: BaseGenerator = get_generator(llm_provider or cfg.LLM_PROVIDER)

        self._top_k = cfg.TOP_K

    def ingest(self, directory: str | Path) -> int:
        """
        Load all documents from *directory*, chunk them, embed, and store.
        Returns the number of chunks indexed.
        """
        directory = Path(directory)
        print(f"\n=== Ingestion: {directory} ===")

        documents = self._loader.load_directory(directory)
        if not documents:
            print("No documents found. Add .txt / .pdf / .docx files to the folder.")
            return 0

        chunks = self._chunker.split_documents(documents)
        print(f"  Split into {len(chunks)} chunks")

        self._store.add_documents(chunks)
        return len(chunks)

    def ingest_text(self, text: str, source: str = "manual") -> int:
        """Ingest a raw text string directly (useful for demos)."""
        from src.document_processor import Document
        doc = Document(text=text, source=source)
        chunks = self._chunker.split_documents([doc])
        self._store.add_documents(chunks)
        return len(chunks)

    def ingest_file(self, path: str | Path) -> int:
        """Ingest a single file (PDF / DOCX / TXT)."""
        from src.document_processor import Document
        path = Path(path)
        text = self._loader.load_file(path)
        doc = Document(text=text, source=path.name)
        chunks = self._chunker.split_documents([doc])
        self._store.add_documents(chunks)
        return len(chunks)

    def query(
        self,
        question: str,
        top_k: int | None = None,
        use_mmr: bool = False,
        return_sources: bool = False,
    ) -> str | dict:
        """
        Answer a question using the indexed documents.

        Parameters
        ----------
        question     : the user's question
        top_k        : how many chunks to retrieve (default from config)
        use_mmr      : use Maximal Marginal Relevance for diverse results
        return_sources: if True, return dict with 'answer' and 'sources'
        """
        top_k = top_k or self._top_k

        if use_mmr:
            hits = self._retriever.retrieve_mmr(question, top_k=top_k)
        else:
            hits = self._retriever.retrieve(question, top_k=top_k)

        if not hits:
            msg = "Документы не найдены. Сначала запустите ingest()."
            if return_sources:
                return {"answer": msg, "sources": []}
            return msg

        context = self._retriever.format_context(hits)
        answer = self._generator.generate(question, context)

        if return_sources:
            sources = [
                {"source": h["source"], "chunk_id": h["chunk_id"], "score": h["score"]}
                for h in hits
            ]
            return {"answer": answer, "sources": sources}

        return answer

    def clear_index(self) -> None:
        """Remove all indexed documents."""
        self._store.clear()

    @property
    def document_count(self) -> int:
        return self._store.count()

    def list_sources(self) -> list[dict]:
        """Return unique indexed files with their chunk counts."""
        return self._store.list_sources()

    def delete_source(self, source: str) -> int:
        """Delete all chunks for the given source file. Returns number deleted."""
        return self._store.delete_source(source)

    def show_retrieved_chunks(self, question: str, top_k: int | None = None) -> None:
        """Debug helper: print retrieved chunks without calling the LLM."""
        hits = self._retriever.retrieve(question, top_k=top_k or self._top_k)
        print(f"\n=== Retrieved chunks for: {question!r} ===\n")
        for i, hit in enumerate(hits, 1):
            print(f"[{i}] score={hit['score']:.3f}  source={hit['source']}")
            print(f"     {hit['text'][:200]!r}…\n")
