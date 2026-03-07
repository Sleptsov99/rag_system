"""Persistent ChromaDB vector store for document chunks and embeddings."""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings

from config import config
from src.document_processor import Document
from src.embeddings import EmbeddingModel


class VectorStore:
    """Persistent ChromaDB collection that holds document chunks + embeddings."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        persist_dir: str | Path | None = None,
        collection_name: str | None = None,
    ):
        self._embedding = embedding_model
        persist_dir = Path(persist_dir or config.CHROMA_DIR)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name or config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"ChromaDB collection '{self._collection.name}' "
              f"({self._collection.count()} docs already stored)")

    def add_documents(self, documents: list[Document], batch_size: int = 100) -> None:
        """
        Embed and store a list of Document chunks.
        Automatically skips duplicates (same source + chunk_id).
        """
        if not documents:
            return

        texts = [doc.text for doc in documents]
        print(f"Embedding {len(texts)} chunks…")
        embeddings = self._embedding.embed_batch(texts)

        ids = [self._make_id(doc) for doc in documents]
        metadatas = [
            {"source": doc.source, "chunk_id": doc.chunk_id, **doc.metadata}
            for doc in documents
        ]

        for start in range(0, len(documents), batch_size):
            end = start + batch_size
            self._collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=texts[start:end],
                metadatas=metadatas[start:end],
            )
        print(f"Stored {len(documents)} chunks. "
              f"Collection total: {self._collection.count()}")

    def clear(self) -> None:
        """Delete all documents from the collection."""
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
        )
        print("Collection cleared.")

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Semantic search: embed the query, find the most similar chunks.

        Returns a list of dicts:
            {
              "text":     str,
              "source":   str,
              "chunk_id": int,
              "score":    float,   # cosine similarity ∈ [0, 1]
            }
        """
        top_k = top_k or config.TOP_K
        if self._collection.count() == 0:
            return []

        query_embedding = self._embedding.embed(query)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = round(1.0 - dist, 4)
            hits.append({
                "text": text,
                "source": meta.get("source", ""),
                "chunk_id": meta.get("chunk_id", 0),
                "score": score,
                **{k: v for k, v in meta.items() if k not in ("source", "chunk_id")},
            })

        return hits

    def count(self) -> int:
        return self._collection.count()

    def list_sources(self) -> list[dict]:
        """Return unique sources with their chunk counts."""
        if self._collection.count() == 0:
            return []
        result = self._collection.get(include=["metadatas"])
        sources: dict[str, int] = {}
        for meta in result["metadatas"]:
            src = meta.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
        return [{"source": src, "chunks": n} for src, n in sorted(sources.items())]

    @staticmethod
    def _make_id(doc: Document) -> str:
        safe = doc.source.replace("/", "_").replace("\\", "_").replace(".", "_")
        return f"{safe}::{doc.chunk_id}"
