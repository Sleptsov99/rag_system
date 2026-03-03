"""
Vector Store  (ChromaDB)
========================
ChromaDB is an open-source, embeddable vector database.
It stores:
  • document text   (the raw chunk)
  • embedding       (the dense float vector)
  • metadata        (source file, chunk_id, …)

All three are retrievable together on a single query.

ChromaDB distance metrics:
  • "cosine"  — cosine similarity (good for normalised embeddings)
  • "l2"      — Euclidean distance
  • "ip"      — inner product (= cosine when vectors are unit-length)
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings

from config import config
from src.document_processor import Document
from src.embeddings import EmbeddingModel


class VectorStore:
    """
    Persistent ChromaDB collection that holds document chunks + embeddings.

    Usage:
        store = VectorStore(embedding_model)
        store.add_documents(chunks)
        results = store.search("какой налог на прибыль?", top_k=5)
    """

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
            metadata={"hnsw:space": "cosine"},   # cosine similarity index
        )
        print(f"ChromaDB collection '{self._collection.name}' "
              f"({self._collection.count()} docs already stored)")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

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

        # Build IDs: source_path::chunk_id  (made safe for ChromaDB)
        ids = [self._make_id(doc) for doc in documents]
        metadatas = [
            {"source": doc.source, "chunk_id": doc.chunk_id, **doc.metadata}
            for doc in documents
        ]

        # Insert in batches to avoid memory spikes
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

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

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
            # ChromaDB "cosine" space returns distance = 1 - similarity
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_id(doc: Document) -> str:
        safe = doc.source.replace("/", "_").replace("\\", "_").replace(".", "_")
        return f"{safe}::{doc.chunk_id}"
