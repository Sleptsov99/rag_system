"""Wraps sentence-transformers to produce dense vector representations."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import config


class EmbeddingModel:
    """Thin wrapper around SentenceTransformer."""

    def __init__(self, model_name: str | None = None):
        model_name = model_name or config.EMBEDDING_MODEL
        print(f"Loading embedding model '{model_name}'…")
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()
        print(f"  Embedding dimension: {self.dim}")

    def embed(self, text: str) -> list[float]:
        """Embed a single string. Returns a Python list (ChromaDB-compatible)."""
        vector: np.ndarray = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> list[list[float]]:
        """Embed a list of texts in mini-batches."""
        vectors: np.ndarray = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return vectors.tolist()

    def similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """Cosine similarity between two already-normalised vectors."""
        a = np.array(vec_a)
        b = np.array(vec_b)
        return float(np.dot(a, b))
