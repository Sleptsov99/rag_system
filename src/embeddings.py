"""
Embeddings
==========
Wraps sentence-transformers to produce dense vector representations.

Key concepts:
  • An embedding is a fixed-length float vector (e.g. 384-dim) that encodes
    the semantic meaning of a text.
  • Similar texts → nearby vectors in the embedding space.
  • We use the same model for both documents AND queries so that their
    vectors live in the same space and cosine similarity is meaningful.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import config


class EmbeddingModel:
    """
    Thin wrapper around SentenceTransformer.

    Model choice: 'paraphrase-multilingual-MiniLM-L12-v2'
      • 384-dimensional embeddings
      • Supports 50+ languages including Russian
      • ~120 MB on disk; fast even on CPU
    """

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
        """
        Embed a list of texts in mini-batches.
        normalize_embeddings=True → cosine similarity = dot product.
        """
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
