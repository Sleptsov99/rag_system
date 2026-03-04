"""Semantic retriever with optional MMR re-ranking."""

from __future__ import annotations

import numpy as np

from config import config
from src.vector_store import VectorStore


class Retriever:

    def __init__(self, vector_store: VectorStore):
        self._store = vector_store

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        Return the top-k most semantically similar chunks.
        Results are sorted by score (highest first).
        """
        return self._store.search(query, top_k=top_k or config.TOP_K)

    def retrieve_mmr(
        self,
        query: str,
        top_k: int | None = None,
        fetch_k: int | None = None,
        lambda_mult: float = 0.5,
    ) -> list[dict]:
        """
        Maximal Marginal Relevance retrieval.

        Balances relevance (to the query) and diversity (between results).
        lambda_mult ∈ [0, 1]:
          • 1.0 → pure relevance (same as standard retrieval)
          • 0.0 → pure diversity
          • 0.5 → balanced (default)

        Algorithm:
          1. Fetch fetch_k candidates from the vector store.
          2. Greedily pick top_k from them by maximising:
               MMR = λ · sim(chunk, query) − (1−λ) · max_sim(chunk, selected)
        """
        top_k = top_k or config.TOP_K
        fetch_k = fetch_k or top_k * 4

        candidates = self._store.search(query, top_k=fetch_k)
        if len(candidates) <= top_k:
            return candidates

        embedding_model = self._store._embedding
        query_vec = np.array(embedding_model.embed(query))
        cand_vecs = np.array([
            embedding_model.embed(c["text"]) for c in candidates
        ])

        selected_indices: list[int] = []
        remaining = list(range(len(candidates)))

        while len(selected_indices) < top_k and remaining:
            mmr_scores: list[float] = []
            for i in remaining:
                relevance = float(np.dot(cand_vecs[i], query_vec))
                if selected_indices:
                    redundancy = max(
                        float(np.dot(cand_vecs[i], cand_vecs[j]))
                        for j in selected_indices
                    )
                else:
                    redundancy = 0.0
                mmr_scores.append(lambda_mult * relevance - (1 - lambda_mult) * redundancy)

            best_local = int(np.argmax(mmr_scores))
            best_global = remaining[best_local]
            selected_indices.append(best_global)
            remaining.remove(best_global)

        return [candidates[i] for i in selected_indices]

    def format_context(self, hits: list[dict]) -> str:
        """
        Assemble retrieved chunks into a numbered context block
        that will be injected into the LLM prompt.
        """
        if not hits:
            return "Relevant documents not found."

        parts: list[str] = []
        for rank, hit in enumerate(hits, start=1):
            source = hit.get("source", "unknown")
            score = hit.get("score", 0.0)
            text = hit.get("text", "").strip()
            parts.append(
                f"[{rank}] Source: {source} (relevance: {score:.2f})\n{text}"
            )
        return "\n\n---\n\n".join(parts)
