"""Hybrid search: combines vector similarity with full-text keyword overlap."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.search.vector import VectorSearch, _keyword_overlap, _tokenize

if TYPE_CHECKING:
    from app.db.models.episode import Episode


class HybridSearch:
    """Two-stage hybrid retrieval.

    Stage 1: VectorSearch produces cosine + keyword scores.
    Stage 2: Optionally re-score with a richer text field (e.g. full summary)
             using a configurable vector/keyword split.
    """

    def __init__(
        self,
        episodes: list[Episode],
        vector_weight: float = 0.85,
        keyword_weight: float = 0.15,
    ) -> None:
        if abs(vector_weight + keyword_weight - 1.0) > 1e-6:
            raise ValueError("vector_weight + keyword_weight must equal 1.0")
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self._vector_search = VectorSearch(episodes)

    def search(
        self,
        query_embedding: list[float],
        query_text: str,
        threshold: float = 0.0,
    ) -> list[tuple[float, Episode]]:
        """Return (score, episode) pairs above threshold, sorted descending."""
        query_tokens = _tokenize(query_text)
        raw_results = self._vector_search.search(
            query_embedding=query_embedding,
            query_tokens=query_tokens,
            keyword_boost_weight=self.keyword_weight,
        )
        filtered = [(score, ep) for score, ep in raw_results if score >= threshold]
        return sorted(filtered, key=lambda item: item[0], reverse=True)
