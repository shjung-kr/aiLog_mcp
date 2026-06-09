"""Batch vector search over in-memory episode embeddings using numpy."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import re as re_module

    from app.db.models.episode import Episode

EMBEDDING_KEY = "semantic_embedding"
TITLE_EMBEDDING_KEY = "title_embedding"
SEMANTIC_TEXT_KEY = "semantic_text"

_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]{2,}")


class VectorSearch:
    """Numpy-based batch cosine search over episodes.

    Replaces the Python-loop approach in RetrievalService, reducing O(N) Python
    iterations to a single matrix multiply for the embedding dimension.
    """

    def __init__(self, episodes: list[Episode]) -> None:
        self._episodes = episodes
        self._embeddings: list[list[float]] = []
        self._title_embeddings: list[list[float] | None] = []
        self._valid_indices: list[int] = []

        for idx, ep in enumerate(episodes):
            metadata = ep.metadata_json or {}
            emb = metadata.get(EMBEDDING_KEY)
            if isinstance(emb, list) and emb:
                self._valid_indices.append(idx)
                self._embeddings.append([float(v) for v in emb])
                title_emb = metadata.get(TITLE_EMBEDDING_KEY)
                self._title_embeddings.append(
                    [float(v) for v in title_emb] if isinstance(title_emb, list) else None
                )

    def search(
        self,
        query_embedding: list[float],
        query_tokens: set[str],
        keyword_boost_weight: float = 0.15,
        failed_recall_re: re_module.Pattern | None = None,
    ) -> list[tuple[float, Episode]]:
        """Return (score, episode) pairs for all episodes that have an embedding.

        Scores combine cosine similarity (85%) and keyword overlap (15%).
        LTM-promoted episodes receive a +0.05 boost.
        """
        if not self._valid_indices or not query_embedding:
            return []

        # Build embedding matrix: shape (N, dim)
        matrix = np.array(self._embeddings, dtype=np.float32)
        qv = np.array(query_embedding, dtype=np.float32)

        norm_q = np.linalg.norm(qv)
        if norm_q == 0:
            return []
        qv_norm = qv / norm_q

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        cosines = (matrix / norms) @ qv_norm  # shape (N,)

        results: list[tuple[float, Episode]] = []
        for local_idx, global_idx in enumerate(self._valid_indices):
            episode = self._episodes[global_idx]
            metadata = episode.metadata_json or {}
            semantic_text = metadata.get(SEMANTIC_TEXT_KEY, "") or ""

            if failed_recall_re and failed_recall_re.search(semantic_text[:300]):
                continue

            cosine = float(cosines[local_idx])

            # Title embedding boost
            title_emb = self._title_embeddings[local_idx]
            if title_emb is not None:
                tv = np.array(title_emb, dtype=np.float32)
                norm_t = np.linalg.norm(tv)
                if norm_t > 0:
                    cosine_title = float(np.dot(qv_norm, tv / norm_t))
                    cosine = max(cosine, cosine_title)

            keyword_score = _keyword_overlap(query_tokens, semantic_text)
            score = cosine * (1 - keyword_boost_weight) + keyword_score * keyword_boost_weight
            if metadata.get("promoted_to_ltm"):
                score += 0.05

            results.append((score, episode))

        return results


def _tokenize(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2}


def _keyword_overlap(query_tokens: set[str], text: str) -> float:
    if not query_tokens or not text:
        return 0.0
    text_tokens = _tokenize(text)
    if not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)
