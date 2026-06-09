"""Tests for numpy-based VectorSearch."""

from unittest.mock import MagicMock

from app.search.vector import VectorSearch


def _make_episode(episode_id: str, embedding: list[float], semantic_text: str = "", promoted: bool = False) -> MagicMock:
    ep = MagicMock()
    ep.episode_id = episode_id
    ep.metadata_json = {
        "semantic_embedding": embedding,
        "semantic_text": semantic_text,
        "promoted_to_ltm": promoted,
    }
    return ep


class TestVectorSearch:
    def test_returns_empty_when_no_episodes(self):
        vs = VectorSearch([])
        results = vs.search(query_embedding=[1.0, 0.0], query_tokens=set())
        assert results == []

    def test_returns_empty_when_no_embeddings(self):
        ep = MagicMock()
        ep.metadata_json = {}
        vs = VectorSearch([ep])
        results = vs.search(query_embedding=[1.0, 0.0], query_tokens=set())
        assert results == []

    def test_identical_embedding_scores_near_one(self):
        import math
        ep = _make_episode("ep1", [1.0, 0.0, 0.0])
        vs = VectorSearch([ep])
        results = vs.search(query_embedding=[1.0, 0.0, 0.0], query_tokens=set())
        assert len(results) == 1
        score, episode = results[0]
        # keyword_boost_weight=0.15, keyword overlap=0 → score = cosine * 0.85 ≈ 0.85
        assert math.isclose(score, 0.85, abs_tol=1e-5)

    def test_orthogonal_embedding_scores_near_zero(self):
        ep = _make_episode("ep1", [0.0, 1.0, 0.0])
        vs = VectorSearch([ep])
        results = vs.search(query_embedding=[1.0, 0.0, 0.0], query_tokens=set())
        assert len(results) == 1
        assert abs(results[0][0]) < 0.01

    def test_ltm_promoted_gets_boost(self):
        import math
        ep_normal = _make_episode("ep1", [1.0, 0.0])
        ep_promoted = _make_episode("ep2", [1.0, 0.0], promoted=True)
        vs = VectorSearch([ep_normal, ep_promoted])
        results = vs.search(query_embedding=[1.0, 0.0], query_tokens=set())
        scores = {r[1].episode_id: r[0] for r in results}
        assert scores["ep2"] > scores["ep1"]
        assert math.isclose(scores["ep2"] - scores["ep1"], 0.05, abs_tol=1e-5)

    def test_failed_recall_episode_skipped(self):
        import re
        bad_text = "자동으로 불러오진 못했습니다"
        ep = _make_episode("ep1", [1.0, 0.0], semantic_text=bad_text)
        pattern = re.compile(r"자동으로\s*불러오[진]?\s*못")
        vs = VectorSearch([ep])
        results = vs.search(
            query_embedding=[1.0, 0.0],
            query_tokens=set(),
            failed_recall_re=pattern,
        )
        assert results == []

    def test_keyword_overlap_boosts_score(self):
        ep = _make_episode("ep1", [1.0, 0.0], semantic_text="파이썬 머신러닝 모델 학습")
        vs = VectorSearch([ep])
        results_no_kw = vs.search(query_embedding=[1.0, 0.0], query_tokens=set())
        results_with_kw = vs.search(query_embedding=[1.0, 0.0], query_tokens={"파이썬", "모델"})
        assert results_with_kw[0][0] > results_no_kw[0][0]
