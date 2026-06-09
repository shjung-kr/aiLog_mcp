"""Tests for RetrievalService — mocked LLM and DB."""

from unittest.mock import MagicMock, patch

from app.services.retrieval_service import (
    EMBEDDING_METADATA_KEY,
    RETRIEVAL_SCORE_THRESHOLD,
    RetrievalService,
    SEMANTIC_TEXT_METADATA_KEY,
    TITLE_EMBEDDING_METADATA_KEY,
)


def _embedding(value: float = 1.0) -> list[float]:
    """Unit vector in 3D along value direction (simplified)."""
    return [value, 0.0, 0.0]


def _episode(episode_id: str, embedding: list[float] | None = None, semantic_text: str = "") -> MagicMock:
    ep = MagicMock()
    ep.episode_id = episode_id
    ep.title = f"Episode {episode_id}"
    ep.summary = "summary"
    ep.keywords = ["test"]
    ep.start_at = MagicMock()
    ep.start_at.isoformat.return_value = "2024-01-01T00:00:00+00:00"
    ep.metadata_json = {}
    if embedding is not None:
        ep.metadata_json[EMBEDDING_METADATA_KEY] = embedding
    if semantic_text:
        ep.metadata_json[SEMANTIC_TEXT_METADATA_KEY] = semantic_text
    return ep


def _make_service(episodes: list) -> tuple[RetrievalService, MagicMock, MagicMock]:
    ep_repo = MagicMock()
    ep_repo.list_all.return_value = episodes
    rawlog_svc = MagicMock()
    llm = MagicMock()
    search_repo = MagicMock()
    svc = RetrievalService(
        episode_repository=ep_repo,
        rawlog_service=rawlog_svc,
        llm_client=llm,
        search_repository=search_repo,
    )
    return svc, llm, search_repo


class TestRetrieveForQuery:
    def test_empty_query_returns_none(self):
        svc, _, _ = _make_service([])
        context, items = svc.retrieve_for_query("  ")
        assert context is None
        assert items == []

    def test_no_episodes_with_embeddings_returns_none(self):
        ep = _episode("ep1")  # no embedding
        svc, llm, _ = _make_service([ep])
        llm.embed_texts.return_value = [[1.0, 0.0, 0.0]]
        context, items = svc.retrieve_for_query("테스트 쿼리")
        assert context is None

    def test_high_score_episode_passes_threshold(self):
        ep = _episode("ep1", embedding=[1.0, 0.0, 0.0], semantic_text="파이썬 머신러닝")
        svc, llm, _ = _make_service([ep])
        llm.embed_texts.return_value = [[1.0, 0.0, 0.0]]  # identical → cosine=1.0
        llm.curate_episodes.return_value = (["ep1"], "relevant")
        context, items = svc.retrieve_for_query("파이썬 머신러닝에 대해")
        assert context is not None
        assert len(items) == 1
        assert items[0]["episode_id"] == "ep1"

    def test_curator_filters_out_low_relevance(self):
        ep = _episode("ep1", embedding=[1.0, 0.0, 0.0], semantic_text="파이썬 머신러닝")
        svc, llm, _ = _make_service([ep])
        llm.embed_texts.return_value = [[1.0, 0.0, 0.0]]
        llm.curate_episodes.return_value = ([], "not relevant")
        context, items = svc.retrieve_for_query("파이썬 머신러닝에 대해")
        assert context is None
        assert items == []

    def test_search_log_is_written(self):
        ep = _episode("ep1", embedding=[1.0, 0.0, 0.0], semantic_text="파이썬")
        svc, llm, search_repo = _make_service([ep])
        llm.embed_texts.return_value = [[1.0, 0.0, 0.0]]
        llm.curate_episodes.return_value = (["ep1"], "relevant")
        svc.retrieve_for_query("파이썬")
        search_repo.create.assert_called_once()

    def test_parse_recall_query_strips_meta_terms(self):
        svc, _, _ = _make_service([])
        result = svc._parse_recall_query("파이썬 이야기했던 거 기억해?")
        assert result.recall_intent is True
        assert "파이썬" in result.content_query
        assert "기억" not in result.content_query
