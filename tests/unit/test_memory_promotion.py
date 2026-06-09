"""Tests for MemoryPromotionService."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, call

from app.services.memory_promotion_service import (
    IMPORTANCE_THRESHOLD,
    MIN_CONTRIBUTING_SESSIONS,
    MemoryPromotionService,
)


def _episode(
    episode_id: str = "ep1",
    importance: float | None = None,
    episode_type: str = "topic",
    contributing_sessions: list[str] | None = None,
    promoted: bool = False,
) -> MagicMock:
    ep = MagicMock()
    ep.episode_id = episode_id
    ep.importance_score = importance
    ep.episode_type = episode_type
    ep.title = "Test Episode"
    ep.summary = "A test episode"
    ep.keywords = ["test"]
    metadata: dict = {}
    if contributing_sessions:
        metadata["contributing_session_ids"] = contributing_sessions
    if promoted:
        metadata["promoted_to_ltm"] = True
    ep.metadata_json = metadata
    return ep


def _make_service():
    ltm_repo = MagicMock()
    ltm_repo.get_by_episode_id.return_value = None
    ltm_repo.create.side_effect = lambda m: m
    episode_service = MagicMock()
    episode_service.update_episode.return_value = None
    return MemoryPromotionService(ltm_repo, episode_service), ltm_repo, episode_service


class TestShouldPromote:
    def test_high_importance_promotes(self):
        svc, _, _ = _make_service()
        ep = _episode(importance=IMPORTANCE_THRESHOLD)
        assert svc._should_promote(ep)

    def test_below_threshold_does_not_promote(self):
        svc, _, _ = _make_service()
        ep = _episode(importance=IMPORTANCE_THRESHOLD - 0.01)
        assert not svc._should_promote(ep)

    def test_enough_contributing_sessions_promotes(self):
        svc, _, _ = _make_service()
        sessions = [f"s{i}" for i in range(MIN_CONTRIBUTING_SESSIONS)]
        ep = _episode(importance=0.0, contributing_sessions=sessions)
        assert svc._should_promote(ep)

    def test_too_few_sessions_does_not_promote(self):
        svc, _, _ = _make_service()
        ep = _episode(importance=0.0, contributing_sessions=["s1"])
        assert not svc._should_promote(ep)

    def test_none_importance_and_no_sessions_does_not_promote(self):
        svc, _, _ = _make_service()
        ep = _episode(importance=None)
        assert not svc._should_promote(ep)


class TestPromoteFromEpisodes:
    def test_creates_new_memory_for_eligible_episode(self):
        svc, ltm_repo, episode_service = _make_service()
        ep = _episode(importance=IMPORTANCE_THRESHOLD)
        results = svc.promote_from_episodes([ep])
        assert len(results) == 1
        ltm_repo.create.assert_called_once()

    def test_updates_existing_memory(self):
        svc, ltm_repo, episode_service = _make_service()
        existing_mem = MagicMock()
        ltm_repo.get_by_episode_id.return_value = existing_mem
        ltm_repo.update.return_value = existing_mem
        ep = _episode(importance=IMPORTANCE_THRESHOLD)
        results = svc.promote_from_episodes([ep])
        assert len(results) == 1
        ltm_repo.update.assert_called_once()
        ltm_repo.create.assert_not_called()

    def test_skips_ineligible_episode(self):
        svc, ltm_repo, _ = _make_service()
        ep = _episode(importance=0.1)
        results = svc.promote_from_episodes([ep])
        assert results == []
        ltm_repo.create.assert_not_called()

    def test_flags_episode_after_promotion(self):
        svc, _, episode_service = _make_service()
        ep = _episode(importance=IMPORTANCE_THRESHOLD)
        svc.promote_from_episodes([ep])
        episode_service.update_episode.assert_called_once_with(ep)
        assert ep.metadata_json.get("promoted_to_ltm") is True
