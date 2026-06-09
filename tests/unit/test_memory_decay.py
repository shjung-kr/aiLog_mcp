"""Tests for MemoryDecay."""

import math
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from app.pipeline.memory.memory_decay import MemoryDecay, MIN_IMPORTANCE


def _ep(importance: float, end_at: datetime, promoted: bool = False) -> MagicMock:
    ep = MagicMock()
    ep.importance_score = importance
    ep.end_at = end_at
    ep.metadata_json = {"promoted_to_ltm": promoted} if promoted else {}
    return ep


def _mem(importance: float, created_at: datetime) -> MagicMock:
    mem = MagicMock()
    mem.importance_score = importance
    mem.created_at = created_at
    return mem


NOW = datetime.now(timezone.utc)


class TestMemoryDecay:
    def test_fresh_episode_barely_decays(self):
        decay = MemoryDecay(half_life_days=90.0)
        ep = _ep(1.0, NOW - timedelta(hours=1))
        result = decay.decay_episode(ep)
        # age < 0.1 days → decay factor ≈ 1, score stays near 1.0
        assert result is not None
        assert result > 0.99

    def test_old_episode_decays_significantly(self):
        decay = MemoryDecay(half_life_days=90.0)
        ep = _ep(1.0, NOW - timedelta(days=90))
        result = decay.decay_episode(ep)
        # After one half-life, score should be ≈ 0.5
        assert result is not None
        assert math.isclose(result, 0.5, abs_tol=0.01)

    def test_very_old_episode_floors_at_min(self):
        decay = MemoryDecay(half_life_days=90.0)
        ep = _ep(1.0, NOW - timedelta(days=3650))
        result = decay.decay_episode(ep)
        assert result == MIN_IMPORTANCE

    def test_promoted_episode_decays_slower(self):
        decay = MemoryDecay(half_life_days=90.0)
        age = NOW - timedelta(days=90)
        ep_normal = _ep(1.0, age, promoted=False)
        ep_promoted = _ep(1.0, age, promoted=True)
        score_normal = decay.decay_episode(ep_normal)
        score_promoted = decay.decay_episode(ep_promoted)
        assert score_promoted > score_normal

    def test_none_importance_returns_none(self):
        decay = MemoryDecay()
        ep = _ep(1.0, NOW)
        ep.importance_score = None
        assert decay.decay_episode(ep) is None

    def test_apply_to_episodes_mutates_in_place(self):
        decay = MemoryDecay(half_life_days=90.0)
        ep = _ep(1.0, NOW - timedelta(days=90))
        original_id = id(ep)
        result = decay.apply_to_episodes([ep])
        assert result[0] is ep  # same object
        assert ep.importance_score < 1.0

    def test_ltm_memory_decays_slower_than_episode(self):
        decay = MemoryDecay(half_life_days=90.0)
        age = NOW - timedelta(days=90)
        ep = _ep(1.0, age)
        mem = _mem(1.0, age)
        ep_score = decay.decay_episode(ep)
        mem_score = decay.decay_memory(mem)
        assert mem_score > ep_score
