"""Time-based importance decay for episodes and long-term memories."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.db.models.episode import Episode
    from app.db.models.long_term_memory import LongTermMemory

# Half-life: importance halves after this many days without a contributing session update.
DEFAULT_HALF_LIFE_DAYS = 90.0
MIN_IMPORTANCE = 0.05  # floor — never decay to zero so items stay searchable


class MemoryDecay:
    """Applies exponential decay to importance scores.

    Decay formula:  new_score = base * 2^(-age_days / half_life) + MIN_IMPORTANCE
    Items that have been promoted to LTM decay more slowly (2× half-life).
    """

    def __init__(self, half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> None:
        self.half_life_days = half_life_days

    def decay_episode(self, episode: Episode) -> float | None:
        """Return the decayed importance score, or None if episode has no score."""
        if episode.importance_score is None:
            return None

        metadata = episode.metadata_json or {}
        promoted = bool(metadata.get("promoted_to_ltm"))
        half_life = self.half_life_days * 2 if promoted else self.half_life_days

        age_days = self._age_days(episode.end_at)
        return self._apply_decay(episode.importance_score, age_days, half_life)

    def decay_memory(self, memory: LongTermMemory) -> float | None:
        """Return the decayed importance score for a long-term memory."""
        if memory.importance_score is None:
            return None
        age_days = self._age_days(memory.created_at)
        # LTM items decay at 3× the normal half-life — they are meant to persist.
        return self._apply_decay(memory.importance_score, age_days, self.half_life_days * 3)

    def apply_to_episodes(self, episodes: list[Episode]) -> list[Episode]:
        """Mutate importance_score in-place and return the list."""
        for ep in episodes:
            decayed = self.decay_episode(ep)
            if decayed is not None:
                ep.importance_score = decayed
        return episodes

    def apply_to_memories(self, memories: list[LongTermMemory]) -> list[LongTermMemory]:
        """Mutate importance_score in-place and return the list."""
        for mem in memories:
            decayed = self.decay_memory(mem)
            if decayed is not None:
                mem.importance_score = decayed
        return memories

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _age_days(ts: datetime | None) -> float:
        if ts is None:
            return 0.0
        now = datetime.now(timezone.utc)
        aware = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        delta = now - aware
        return max(0.0, delta.total_seconds() / 86400.0)

    @staticmethod
    def _apply_decay(base: float, age_days: float, half_life: float) -> float:
        if half_life <= 0:
            return base
        decayed = base * math.pow(2.0, -age_days / half_life)
        return max(MIN_IMPORTANCE, decayed)
