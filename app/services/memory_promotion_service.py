from app.db.models.episode import Episode
from app.db.models.long_term_memory import LongTermMemory
from app.db.repositories.long_term_memory_repository import LongTermMemoryRepository
from app.services.episode_service import EpisodeService
from app.utils.datetime import utc_now
from app.utils.ids import new_id

IMPORTANCE_THRESHOLD = 0.72
MIN_CONTRIBUTING_SESSIONS = 2
LTM_PROMOTED_FLAG = "promoted_to_ltm"
SEMANTIC_TEXT_KEY = "semantic_text"

_EPISODE_TYPE_TO_MEMORY_TYPE: dict[str, str] = {
    "decision": "decision",
    "insight": "insight",
}


class MemoryPromotionService:
    def __init__(
        self,
        ltm_repository: LongTermMemoryRepository,
        episode_service: EpisodeService,
    ) -> None:
        self.ltm_repository = ltm_repository
        self.episode_service = episode_service

    def promote_from_episodes(self, episodes: list[Episode]) -> list[LongTermMemory]:
        results: list[LongTermMemory] = []
        for episode in episodes:
            if not self._should_promote(episode):
                continue
            existing = self.ltm_repository.get_by_episode_id(episode.episode_id)
            if existing:
                memory = self._update_memory(existing, episode)
            else:
                memory = self._create_memory(episode)
            results.append(memory)
        return results

    def run_full_promotion(self, limit: int = 500) -> list[LongTermMemory]:
        """Scan all episodes and promote eligible ones. Useful for backfill."""
        episodes = self.episode_service.list_all_episodes(limit=limit)
        return self.promote_from_episodes(episodes)

    def _should_promote(self, episode: Episode) -> bool:
        if episode.importance_score is not None and episode.importance_score >= IMPORTANCE_THRESHOLD:
            return True
        metadata = episode.metadata_json or {}
        contributing = metadata.get("contributing_session_ids") or []
        if len(contributing) >= MIN_CONTRIBUTING_SESSIONS:
            return True
        return False

    def _memory_text(self, episode: Episode) -> str:
        metadata = episode.metadata_json or {}
        semantic_text = metadata.get(SEMANTIC_TEXT_KEY)
        if isinstance(semantic_text, str) and semantic_text.strip():
            return semantic_text.strip()
        return episode.summary.strip()

    def _memory_type(self, episode: Episode) -> str:
        return _EPISODE_TYPE_TO_MEMORY_TYPE.get(episode.episode_type, "knowledge")

    def _create_memory(self, episode: Episode) -> LongTermMemory:
        memory = LongTermMemory(
            memory_id=new_id(),
            episode_id=episode.episode_id,
            title=episode.title,
            memory_text=self._memory_text(episode),
            memory_type=self._memory_type(episode),
            importance_score=episode.importance_score,
            created_at=utc_now(),
            metadata_json={
                "source_episode_type": episode.episode_type,
                "keywords": episode.keywords or [],
                "contributing_session_ids": (episode.metadata_json or {}).get("contributing_session_ids", []),
            },
        )
        created = self.ltm_repository.create(memory)
        self._flag_episode(episode)
        return created

    def _update_memory(self, memory: LongTermMemory, episode: Episode) -> LongTermMemory:
        memory.title = episode.title
        memory.memory_text = self._memory_text(episode)
        memory.memory_type = self._memory_type(episode)
        memory.importance_score = episode.importance_score
        memory.metadata_json = {
            **(memory.metadata_json or {}),
            "source_episode_type": episode.episode_type,
            "keywords": episode.keywords or [],
            "contributing_session_ids": (episode.metadata_json or {}).get("contributing_session_ids", []),
        }
        updated = self.ltm_repository.update(memory)
        self._flag_episode(episode)
        return updated

    def _flag_episode(self, episode: Episode) -> None:
        """Mark the episode so retrieval can apply a priority boost without an extra DB query."""
        episode.metadata_json = {
            **(episode.metadata_json or {}),
            LTM_PROMOTED_FLAG: True,
        }
        self.episode_service.update_episode(episode)
