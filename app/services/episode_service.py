from app.db.models.episode import Episode
from app.db.models.rawlog import RawLog
from app.db.repositories.episode_repository import EpisodeRepository
from app.services.rawlog_service import RawLogService
from app.utils.ids import new_id


class EpisodeService:
    def __init__(self, episode_repository: EpisodeRepository, rawlog_service: RawLogService) -> None:
        self.episode_repository = episode_repository
        self.rawlog_service = rawlog_service

    def create_episode(
        self,
        title: str,
        summary: str,
        episode_type: str,
        rawlog_ids: list[str],
        emotion_signal: str | None = None,
        importance_score: float | None = None,
        source_session_id: str | None = None,
        keywords: list[str] | None = None,
        metadata: dict | None = None,
    ) -> Episode:
        if not title.strip():
            raise ValueError("title must not be empty")
        if not summary.strip():
            raise ValueError("summary must not be empty")
        if not episode_type.strip():
            raise ValueError("episode_type must not be empty")
        if not rawlog_ids:
            raise ValueError("rawlog_ids must not be empty")

        rawlogs = self._load_rawlogs(rawlog_ids)
        if len(rawlogs) != len(rawlog_ids):
            raise ValueError("all rawlog_ids must exist")

        ordered_rawlogs = self._order_rawlogs(rawlogs)
        start_rawlog = ordered_rawlogs[0]
        end_rawlog = ordered_rawlogs[-1]
        session_ids = sorted({rawlog.session_id for rawlog in ordered_rawlogs})
        episode = Episode(
            episode_id=new_id(),
            title=title.strip()[:255],
            summary=summary.strip(),
            episode_type=episode_type.strip(),
            start_rawlog_id=start_rawlog.rawlog_id,
            end_rawlog_id=end_rawlog.rawlog_id,
            start_at=start_rawlog.occurred_at,
            end_at=end_rawlog.occurred_at,
            emotion_signal=emotion_signal,
            importance_score=importance_score,
            source_session_id=source_session_id or start_rawlog.session_id,
            keywords=keywords,
            metadata_json={
                **(metadata or {}),
                "rawlog_count": len(ordered_rawlogs),
                "source_session_ids": session_ids,
            },
        )
        created = self.episode_repository.create(episode)
        self.episode_repository.replace_rawlog_links(
            created.episode_id,
            [rawlog.rawlog_id for rawlog in ordered_rawlogs],
        )
        return created

    def merge_episode(
        self,
        episode: Episode,
        title: str,
        summary: str,
        rawlog_ids: list[str],
        episode_type: str | None = None,
        emotion_signal: str | None = None,
        importance_score: float | None = None,
        keywords: list[str] | None = None,
        metadata: dict | None = None,
    ) -> Episode:
        existing_rawlog_ids = self.list_episode_rawlog_ids(episode.episode_id)
        combined_rawlog_ids = [*existing_rawlog_ids]
        for rawlog_id in rawlog_ids:
            if rawlog_id not in combined_rawlog_ids:
                combined_rawlog_ids.append(rawlog_id)

        rawlogs = self._load_rawlogs(combined_rawlog_ids)
        if len(rawlogs) != len(combined_rawlog_ids):
            raise ValueError("all rawlog_ids must exist")

        ordered_rawlogs = self._order_rawlogs(rawlogs)
        start_rawlog = ordered_rawlogs[0]
        end_rawlog = ordered_rawlogs[-1]
        session_ids = sorted({rawlog.session_id for rawlog in ordered_rawlogs})

        existing_metadata = episode.metadata_json or {}
        contributing: set[str] = set(existing_metadata.get("contributing_session_ids") or [])
        contributing.update(session_ids)

        episode.title = self._merge_title(episode.title, title)
        episode.summary = self._merge_summary(episode.summary, summary)
        if episode_type:
            episode.episode_type = episode_type
        episode.start_rawlog_id = start_rawlog.rawlog_id
        episode.end_rawlog_id = end_rawlog.rawlog_id
        episode.start_at = start_rawlog.occurred_at
        episode.end_at = end_rawlog.occurred_at
        episode.emotion_signal = emotion_signal or episode.emotion_signal
        episode.importance_score = max(
            score for score in [episode.importance_score, importance_score] if score is not None
        ) if episode.importance_score is not None or importance_score is not None else None
        episode.keywords = self._merge_keywords(episode.keywords, keywords)
        episode.metadata_json = {
            **existing_metadata,
            **(metadata or {}),
            "rawlog_count": len(ordered_rawlogs),
            "source_session_ids": session_ids,
            "contributing_session_ids": sorted(contributing),
            "merged": True,
        }

        updated = self.episode_repository.update(episode)
        self.episode_repository.replace_rawlog_links(
            updated.episode_id,
            [rawlog.rawlog_id for rawlog in ordered_rawlogs],
        )
        return updated

    def get_episode(self, episode_id: str) -> Episode | None:
        return self.episode_repository.get_by_id(episode_id)

    def require_episode(self, episode_id: str) -> Episode:
        episode = self.get_episode(episode_id)
        if episode is None:
            raise LookupError(f"Episode '{episode_id}' not found")
        return episode

    def list_episodes(self, limit: int = 50, source_session_id: str | None = None) -> list[Episode]:
        return self.episode_repository.list_episodes(limit=limit, source_session_id=source_session_id)

    def list_all_episodes(self, limit: int = 200) -> list[Episode]:
        return self.episode_repository.list_all(limit=limit)

    def update_episode(self, episode: Episode) -> Episode:
        return self.episode_repository.update(episode)

    def list_episode_rawlog_ids(self, episode_id: str) -> list[str]:
        return self.episode_repository.list_rawlog_ids(episode_id)

    def clear_session_episodes(self, session_id: str) -> None:
        self.episode_repository.delete_by_session_id(session_id)

    def get_by_rawlog_range(self, start_rawlog_id: str, end_rawlog_id: str) -> Episode | None:
        return self.episode_repository.get_by_rawlog_range(start_rawlog_id, end_rawlog_id)

    def delete_episode(self, episode_id: str) -> None:
        self.episode_repository.delete_by_id(episode_id)

    def _load_rawlogs(self, rawlog_ids: list[str]) -> list[RawLog]:
        return self.rawlog_service.list_rawlogs_by_ids(rawlog_ids)

    def _order_rawlogs(self, rawlogs: list[RawLog]) -> list[RawLog]:
        return sorted(rawlogs, key=lambda rawlog: (rawlog.occurred_at, rawlog.session_id, rawlog.sequence_no))

    def _merge_keywords(self, existing: list[str] | None, incoming: list[str] | None) -> list[str] | None:
        merged: list[str] = []
        for keyword in [*(existing or []), *(incoming or [])]:
            if keyword and keyword not in merged:
                merged.append(keyword)
        return merged or None

    def _merge_title(self, existing: str, incoming: str) -> str:
        existing = existing.strip()
        incoming = incoming.strip()
        if not incoming or incoming == existing or incoming in existing:
            return existing
        if existing in incoming:
            return incoming[:255]
        return f"{existing} / {incoming}"[:255]

    def _merge_summary(self, existing: str, incoming: str) -> str:
        existing = existing.strip()
        incoming = incoming.strip()
        if not incoming or incoming == existing or incoming in existing:
            return existing
        if existing in incoming:
            return incoming
        return f"{existing} {incoming}"
