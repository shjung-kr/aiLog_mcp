from sqlalchemy import delete, distinct, func, select
from sqlalchemy.orm import Session

from app.db.models.episode import Episode
from app.db.models.episode_rawlog import EpisodeRawLog
from app.db.models.rawlog import RawLog


class EpisodeRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, episode: Episode) -> Episode:
        self.db.add(episode)
        self.db.flush()
        self.db.refresh(episode)
        return episode

    def get_by_id(self, episode_id: str) -> Episode | None:
        stmt = select(Episode).where(Episode.episode_id == episode_id)
        return self.db.execute(stmt).scalar_one_or_none()

    def list_episodes(self, limit: int = 50, source_session_id: str | None = None) -> list[Episode]:
        stmt = select(Episode)
        if source_session_id is not None:
            stmt = (
                stmt.join(EpisodeRawLog, EpisodeRawLog.episode_id == Episode.episode_id)
                .join(RawLog, RawLog.rawlog_id == EpisodeRawLog.rawlog_id)
                .where(RawLog.session_id == source_session_id)
                .distinct()
            )
        stmt = stmt.order_by(Episode.start_at.desc(), Episode.episode_id.desc()).limit(limit)
        return list(self.db.execute(stmt).scalars().all())

    def list_all(self, limit: int = 200) -> list[Episode]:
        stmt = select(Episode).order_by(Episode.start_at.desc(), Episode.episode_id.desc()).limit(limit)
        return list(self.db.execute(stmt).scalars().all())

    def update(self, episode: Episode) -> Episode:
        self.db.add(episode)
        self.db.flush()
        self.db.refresh(episode)
        return episode

    def delete_by_session_id(self, session_id: str) -> None:
        episode_ids = list(
            self.db.execute(
                select(distinct(EpisodeRawLog.episode_id))
                .join(RawLog, RawLog.rawlog_id == EpisodeRawLog.rawlog_id)
                .where(RawLog.session_id == session_id)
            ).scalars()
        )
        if not episode_ids:
            return
        self.db.execute(
            delete(EpisodeRawLog).where(
                EpisodeRawLog.episode_id.in_(episode_ids),
                EpisodeRawLog.rawlog_id.in_(select(RawLog.rawlog_id).where(RawLog.session_id == session_id)),
            )
        )
        orphan_episode_ids = list(
            self.db.execute(
                select(Episode.episode_id)
                .outerjoin(EpisodeRawLog, EpisodeRawLog.episode_id == Episode.episode_id)
                .where(Episode.episode_id.in_(episode_ids))
                .group_by(Episode.episode_id)
                .having(func.count(EpisodeRawLog.rawlog_id) == 0)
            ).scalars()
        )
        if orphan_episode_ids:
            self.db.execute(delete(Episode).where(Episode.episode_id.in_(orphan_episode_ids)))
        self.db.flush()

    def delete_by_id(self, episode_id: str) -> None:
        self.db.execute(delete(EpisodeRawLog).where(EpisodeRawLog.episode_id == episode_id))
        self.db.execute(delete(Episode).where(Episode.episode_id == episode_id))
        self.db.flush()

    def replace_rawlog_links(self, episode_id: str, rawlog_ids: list[str]) -> None:
        self.db.execute(delete(EpisodeRawLog).where(EpisodeRawLog.episode_id == episode_id))
        for position, rawlog_id in enumerate(rawlog_ids, start=1):
            self.db.add(EpisodeRawLog(episode_id=episode_id, rawlog_id=rawlog_id, position=position))
        self.db.flush()

    def get_by_rawlog_range(self, start_rawlog_id: str, end_rawlog_id: str) -> Episode | None:
        stmt = select(Episode).where(
            Episode.start_rawlog_id == start_rawlog_id,
            Episode.end_rawlog_id == end_rawlog_id,
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def list_rawlog_ids(self, episode_id: str) -> list[str]:
        stmt = (
            select(EpisodeRawLog.rawlog_id)
            .where(EpisodeRawLog.episode_id == episode_id)
            .order_by(EpisodeRawLog.position.asc())
        )
        return list(self.db.execute(stmt).scalars().all())

    def list_rawlogs(self, episode_id: str) -> list[RawLog]:
        stmt = (
            select(RawLog)
            .join(EpisodeRawLog, EpisodeRawLog.rawlog_id == RawLog.rawlog_id)
            .where(EpisodeRawLog.episode_id == episode_id)
            .order_by(EpisodeRawLog.position.asc())
        )
        return list(self.db.execute(stmt).scalars().all())
