from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.db.models.rawlog import RawLog


class RawLogRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, rawlog: RawLog) -> RawLog:
        self.db.add(rawlog)
        self.db.flush()
        self.db.refresh(rawlog)
        return rawlog

    def list_by_session_id(self, session_id: str) -> list[RawLog]:
        stmt = (
            select(RawLog)
            .where(RawLog.session_id == session_id)
            .order_by(RawLog.sequence_no.asc(), RawLog.occurred_at.asc(), RawLog.rawlog_id.asc())
        )
        return list(self.db.execute(stmt).scalars().all())

    def list_by_session_and_speaker(
        self, session_id: str, speaker_type: str, limit: int = 100
    ) -> list[RawLog]:
        stmt = (
            select(RawLog)
            .where(RawLog.session_id == session_id, RawLog.speaker_type == speaker_type)
            .order_by(RawLog.sequence_no.desc())
            .limit(limit)
        )
        return list(self.db.execute(stmt).scalars().all())

    def list_by_speaker(self, speaker_type: str, limit: int = 100) -> list[RawLog]:
        stmt = (
            select(RawLog)
            .where(RawLog.speaker_type == speaker_type)
            .order_by(RawLog.occurred_at.desc())
            .limit(limit)
        )
        return list(self.db.execute(stmt).scalars().all())

    def list_by_ids(self, rawlog_ids: list[str]) -> list[RawLog]:
        if not rawlog_ids:
            return []
        stmt = select(RawLog).where(RawLog.rawlog_id.in_(rawlog_ids))
        return list(self.db.execute(stmt).scalars().all())

    def get_latest_for_session(self, session_id: str) -> RawLog | None:
        stmt = (
            select(RawLog)
            .where(RawLog.session_id == session_id)
            .order_by(desc(RawLog.sequence_no), desc(RawLog.occurred_at), desc(RawLog.rawlog_id))
            .limit(1)
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def get_next_sequence_no(self, session_id: str) -> int:
        latest = self.get_latest_for_session(session_id)
        return 1 if latest is None else latest.sequence_no + 1
