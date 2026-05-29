from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.search_log import SearchLog


class SearchRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, log: SearchLog) -> SearchLog:
        self.db.add(log)
        self.db.flush()
        return log

    def list_recent(self, limit: int = 50, session_id: str | None = None) -> list[SearchLog]:
        stmt = select(SearchLog)
        if session_id is not None:
            stmt = stmt.where(SearchLog.session_id == session_id)
        stmt = stmt.order_by(SearchLog.created_at.desc()).limit(limit)
        return list(self.db.execute(stmt).scalars().all())

    def get_by_id(self, log_id: str) -> SearchLog | None:
        stmt = select(SearchLog).where(SearchLog.log_id == log_id)
        return self.db.execute(stmt).scalar_one_or_none()
