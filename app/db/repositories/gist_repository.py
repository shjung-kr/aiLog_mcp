from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db.models.gist import Gist
from app.db.models.rawlog import RawLog


class GistRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create_many(self, gists: list[Gist]) -> list[Gist]:
        for gist in gists:
            self.db.add(gist)
        self.db.flush()
        for gist in gists:
            self.db.refresh(gist)
        return gists

    def list_by_session_id(self, session_id: str) -> list[Gist]:
        stmt = (
            select(Gist)
            .join(RawLog, RawLog.rawlog_id == Gist.start_rawlog_id)
            .where(RawLog.session_id == session_id)
            .order_by(Gist.created_at.asc(), Gist.gist_id.asc())
        )
        return list(self.db.execute(stmt).scalars().all())

    def delete_by_session_id(self, session_id: str) -> None:
        subq = select(RawLog.rawlog_id).where(RawLog.session_id == session_id)
        self.db.execute(delete(Gist).where(Gist.start_rawlog_id.in_(subq)))
        self.db.flush()

    def list_all(self, limit: int = 100) -> list[Gist]:
        stmt = select(Gist).order_by(Gist.created_at.desc(), Gist.gist_id.desc()).limit(limit)
        return list(self.db.execute(stmt).scalars().all())
