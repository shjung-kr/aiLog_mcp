from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.session import Session as SessionModel


class SessionRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, session: SessionModel) -> SessionModel:
        self.db.add(session)
        self.db.flush()
        self.db.refresh(session)
        return session

    def get_by_id(self, session_id: str) -> SessionModel | None:
        stmt = select(SessionModel).where(SessionModel.session_id == session_id)
        return self.db.execute(stmt).scalar_one_or_none()

    def update_last_activity(self, session: SessionModel, occurred_at) -> SessionModel:
        session.last_activity_at = occurred_at
        self.db.add(session)
        self.db.flush()
        self.db.refresh(session)
        return session

    def list_sessions(self, limit: int = 50) -> list[SessionModel]:
        stmt = (
            select(SessionModel)
            .order_by(SessionModel.last_activity_at.desc(), SessionModel.started_at.desc())
            .limit(limit)
        )
        return list(self.db.execute(stmt).scalars().all())
