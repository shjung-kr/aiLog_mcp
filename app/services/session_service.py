from app.db.models.session import Session
from app.db.repositories.session_repository import SessionRepository
from app.utils.datetime import utc_now
from app.utils.ids import new_id


class SessionService:
    def __init__(self, session_repository: SessionRepository) -> None:
        self.session_repository = session_repository

    def create_session(self, user_id: str | None = None, title: str | None = None) -> Session:
        now = utc_now()
        session = Session(
            session_id=new_id(),
            user_id=user_id,
            title=title,
            started_at=now,
            last_activity_at=now,
            status="active",
        )
        return self.session_repository.create(session)

    def get_session(self, session_id: str) -> Session | None:
        return self.session_repository.get_by_id(session_id)

    def require_session(self, session_id: str) -> Session:
        session = self.get_session(session_id)
        if session is None:
            raise LookupError(f"Session '{session_id}' not found")
        return session

    def update_last_activity(self, session_id: str, occurred_at) -> Session:
        session = self.require_session(session_id)
        return self.session_repository.update_last_activity(session, occurred_at)

    def list_sessions(self, limit: int = 50) -> list[Session]:
        return self.session_repository.list_sessions(limit=limit)
