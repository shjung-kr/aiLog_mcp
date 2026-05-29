from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.turn import Turn


class TurnRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, turn: Turn) -> Turn:
        self.db.add(turn)
        self.db.flush()
        self.db.refresh(turn)
        return turn

    def get_by_rawlog_range(self, start_rawlog_id: str, end_rawlog_id: str) -> Turn | None:
        stmt = select(Turn).where(
            Turn.start_rawlog_id == start_rawlog_id,
            Turn.end_rawlog_id == end_rawlog_id,
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def list_by_session_id(self, session_id: str) -> list[Turn]:
        stmt = (
            select(Turn)
            .where(Turn.session_id == session_id)
            .order_by(Turn.started_at.asc(), Turn.turn_id.asc())
        )
        return list(self.db.execute(stmt).scalars().all())
