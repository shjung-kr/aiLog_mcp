from datetime import datetime

from sqlalchemy import DateTime, JSON, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Turn(Base):
    __tablename__ = "turns"
    __table_args__ = (
        UniqueConstraint("start_rawlog_id", "end_rawlog_id", name="uq_turns_rawlog_range"),
    )

    turn_id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, index=True)
    start_rawlog_id: Mapped[str] = mapped_column(String, index=True)
    end_rawlog_id: Mapped[str] = mapped_column(String, index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    ended_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    metadata_json: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
