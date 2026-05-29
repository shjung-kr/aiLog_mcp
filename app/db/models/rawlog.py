from datetime import datetime

from sqlalchemy import JSON, DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class RawLog(Base):
    __tablename__ = "raw_logs"
    __table_args__ = (
        UniqueConstraint("session_id", "sequence_no", name="uq_raw_logs_session_sequence"),
    )

    rawlog_id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, index=True)
    sequence_no: Mapped[int] = mapped_column(Integer, index=True)
    speaker_type: Mapped[str] = mapped_column(String(32), index=True)
    content: Mapped[str] = mapped_column(Text)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    message_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    reply_to_rawlog_id: Mapped[str | None] = mapped_column(String, nullable=True)
    source_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    stored_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
