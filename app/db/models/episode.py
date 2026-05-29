from datetime import datetime

from sqlalchemy import DateTime, Float, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Episode(Base):
    __tablename__ = "episodes"

    episode_id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(String(255))
    summary: Mapped[str] = mapped_column(Text)
    episode_type: Mapped[str] = mapped_column(String(64), index=True)
    start_rawlog_id: Mapped[str] = mapped_column(String, index=True)
    end_rawlog_id: Mapped[str] = mapped_column(String, index=True)
    start_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    end_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    emotion_signal: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    importance_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    source_session_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    keywords: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
