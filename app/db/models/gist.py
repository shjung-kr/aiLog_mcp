from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Text, DateTime, Float, JSON
from app.db.base import Base

class Gist(Base):
    __tablename__ = "gists"

    gist_id: Mapped[str] = mapped_column(String, primary_key=True)
    start_rawlog_id: Mapped[str] = mapped_column(String, index=True)
    end_rawlog_id: Mapped[str] = mapped_column(String, index=True)
    title: Mapped[str] = mapped_column(String(255))
    gist_text: Mapped[str] = mapped_column(Text)
    topic: Mapped[str | None] = mapped_column(String(255), nullable=True)
    intent: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True))
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
