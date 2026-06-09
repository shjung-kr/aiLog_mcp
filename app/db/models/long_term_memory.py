from datetime import datetime

from sqlalchemy import DateTime, Float, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class LongTermMemory(Base):
    __tablename__ = "long_term_memories"

    memory_id: Mapped[str] = mapped_column(String, primary_key=True)
    episode_id: Mapped[str] = mapped_column(String, index=True)
    title: Mapped[str] = mapped_column(String(255))
    memory_text: Mapped[str] = mapped_column(Text)
    memory_type: Mapped[str] = mapped_column(String(64), index=True)
    importance_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
