from datetime import datetime

from sqlalchemy import DateTime, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class SearchLog(Base):
    __tablename__ = "search_logs"

    log_id: Mapped[str] = mapped_column(String, primary_key=True)
    query: Mapped[str] = mapped_column(Text)
    session_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    retrieved_json: Mapped[list | None] = mapped_column("retrieved", JSON, nullable=True)
    curated_json: Mapped[list | None] = mapped_column("curated", JSON, nullable=True)
    query_parse_json: Mapped[dict | None] = mapped_column("query_parse", JSON, nullable=True)
    used_episode_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    curator_reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
