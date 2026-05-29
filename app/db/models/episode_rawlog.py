from sqlalchemy import Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class EpisodeRawLog(Base):
    __tablename__ = "episode_rawlogs"
    __table_args__ = (
        UniqueConstraint("episode_id", "rawlog_id", name="uq_episode_rawlogs_episode_rawlog"),
    )

    episode_id: Mapped[str] = mapped_column(String, primary_key=True)
    rawlog_id: Mapped[str] = mapped_column(String, primary_key=True)
    position: Mapped[int] = mapped_column(Integer)
