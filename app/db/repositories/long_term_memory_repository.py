from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db.models.long_term_memory import LongTermMemory


class LongTermMemoryRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, memory: LongTermMemory) -> LongTermMemory:
        self.db.add(memory)
        self.db.flush()
        self.db.refresh(memory)
        return memory

    def get_by_id(self, memory_id: str) -> LongTermMemory | None:
        stmt = select(LongTermMemory).where(LongTermMemory.memory_id == memory_id)
        return self.db.execute(stmt).scalar_one_or_none()

    def get_by_episode_id(self, episode_id: str) -> LongTermMemory | None:
        stmt = select(LongTermMemory).where(LongTermMemory.episode_id == episode_id)
        return self.db.execute(stmt).scalar_one_or_none()

    def get_by_memory_type(self, memory_type: str) -> LongTermMemory | None:
        stmt = select(LongTermMemory).where(LongTermMemory.memory_type == memory_type).limit(1)
        return self.db.execute(stmt).scalar_one_or_none()

    def list_all(self, limit: int = 200) -> list[LongTermMemory]:
        stmt = (
            select(LongTermMemory)
            .order_by(LongTermMemory.importance_score.desc().nullslast(), LongTermMemory.created_at.desc())
            .limit(limit)
        )
        return list(self.db.execute(stmt).scalars().all())

    def update(self, memory: LongTermMemory) -> LongTermMemory:
        self.db.add(memory)
        self.db.flush()
        self.db.refresh(memory)
        return memory

    def delete_by_id(self, memory_id: str) -> None:
        self.db.execute(delete(LongTermMemory).where(LongTermMemory.memory_id == memory_id))
        self.db.flush()
