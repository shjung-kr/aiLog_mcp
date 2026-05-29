import json

from app.db.models.long_term_memory import LongTermMemory
from app.db.repositories.long_term_memory_repository import LongTermMemoryRepository
from app.db.repositories.rawlog_repository import RawLogRepository
from app.llm.client import LLMClient
from app.utils.datetime import utc_now
from app.utils.ids import new_id

STYLE_MEMORY_TYPE = "user_style"
STYLE_MEMORY_ID = "user-style-profile"
UPDATE_EVERY_N_TURNS = 10


class UserStyleService:
    def __init__(
        self,
        ltm_repository: LongTermMemoryRepository,
        rawlog_repository: RawLogRepository,
        llm_client: LLMClient,
    ) -> None:
        self.ltm_repository = ltm_repository
        self.rawlog_repository = rawlog_repository
        self.llm_client = llm_client

    def get_style_profile(self) -> dict | None:
        record = self.ltm_repository.get_by_memory_type(STYLE_MEMORY_TYPE)
        if record is None:
            return None
        try:
            return json.loads(record.memory_text)
        except (json.JSONDecodeError, TypeError):
            return None

    def consume_update_notification(self) -> bool:
        """notified=False 인 경우 True 반환 후 플래그를 True로 소비."""
        record = self.ltm_repository.get_by_memory_type(STYLE_MEMORY_TYPE)
        if record is None:
            return False
        try:
            profile = json.loads(record.memory_text)
        except (json.JSONDecodeError, TypeError):
            return False
        if profile.get("notified", True):
            return False
        profile["notified"] = True
        record.memory_text = json.dumps(profile, ensure_ascii=False)
        self.ltm_repository.update(record)
        return True

    def should_update(self, session_id: str) -> bool:
        session_logs = self.rawlog_repository.list_by_session_and_speaker(
            session_id=session_id, speaker_type="user", limit=200
        )
        return len(session_logs) % UPDATE_EVERY_N_TURNS == 0 and len(session_logs) > 0

    def analyze_and_update(self, session_id: str | None = None) -> dict | None:
        # Use cross-session messages for richer style signal
        user_logs = self.rawlog_repository.list_by_speaker(speaker_type="user", limit=60)
        if len(user_logs) < 5:
            return None

        messages = [log.content for log in user_logs]
        existing = self.get_style_profile()

        new_profile = self.llm_client.analyze_user_style(messages, existing_profile=existing)
        if not new_profile:
            return None

        new_profile["updated_at"] = utc_now().isoformat()
        new_profile["notified"] = False
        profile_text = json.dumps(new_profile, ensure_ascii=False)

        record = self.ltm_repository.get_by_memory_type(STYLE_MEMORY_TYPE)
        if record is None:
            record = LongTermMemory(
                memory_id=STYLE_MEMORY_ID,
                episode_id="system",
                title="User Style Profile",
                memory_text=profile_text,
                memory_type=STYLE_MEMORY_TYPE,
                importance_score=1.0,
                created_at=utc_now(),
            )
            self.ltm_repository.create(record)
        else:
            record.memory_text = profile_text
            record.created_at = utc_now()
            self.ltm_repository.update(record)

        return new_profile
